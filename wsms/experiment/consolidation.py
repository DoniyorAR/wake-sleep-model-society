"""One wake-sleep run: N simulated days under one consolidation policy.

Every day the candidate memories of the awake phase (read from a cache, see `wake.py`) pass the
write gate; at night the accepted ones are trained into the LoRA adapter together with replay and
homeostatic rehearsal; after every `eval_every`-th night all questions are asked without context.
"""
import os
import random
from dataclasses import asdict, dataclass

from wsms.data.world import World
from wsms.memory.gate import WriteGate, accepted
from wsms.memory.rehearsal import RefusalSampler, sample_anchors, sample_replay
from wsms.models.prompts import closed_book
from wsms.utils.io import read_json, write_json
from wsms.utils.text import CODE_LABEL, classify, has, is_abstain

FACT_FIELDS = ("fid", "kind", "day", "rel", "attack", "attack_day", "mode", "remention_day", "confirm_day",
               "correct_day", "true", "false")


@dataclass
class RunConfig:
    model: str                       # sleeping model
    cache: str                       # awake-phase cache of the stream
    anchors: str | None = None       # anchor answers for homeostatic rehearsal
    policy: str = "naive"            # naive | doc_only | oracle | gate
    support: str = "society"         # society | self | none
    veto: str = "society"            # society | self | ledger | none
    hearsay: str = "tag"             # tag | quarantine | plain
    k: int = 2                       # mentions needed when hearsay = quarantine
    tau: float = 0.9                 # confidence needed for a self veto
    refuse: float = 0.0              # refusal examples, as a share of the night's new examples
    anchor: float = 0.0              # anchor examples, as a share of the night's new examples
    days: int = 30
    seed: int = 0
    epochs: int = 4
    lr: float = 2e-4
    bs: int = 8
    r: int = 16                      # LoRA rank
    replay: float = 1.0              # replayed examples, as a share of the night's new examples
    eval_every: int = 3
    mmlu_n: int = 300
    out: str = "results"
    name: str | None = None          # name of the condition; defaults to the policy


def build_candidates(world, day, rows):
    """Candidate memories of one day: every probe of every episode with the cached awake answer."""
    cands = [{"ep": ep, "pr": pr} for ep in world.schedule[day] for pr in ep["probes"]]
    assert len(rows) == len(cands), (day, len(rows), len(cands))
    for c, row in zip(cands, rows):
        pr = c["pr"]
        c["rec"] = world.real[pr["fid"]] if pr["real"] else world.facts[pr["fid"]]
        c["key"] = (pr["real"], pr["fid"])
        c["answer"], c["row"] = row["answer"], row
        c["label"] = CODE_LABEL[classify(c["answer"], c["rec"]["true_key"], c["rec"]["false_key"])]
        c["stated_ok"] = bool(pr["stated_key"]) and has(c["answer"], pr["stated_key"])
        c["target"], c["level"], c["why"] = c["answer"], "fact", ""
    return cands


def evaluate(engine, world, day):
    """Closed-book test of everything the stream has contained up to `day`. Returns codes and answers."""
    qs, meta = [], []
    for f in world.facts:
        if f["day"] <= day:
            qs.append(f["test_q"])
            meta.append(("f", f["fid"], f["true_key"], f["false_key"]))
    for r in world.real:
        qs.append(r["test_q"])
        meta.append(("r", r["rid"], r["true_key"], r["false_key"]))
    for i, u in enumerate(world.unknown):
        qs.append(u["test_q"])
        meta.append(("u", i, None, None))
    for i, u in enumerate(world.ku):
        if u["day"] <= day:
            qs.append(u["test_q"])
            meta.append(("k", i, None, None))
    answers, _ = engine.generate([closed_book(q) for q in qs], max_new_tokens=32)
    fc, rc = [-1] * len(world.facts), [0] * len(world.real)
    uc, kc = [0] * len(world.unknown), [-1] * len(world.ku)
    fa, ra, ua = [""] * len(world.facts), [""] * len(world.real), [""] * len(world.unknown)
    for (kind, i, tk, fk), a in zip(meta, answers):
        if kind == "f":
            fc[i], fa[i] = classify(a, tk, fk), a
        elif kind == "r":
            rc[i], ra[i] = classify(a, tk, fk), a
        elif kind == "u":
            uc[i], ua[i] = (3 if is_abstain(a) else 0), a
        else:
            kc[i] = 3 if is_abstain(a) else 0
    return ({"fact_codes": fc, "real_codes": rc, "unk_codes": uc, "ku_codes": kc},
            {"fact_answers": fa, "real_answers": ra, "unk_answers": ua})


def _progress(name, cfg, world, day, n_cands, n_accept, loss, codes):
    msg = (f"[{name} s{cfg.seed}] day {day:2d} cands {n_cands:2d} acc {n_accept:2d} "
           f"loss {loss if loss is None else round(loss, 3)}")
    if codes:
        fc = codes["fact_codes"]
        tr = [fc[f["fid"]] in (1, 4) for f in world.facts if f["day"] <= day and f["kind"] in ("doc", "user_true")]
        fm = [fc[f["fid"]] == 2 for f in world.facts if f["day"] <= day and f["false_key"]
              and (f["kind"] != "doc" or (f["attack_day"] or 99) <= day)]
        msg += (f" | recall {sum(tr) / max(1, len(tr)):.3f} false-assert {sum(fm) / max(1, len(fm)):.3f}"
                f" unk-abstain {sum(c == 3 for c in codes['unk_codes']) / len(codes['unk_codes']):.2f}"
                f" real-ok {sum(c == 1 for c in codes['real_codes']) / len(codes['real_codes']):.2f}")
    print(msg, flush=True)


def run_stream(cfg, engine=None, mmlu_items=None, save=True, verbose=True):
    """Run one stream under one policy and return the result record (also saved as JSON).

    `engine` and `mmlu_items` can be passed in; by default the model of `cfg` and MMLU are loaded.
    """
    name = cfg.name or cfg.policy
    cache = read_json(cfg.cache)
    assert cache["seed"] == cfg.seed and cache["days"] >= cfg.days
    world = World(cfg.seed, days=cache["days"])
    world.set_real(cache["known"])
    by_day = {}
    for row in cache["cands"]:
        by_day.setdefault(row["day"], []).append(row)
    anchors = read_json(cfg.anchors) if cfg.anchors and cfg.anchor > 0 else []

    if engine is None:
        from wsms.models.engine import Engine
        engine = Engine(cfg.model, r=cfg.r, alpha=2 * cfg.r, lr=cfg.lr, seed=cfg.seed)
    rng = random.Random(1000 + cfg.seed)
    refusals = RefusalSampler(world, rng)
    if mmlu_items is None:
        mmlu_items = []
        if cfg.mmlu_n > 0:
            from wsms.data.benchmarks import load_mmlu
            mmlu_items = load_mmlu(cfg.mmlu_n)
    mmlu_base = engine.mmlu(mmlu_items, adapter=False) if mmlu_items else None

    codes0, texts0 = evaluate(engine, world, 0)
    result = {"args": asdict(cfg), "name": name, "n_trainable": engine.n_trainable,
              "real_known": len(cache["known"]), "mmlu_base": mmlu_base, "helper": cache.get("helper"),
              "facts": [{k: f.get(k) for k in FACT_FIELDS} for f in world.facts],
              "real": [{k: r[k] for k in ("rid", "subject", "attack", "attack_day")} for r in world.real],
              "night0": codes0, "night0_unk_answers": texts0["unk_answers"][:10], "nights": []}

    gate = WriteGate(cfg.policy, cfg.support, cfg.veto, cfg.hearsay, cfg.k, cfg.tau)
    gated = cfg.policy == "gate"
    pool_list, pool_map = [], {}     # written memories: append-only (baselines) or keyed (gate)
    for day in range(1, cfg.days + 1):
        # ------------------------------------------------------------ wake (from cache) and gate
        cands = build_candidates(world, day, by_day.get(day, []))
        for c in cands:
            gate.decide(c, day)
        written = accepted(cands)
        # ------------------------------------------------------------ sleep
        new = [(q, c["target"]) for c in written for q in c["rec"]["train_q"]]
        old = list(pool_map.values()) if gated else pool_list
        replay = sample_replay(rng, old, int(round(cfg.replay * len(new)))) if old and new else []
        refusals.note_written(written)
        extra = []
        if new and cfg.refuse > 0:
            extra += refusals.sample(max(1, round(cfg.refuse * len(new))))
        if new and anchors:
            extra += sample_anchors(rng, anchors, max(1, round(cfg.anchor * len(new))))
        loss, steps = engine.sleep(new + replay + extra, epochs=cfg.epochs, bs=cfg.bs)
        for c in written:
            if gated:
                pool_map[c["key"]] = (c["rec"]["train_q"], c["target"])
            else:
                pool_list.append((c["rec"]["train_q"], c["target"]))
        gate.commit(written)
        # ------------------------------------------------------------ morning test
        full = day % cfg.eval_every == 0 or day == cfg.days
        codes, texts = evaluate(engine, world, day) if full else (None, None)
        night = {"day": day, "n_cands": len(cands), "n_accept": len(written), "n_new": len(new),
                 "n_replay": len(replay), "n_extra": len(extra), "loss": loss, "steps": steps,
                 "wake": [{"role": c["pr"]["role"], "real": c["pr"]["real"], "fid": c["pr"]["fid"],
                           "rem": c["pr"]["remention"], "src": c["ep"]["source"], "label": c["label"],
                           "stated_ok": c["stated_ok"], "decision": c["decision"], "why": c["why"]}
                          for c in cands]}
        if full:
            night.update(codes)
        result["nights"].append(night)
        if verbose:
            _progress(name, cfg, world, day, len(cands), len(written), loss, codes)
        if day == cfg.days:
            result["final_texts"] = texts
    result["mmlu_final"] = engine.mmlu(mmlu_items, adapter=True) if mmlu_items else None
    if save:
        out_path = os.path.join(cfg.out, f"{name}_s{cfg.seed}.json")
        write_json(result, out_path)
        if verbose:
            print(f"[{name} s{cfg.seed}] done | mmlu {mmlu_base} -> {result['mmlu_final']} | saved {out_path}",
                  flush=True)
    return result
