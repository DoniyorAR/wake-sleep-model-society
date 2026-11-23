"""One wake-sleep run: N simulated days under one consolidation policy.

Policies
  naive     every awake answer is written to the weights as a plain fact
  doc_only  only answers taken from documents are written
  oracle    exactly the answers that are true in the world are written (upper bound)
  gate      trust-layered consolidation (this work):
              support   the source must state the answer        (society | self | none)
              veto      no conflict with ledger / known facts    (society | self | ledger | none)
              hearsay   what happens to an unconfirmed user claim (tag | quarantine | plain)
              refuse / anchor   homeostatic rehearsal doses (share of the night's new examples)

The awake phase is read from a cache written by wake.py (it only uses frozen models).
"""
import argparse
import json
import os
import random
import time

import torch

from engine import Engine, closed_book, load_mmlu
from world import (HEARSAY_PREFIX, REL_NAMES, RELATIONS, World, _F_A, _F_B, _S_A, _S_B, classify,
                   has, is_abstain, same_answer)

ABSTAIN_TEXT = "I do not know."
LABEL = {1: "true", 2: "false", 3: "abstain", 0: "other", 4: "true", 5: "false", 6: "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--policy", default="naive", choices=["naive", "doc_only", "oracle", "gate"])
    ap.add_argument("--support", default="society", choices=["society", "self", "none"])
    ap.add_argument("--veto", default="society", choices=["society", "self", "ledger", "none"])
    ap.add_argument("--hearsay", default="tag", choices=["tag", "quarantine", "plain"])
    ap.add_argument("--k", type=int, default=2, help="mentions needed when hearsay=quarantine")
    ap.add_argument("--tau", type=float, default=0.9, help="confidence needed for a self veto")
    ap.add_argument("--refuse", type=float, default=0.0)
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--replay", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=3)
    ap.add_argument("--mmlu_n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or args.policy
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{name}_s{args.seed}.json")
    t_start = time.time()

    cache = json.load(open(args.cache))
    assert cache["seed"] == args.seed and cache["days"] >= args.days
    w = World(args.seed, days=cache["days"])
    w.set_real(cache["known"])
    by_day = {}
    for row in cache["cands"]:
        by_day.setdefault(row["day"], []).append(row)
    anchors = json.load(open(args.anchors)) if args.anchors and args.anchor > 0 else []

    eng = Engine(args.model, r=args.r, alpha=2 * args.r, lr=args.lr, seed=args.seed)
    rng = random.Random(1000 + args.seed)
    used_names = set(w._names[:w._name_i])

    def fresh_name():
        while True:
            n = f"{rng.choice(_F_A)}{rng.choice(_F_B)} {rng.choice(_S_A)}{rng.choice(_S_B)}"
            if n not in used_names:
                return n

    ku_set = {(k["person"], k["rel"]) for k in w.ku}
    consolidated, known_people = set(), []

    def refusals(n):
        """Refusal practice: unseen names, and unconsolidated attributes of known people."""
        out = []
        for i in range(n):
            person, rel = fresh_name(), rng.choice(REL_NAMES)
            if i % 2 == 1 and known_people:
                for _ in range(20):
                    p2, r2 = rng.choice(known_people), rng.choice(REL_NAMES)
                    if (p2, r2) not in consolidated and (p2, r2) not in ku_set:
                        person, rel = p2, r2
                        break
            out.append((rng.choice(RELATIONS[rel]["train_q"]).format(p=person), ABSTAIN_TEXT))
        return out

    mmlu_items = load_mmlu(args.mmlu_n) if args.mmlu_n > 0 else []
    mmlu_base = eng.mmlu(mmlu_items, adapter=False) if mmlu_items else None

    def evaluate(day):
        qs, meta = [], []
        for f in w.facts:
            if f["day"] <= day:
                qs.append(f["test_q"])
                meta.append(("f", f["fid"], f["true_key"], f["false_key"]))
        for r in w.real:
            qs.append(r["test_q"])
            meta.append(("r", r["rid"], r["true_key"], r["false_key"]))
        for i, u in enumerate(w.unknown):
            qs.append(u["test_q"])
            meta.append(("u", i, None, None))
        for i, u in enumerate(w.ku):
            if u["day"] <= day:
                qs.append(u["test_q"])
                meta.append(("k", i, None, None))
        answers, _ = eng.generate([closed_book(q) for q in qs], max_new_tokens=32)
        fc, rc, uc, kc = [-1] * len(w.facts), [0] * len(w.real), [0] * len(w.unknown), [-1] * len(w.ku)
        fa, ra, ua = [""] * len(w.facts), [""] * len(w.real), [""] * len(w.unknown)
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

    codes0, texts0 = evaluate(0)
    keep = ("fid", "kind", "day", "rel", "attack", "attack_day", "mode", "remention_day", "confirm_day",
            "correct_day", "true", "false")
    result = {"args": vars(args), "name": name, "n_trainable": eng.n_trainable,
              "real_known": len(cache["known"]), "mmlu_base": mmlu_base, "helper": cache.get("helper"),
              "facts": [{k: f.get(k) for k in keep} for f in w.facts],
              "real": [{k: r[k] for k in ("rid", "subject", "attack", "attack_day")} for r in w.real],
              "night0": codes0, "night0_unk_answers": texts0["unk_answers"][:10], "nights": []}

    gated = args.policy == "gate"
    pool_list, pool_map = [], {}     # written memories: append-only (baselines) or keyed (gate)
    ledger = {}                      # (real?, fid) -> {"answer", "level"}
    evidence = {}
    for day in range(1, args.days + 1):
        t0 = time.time()
        # ------------------------------------------------------------ WAKE (from cache)
        cands = [{"ep": ep, "pr": pr} for ep in w.schedule[day] for pr in ep["probes"]]
        rows = by_day.get(day, [])
        assert len(rows) == len(cands), (day, len(rows), len(cands))
        for c, row in zip(cands, rows):
            pr = c["pr"]
            c["rec"] = w.real[pr["fid"]] if pr["real"] else w.facts[pr["fid"]]
            c["key"] = (pr["real"], pr["fid"])
            c["answer"], c["row"] = row["answer"], row
            c["label"] = LABEL[classify(c["answer"], c["rec"]["true_key"], c["rec"]["false_key"])]
            c["stated_ok"] = bool(pr["stated_key"]) and has(c["answer"], pr["stated_key"])
            c["target"], c["level"], c["why"] = c["answer"], "fact", ""
        # ------------------------------------------------------------ GATE
        for c in cands:
            row, a, src = c["row"], c["answer"], c["ep"]["source"]
            if args.policy == "naive":
                c["decision"] = "accept_fact"
            elif args.policy == "doc_only":
                c["decision"] = "accept_fact" if src == "doc" else "reject_source"
            elif args.policy == "oracle":
                c["decision"] = "accept_fact" if c["label"] == "true" else "reject_oracle"
            else:
                ok = True
                if args.support != "none":
                    ok = row["p_self"] >= 0.5
                    if args.support == "society" and row["p_helper"] is not None:
                        ok = ok and row["p_helper"] >= 0.5
                led = ledger.get(c["key"])
                if is_abstain(a):
                    c["decision"] = "reject_abstain"
                elif not ok:
                    c["decision"] = "reject_unsupported"
                elif src == "doc":
                    c["decision"] = "accept_fact"
                    if led and not same_answer(led["answer"], a):
                        c["why"] = "document replaces " + led["level"]
                    elif led and led["level"] == "hearsay":
                        c["why"] = "document confirms hearsay"
                elif led and led["level"] == "fact" and same_answer(led["answer"], a):
                    c["decision"] = "accept_fact"          # the user repeats a known fact
                else:
                    conflict = ""
                    if args.veto != "none" and led and led["level"] == "fact":
                        conflict = "ledger"
                    elif args.veto == "society":
                        s, h = row["cb_self"] or "", row["cb_helper"] or ""
                        if h and not is_abstain(s) and not is_abstain(h) and same_answer(s, h) \
                                and not same_answer(h, a):
                            conflict = "consensus"
                    elif args.veto == "self":
                        s = row["cb_self"] or ""
                        if not is_abstain(s) and (row["cf_self"] or 0) >= args.tau and not same_answer(s, a):
                            conflict = "self"
                    if conflict:
                        c["decision"], c["why"] = "veto", conflict
                    elif args.hearsay == "tag":
                        c["decision"], c["level"] = "accept_hearsay", "hearsay"
                        c["target"] = HEARSAY_PREFIX + a
                    elif args.hearsay == "plain":
                        c["decision"] = "accept_fact"
                    else:
                        ev = evidence.setdefault(c["key"], [])
                        entry = next((e for e in ev if same_answer(e["answer"], a)), None)
                        if entry is None:
                            entry = {"answer": a, "days": set()}
                            ev.append(entry)
                        entry["days"].add(day)
                        c["decision"] = "accept_fact" if len(entry["days"]) >= args.k else "quarantine"
        accepted = [c for c in cands if c["decision"].startswith("accept")]
        # ------------------------------------------------------------ SLEEP
        new = [(q, c["target"]) for c in accepted for q in c["rec"]["train_q"]]
        old = list(pool_map.values()) if gated else pool_list
        replay = []
        if old and new:
            for _ in range(int(round(args.replay * len(new)))):
                qs, tgt = rng.choice(old)
                replay.append((rng.choice(qs), tgt))
        for c in accepted:
            if "person" in c["rec"]:
                if c["rec"]["person"] not in known_people:
                    known_people.append(c["rec"]["person"])
                consolidated.add((c["rec"]["person"], c["rec"]["rel"]))
        extra = []
        if new and args.refuse > 0:
            extra += refusals(max(1, round(args.refuse * len(new))))
        if new and anchors:
            extra += [(x["q"], x["a"]) for x in rng.sample(anchors, min(len(anchors), max(1, round(args.anchor * len(new)))))]
        loss, steps = eng.sleep(new + replay + extra, epochs=args.epochs, bs=args.bs)
        for c in accepted:
            if gated:
                pool_map[c["key"]] = (c["rec"]["train_q"], c["target"])
            else:
                pool_list.append((c["rec"]["train_q"], c["target"]))
            ledger[c["key"]] = {"answer": c["answer"], "level": c["level"]}
        t2 = time.time()
        # ------------------------------------------------------------ MORNING TEST
        full = day % args.eval_every == 0 or day == args.days
        codes, texts = evaluate(day) if full else (None, None)
        t3 = time.time()
        night = {"day": day, "n_cands": len(cands), "n_accept": len(accepted), "n_new": len(new),
                 "n_replay": len(replay), "n_extra": len(extra), "loss": loss, "steps": steps,
                 "t_sleep": round(t2 - t0, 1), "t_eval": round(t3 - t2, 1),
                 "wake": [{"role": c["pr"]["role"], "real": c["pr"]["real"], "fid": c["pr"]["fid"],
                           "rem": c["pr"]["remention"], "src": c["ep"]["source"], "label": c["label"],
                           "stated_ok": c["stated_ok"], "decision": c["decision"], "why": c["why"]}
                          for c in cands]}
        result["nights"].append(night)
        msg = (f"[{name} s{args.seed}] day {day:2d} cands {len(cands):2d} acc {len(accepted):2d} "
               f"loss {loss if loss is None else round(loss, 3)} | {t2 - t0:.0f}+{t3 - t2:.0f}s")
        if full:
            night.update(codes)
            fc = codes["fact_codes"]
            tr = [fc[f["fid"]] in (1, 4) for f in w.facts if f["day"] <= day and f["kind"] in ("doc", "user_true")]
            fm = [fc[f["fid"]] == 2 for f in w.facts if f["day"] <= day and f["false_key"]
                  and (f["kind"] != "doc" or (f["attack_day"] or 99) <= day)]
            msg += (f" | recall {sum(tr) / max(1, len(tr)):.3f} false-assert {sum(fm) / max(1, len(fm)):.3f}"
                    f" unk-abstain {sum(c == 3 for c in codes['unk_codes']) / len(codes['unk_codes']):.2f}"
                    f" real-ok {sum(c == 1 for c in codes['real_codes']) / len(codes['real_codes']):.2f}")
        print(msg, flush=True)
        if day == args.days:
            result["final_texts"] = texts
    result["mmlu_final"] = eng.mmlu(mmlu_items, adapter=True) if mmlu_items else None
    result["minutes"] = round((time.time() - t_start) / 60, 1)
    result["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    with open(out_path, "w") as fh:
        json.dump(result, fh)
    print(f"[{name} s{args.seed}] done in {result['minutes']} min | mmlu {mmlu_base} -> "
          f"{result['mmlu_final']} | peak {result['peak_gb']} GB | saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
