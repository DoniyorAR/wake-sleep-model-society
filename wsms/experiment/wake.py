"""Awake phase of one stream, computed once and reused by every policy.

The awake phase uses only frozen models, so it does not depend on what was learned during sleep:
  * the reader (frozen sleeping model) answers each probe question with the episode in context;
  * support check by the sleeping model and by the helper model (P(yes));
  * closed-book answers of both frozen models to the questions behind user remarks;
  * anchor answers of the frozen sleeping model to general questions (for homeostatic rehearsal).
"""
import gc
import os

from wsms.data.benchmarks import anchor_questions
from wsms.data.world import REAL_FACTS, World, real_screen_questions
from wsms.models.prompts import closed_book, open_book, support_prompt
from wsms.utils.io import write_json
from wsms.utils.text import has


def build_anchors(engine, path, n=600, seed=7):
    """Answers of the frozen model to general questions; returns the number of questions asked."""
    qs = anchor_questions(n, seed)
    ans, _ = engine.generate([closed_book(q) for q in qs], adapter=False)
    write_json([{"q": q, "a": a} for q, a in zip(qs, ans) if a], path, indent=0)
    return len(qs)


def build_wake_cache(sleeper, helper, seed, days, out, anchors=None):
    import torch

    from wsms.models.engine import Engine

    w = World(seed, days=days)
    eng = Engine(sleeper)
    ans, _ = eng.generate([closed_book(q) for q in real_screen_questions()], adapter=False)
    known = [i for i, (a, rf) in enumerate(zip(ans, REAL_FACTS)) if has(a, rf[4]) and not has(a, rf[5])]
    w.set_real(known)
    cands = [(d, ep, pr) for d in range(1, days + 1) for ep in w.schedule[d] for pr in ep["probes"]]
    answers, _ = eng.generate([open_book(ep["text"], pr["question"]) for _, ep, pr in cands], adapter=False)
    answers = [a[:160] for a in answers]
    sup = [support_prompt(ep["text"], pr["question"], a) for (_, ep, pr), a in zip(cands, answers)]
    p_self = eng.yes_prob(sup)
    users = [i for i, (_, ep, _) in enumerate(cands) if ep["source"] == "user"]
    cb_q = [closed_book(cands[i][2]["question"]) for i in users]
    cb_self, cf_self = eng.generate(cb_q, adapter=False)
    if anchors and not os.path.exists(anchors):
        print("anchors:", build_anchors(eng, anchors))
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    p_help = cb_help = cf_help = None
    if helper:
        h = Engine(helper)
        p_help = h.yes_prob(sup)
        cb_help, cf_help = h.generate(cb_q, adapter=False)
    rows = [{"day": d, "answer": a, "p_self": round(p, 4), "p_helper": None, "cb_self": None,
             "cf_self": None, "cb_helper": None, "cf_helper": None}
            for (d, _, _), a, p in zip(cands, answers, p_self)]
    if p_help:
        for r, p in zip(rows, p_help):
            r["p_helper"] = round(p, 4)
    for j, i in enumerate(users):
        rows[i]["cb_self"], rows[i]["cf_self"] = cb_self[j][:100], round(cf_self[j], 3)
        if cb_help:
            rows[i]["cb_helper"], rows[i]["cf_helper"] = cb_help[j][:100], round(cf_help[j], 3)
    write_json({"sleeper": sleeper, "helper": helper, "seed": seed, "days": days,
                "known": known, "cands": rows}, out)
    print(f"wake cache {out}: {len(rows)} candidates, {len(users)} from user remarks, "
          f"{len(known)} known real facts")
    return rows
