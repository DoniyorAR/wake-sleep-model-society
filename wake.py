"""Pre-compute the awake phase of one stream.

The awake phase uses only frozen models, so it does not depend on what was learned during
sleep and can be computed once per (sleeper, helper, seed) and reused by every policy:
  * the reader (frozen sleeper) answers each probe question with the episode in context;
  * support check by the sleeper and by the helper model (P(yes));
  * closed-book answers of both frozen models to the questions behind user remarks;
  * anchor answers of the frozen sleeper to general questions (for homeostatic rehearsal).
"""
import argparse
import gc
import json
import os
import random

import torch

from engine import Engine, closed_book, open_book, support_prompt
from world import REAL_FACTS, World, has, real_screen_questions


def build_anchors(eng, path, n=600, seed=7):
    from datasets import load_dataset
    try:
        qs = [x["question"] for x in load_dataset("stanfordnlp/web_questions", split="train")]
    except Exception:
        qs = [x["question"] for x in load_dataset("google-research-datasets/nq_open", split="train[:6000]")]
    banned = set()
    for s, _, tv, fv, tk, fk in REAL_FACTS:
        banned.update([tk, fk])
        banned.update(w.strip('"').lower() for w in s.split() if w[:1].isupper() or w[:1] == '"')
    banned.discard("the")
    qs = sorted({q.strip() for q in qs if 20 <= len(q) <= 90})
    qs = [q for q in qs if not any(b and b in q.lower() for b in banned)]
    random.Random(seed).shuffle(qs)
    qs = [q[0].upper() + q[1:] + ("" if q.endswith("?") else "?") for q in qs[:n]]
    ans, _ = eng.generate([closed_book(q) for q in qs], adapter=False)
    json.dump([{"q": q, "a": a} for q, a in zip(qs, ans) if a], open(path, "w"), indent=0)
    return len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleeper", required=True)
    ap.add_argument("--helper", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", default=None)
    args = ap.parse_args()

    w = World(args.seed, days=args.days)
    eng = Engine(args.sleeper)
    ans, _ = eng.generate([closed_book(q) for q in real_screen_questions()], adapter=False)
    known = [i for i, (a, rf) in enumerate(zip(ans, REAL_FACTS)) if has(a, rf[4]) and not has(a, rf[5])]
    w.set_real(known)
    cands = [(d, ep, pr) for d in range(1, args.days + 1) for ep in w.schedule[d] for pr in ep["probes"]]
    answers, _ = eng.generate([open_book(ep["text"], pr["question"]) for _, ep, pr in cands], adapter=False)
    answers = [a[:160] for a in answers]
    sup = [support_prompt(ep["text"], pr["question"], a) for (_, ep, pr), a in zip(cands, answers)]
    p_self = eng.yes_prob(sup)
    users = [i for i, (_, ep, _) in enumerate(cands) if ep["source"] == "user"]
    cb_q = [closed_book(cands[i][2]["question"]) for i in users]
    cb_self, cf_self = eng.generate(cb_q, adapter=False)
    if args.anchors and not os.path.exists(args.anchors):
        print("anchors:", build_anchors(eng, args.anchors))
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    p_help = cb_help = cf_help = None
    if args.helper:
        h = Engine(args.helper)
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
    json.dump({"sleeper": args.sleeper, "helper": args.helper, "seed": args.seed, "days": args.days,
               "known": known, "cands": rows}, open(args.out, "w"))
    print(f"wake cache {args.out}: {len(rows)} candidates, {len(users)} from user remarks, "
          f"{len(known)} known real facts")


if __name__ == "__main__":
    main()
