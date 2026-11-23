"""Measured error rates of the two gate signals, from the awake-phase caches (frozen models only).

  eps = P(check passes | answer NOT stated by the source)     (false accept)
  dlt = P(check blocks | answer stated by the source)         (false reject)
  veto: detection on false claims about known real-world facts, false alarms on new claims
"""
import os
from collections import Counter

from wsms.data.world import World
from wsms.utils.io import read_json
from wsms.utils.text import has, is_abstain, same_answer


def gate_error_rates(tag, seeds, wake_dir="results/wake"):
    """Error rates of the support signals and of the conflict veto for one model pair (cache tag)."""
    c = Counter()
    for seed in seeds:
        cache = read_json(os.path.join(wake_dir, f"{tag}_s{seed}.json"))
        w = World(seed, days=cache["days"])
        w.set_real(cache["known"])
        cands = [(ep, pr) for d in range(1, cache["days"] + 1) for ep in w.schedule[d] for pr in ep["probes"]]
        assert len(cands) == len(cache["cands"])
        for (ep, pr), r in zip(cands, cache["cands"]):
            a = r["answer"]
            stated = bool(pr["stated_key"]) and has(a, pr["stated_key"])
            if pr["role"] == "confab":
                c["confab"] += 1
                c["confab_bait"] += (not is_abstain(a))
            if not is_abstain(a):
                s1, s2 = r["p_self"] >= 0.5, r["p_helper"] >= 0.5
                grp = "st" if stated else "ns"
                c[grp] += 1
                c[grp + "_1"] += s1
                c[grp + "_2"] += s2
                c[grp + "_12"] += s1 and s2
            if ep["source"] == "user" and stated:
                s, h = r["cb_self"], r["cb_helper"]
                selfv = (not is_abstain(s)) and r["cf_self"] >= 0.9 and not same_answer(s, a)
                cons = (not is_abstain(s)) and (not is_abstain(h)) and same_answer(s, h) and not same_answer(h, a)
                if pr["real"]:
                    grp = "att"
                elif pr["role"].startswith("overwrite"):
                    continue
                else:
                    grp = "new"
                c[grp] += 1
                c[grp + "_self"] += selfv
                c[grp + "_cons"] += cons
    d = lambda a, b: c[a] / c[b] if c[b] else float("nan")
    res = {"counts": dict(c), "seeds": seeds, "helper": cache["helper"],
           "eps1": d("ns_1", "ns"), "eps2": d("ns_2", "ns"), "eps12": d("ns_12", "ns"),
           "dlt1": 1 - d("st_1", "st"), "dlt2": 1 - d("st_2", "st"), "dlt12": 1 - d("st_12", "st"),
           "bait": d("confab_bait", "confab"),
           "veto_det_self": d("att_self", "att"), "veto_det_cons": d("att_cons", "att"),
           "veto_fa_self": d("new_self", "new"), "veto_fa_cons": d("new_cons", "new")}
    return res


def format_report(tag, res):
    """Two lines of text for one model pair."""
    p = lambda k: f"{100 * res[k]:.1f}%"
    n = res["counts"]
    return [
        f"[{tag}] helper {res['helper']} | not stated n={n['ns']}: eps1 {p('eps1')} eps2 {p('eps2')} "
        f"both {p('eps12')} (product {100 * res['eps1'] * res['eps2']:.1f}%) | stated n={n['st']}: "
        f"dlt1 {p('dlt1')} dlt2 {p('dlt2')} both {p('dlt12')} | bait taken {p('bait')}",
        f"      veto on real-world false claims n={n['att']}: self {p('veto_det_self')} consensus "
        f"{p('veto_det_cons')} | false alarms on new claims n={n['new']}: self {p('veto_fa_self')} "
        f"consensus {p('veto_fa_cons')}"]
