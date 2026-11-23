"""Aggregate the run files of one model: final metrics, curves and gate statistics."""
import glob
import os
import statistics as st
from collections import Counter, defaultdict

from wsms.analysis.metrics import fa_causes, gate_stats, night_metrics, tag_by_age, wake_stats
from wsms.utils.io import read_json

KEYS = ["recall", "recall_plain", "recall_doc", "recall_user", "false_assert", "false_attrib",
        "fa_implant", "fa_implant_once", "fa_implant_persist", "attrib_implant", "fa_corrected",
        "fixed_corrected", "attrib_corrected", "fa_overwrite", "kept_overwrite", "fa_confab", "fa_real",
        "kept_real", "attrib_real", "real_control", "tag_kept", "tag_false", "unk_halluc", "ku_halluc"]
CURVES = ("recall", "false_assert", "unk_halluc", "real_control", "tag_kept")


def ms(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return [float("nan"), 0.0]
    return [st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0]


def load_runs(folder):
    """Run records of one model, grouped by condition name."""
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        if os.path.basename(path) == "summary.json":
            continue
        r = read_json(path)
        runs[r["name"]].append(r)
    return runs


def summarize(folder):
    """Mean and standard deviation over streams for every condition found in `folder`."""
    runs = load_runs(folder)
    summary = {"conditions": {}}
    for name, rs in runs.items():
        n = min(len(r["nights"]) for r in rs)
        nights = [d for d in range(1, n + 1) if night_metrics(rs[0], d) is not None]
        final = [night_metrics(r, n) for r in rs]
        cond = {"seeds": sorted(r["args"]["seed"] for r in rs), "nights": n,
                "final": {k: ms([m[k] for m in final]) for k in KEYS},
                "n_true": ms([m["n_true"] for m in final]), "n_opp": ms([m["n_opp"] for m in final]),
                "n_corrected": ms([m["n_corrected"] for m in final]),
                "pi_confab": ms([m["n_confab"] / m["n_opp"] for m in final]),
                "mmlu_base": ms([r["mmlu_base"] for r in rs]), "mmlu_final": ms([r["mmlu_final"] for r in rs]),
                "written": ms([sum(x["n_accept"] for x in r["nights"]) for r in rs]),
                "curve_nights": nights,
                "curve": {k: [ms([night_metrics(r, d)[k] for r in rs]) for d in nights] for k in CURVES}}
        dec, sup, veto, age = defaultdict(Counter), Counter(), Counter(), defaultdict(Counter)
        reader = Counter()
        for r in rs:
            lab, d = wake_stats(r)
            for k, v in d.items():
                dec[k] += v
            for k, v in lab.items():           # what the awake reader answered to misleading episodes
                grp = "confab" if k.startswith("confab") else "real" if k.startswith("real") else None
                if grp:
                    reader[grp] += sum(v.values())
                    reader[grp + "_false"] += v.get("false", 0)
            s, v = gate_stats(r)
            sup += s
            veto += v
            for b, c in tag_by_age(r).items():
                age[f"{b[0]}-{b[1]}"] += c
        causes = Counter()
        for r in rs:
            causes += fa_causes(r)
        cond["fa_causes"] = dict(causes)
        cond["reader"] = dict(reader)
        cond["decisions"] = {k: dict(v) for k, v in dec.items()}
        cond["support_check"] = {f"{a}|{b}": x for (a, b), x in sup.items()}
        cond["veto_check"] = {f"{a}|{b}": x for (a, b), x in veto.items()}
        cond["tag_by_age"] = {k: dict(v) for k, v in age.items()}
        summary["conditions"][name] = cond
    allr = [r for rs in runs.values() for r in rs]
    summary["base"] = {
        "unk_halluc": ms([1 - sum(c == 3 for c in r["night0"]["unk_codes"]) / len(r["night0"]["unk_codes"]) for r in allr]),
        "real_ok": ms([sum(c == 1 for c in r["night0"]["real_codes"]) / len(r["night0"]["real_codes"]) for r in allr]),
        "mmlu": ms([r["mmlu_base"] for r in allr]), "real_known": ms([r["real_known"] for r in allr])}
    summary["model"] = allr[0]["args"]["model"]
    summary["helper"] = allr[0].get("helper")
    summary["n_trainable"] = allr[0]["n_trainable"]
    return summary


def format_table(summary):
    """Lines of the result table of one model."""
    lines = []
    pct = lambda v: f"{100 * v[0]:5.1f}±{100 * v[1]:4.1f}"
    lines.append(f"model {summary['model']} | helper {summary['helper']} | base: MMLU {100 * summary['base']['mmlu'][0]:.1f}, "
                 f"halluc. on unknown {100 * summary['base']['unk_halluc'][0]:.0f}%")
    hdr = ["condition", "n", "Recall", "R-plain", "R-doc", "R-user", "FalseAss", "Attrib", "Implant", "Corrctd",
           "Overwr", "Confab", "Real", "RealCtl", "TagKept", "TagFalse", "UnkHall", "MMLU"]
    lines.append(" | ".join(f"{h:>10s}" for h in hdr))
    for name, c in summary["conditions"].items():
        f = c["final"]
        row = [name[:10], str(len(c["seeds"]))] + [pct(f[k]) for k in (
            "recall", "recall_plain", "recall_doc", "recall_user", "false_assert", "false_attrib", "fa_implant",
            "fa_corrected", "fa_overwrite", "fa_confab", "fa_real", "real_control", "tag_kept", "tag_false",
            "unk_halluc")] + [f"{100 * c['mmlu_final'][0]:.1f}"]
        lines.append(" | ".join(f"{x:>10s}" for x in row))
    for name, c in summary["conditions"].items():
        if c["support_check"] or c["tag_by_age"]:
            lines.append(f"[{name}] causes {c['fa_causes']} | support {c['support_check']} | veto {c['veto_check']} | tag by age {c['tag_by_age']}")
    return lines
