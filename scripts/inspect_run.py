"""Print the metrics, gate decisions and tag statistics of one or more run files.

example: python scripts/inspect_run.py results/qwen3-4b/wsms_s1.json
"""
import argparse

from wsms.analysis.metrics import gate_stats, night_metrics, tag_by_age, wake_stats
from wsms.utils.io import read_json


def report(paths):
    for path in paths:
        run = read_json(path)
        n = len(run["nights"])
        print("=" * 110)
        print(path.split("/")[-1], "| nights", n, "| mmlu", run["mmlu_base"], "->", run["mmlu_final"])
        for k in range(1, n + 1):
            m = night_metrics(run, k)
            if m is None or (k % 6 and k != n):
                continue
            print(f"  night {k:2d}: recall {m['recall']:.3f} (plain {m['recall_plain']:.3f}; doc {m['recall_doc']:.3f} "
                  f"user {m['recall_user']:.3f}) | false-assert {m['false_assert']:.3f} [implant {m['fa_implant']:.2f} "
                  f"corrected {m['fa_corrected']:.2f} overwrite {m['fa_overwrite']:.2f} confab {m['fa_confab']:.2f} "
                  f"real {m['fa_real']:.2f}] attrib {m['false_attrib']:.3f} | tag kept {m['tag_kept']:.2f} "
                  f"false tag {m['tag_false']:.2f} | real ctl {m['real_control']:.2f} | unk halluc "
                  f"{m['unk_halluc']:.2f} ku {m['ku_halluc']:.2f}")
        lab, dec = wake_stats(run)
        for role in sorted(dec):
            print(f"    {role:24s} {dict(lab[role])} -> {dict(dec[role])}")
        sup, veto = gate_stats(run)
        print("  support:", dict(sup), "| veto:", dict(veto))
        print("  tag by age:", {f"{a}-{b}": dict(c) for (a, b), c in tag_by_age(run).items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run files")
    report(ap.parse_args().runs)


if __name__ == "__main__":
    main()
