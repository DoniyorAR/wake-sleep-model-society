"""Error rates of the two gate signals for one model pair, from the awake-phase caches.

example: python scripts/gate_error_rates.py q15 results/theory_q15.json 1 2 3
"""
import argparse

from wsms.analysis.gate_theory import format_report, gate_error_rates
from wsms.utils.io import write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="cache tag of the model pair: q15 or q4")
    ap.add_argument("out", help="JSON file to write")
    ap.add_argument("seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--wake-dir", default="results/wake")
    args = ap.parse_args()
    res = gate_error_rates(args.tag, args.seeds, args.wake_dir)
    write_json(res, args.out, indent=1)
    print("\n".join(format_report(args.tag, res)))


if __name__ == "__main__":
    main()
