"""Aggregate the run files of one model: writes <folder>/summary.json and prints the result table.

example: python scripts/aggregate_results.py results/qwen2.5-1.5b > results/table_qwen2.5-1.5b.txt
"""
import argparse
import os

from wsms.analysis.aggregate import format_table, summarize
from wsms.utils.io import write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder with the run files of one model")
    args = ap.parse_args()
    summary = summarize(args.folder)
    write_json(summary, os.path.join(args.folder, "summary.json"), indent=1)
    print("\n".join(format_table(summary)))


if __name__ == "__main__":
    main()
