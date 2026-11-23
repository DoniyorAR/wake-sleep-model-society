"""Draw the figures from the summaries of both models.

example: python scripts/make_figures.py results results/figures
"""
import argparse

from wsms.analysis.figures import make_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="results folder with one sub-folder per model")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    font = make_all(args.root, args.out_dir)
    print("figures written to", args.out_dir, "| font:", font)


if __name__ == "__main__":
    main()
