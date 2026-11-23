"""Compute the awake phase of one stream (frozen models only) and save it as a cache.

example: python scripts/build_wake_cache.py --sleeper Qwen/Qwen2.5-1.5B-Instruct \
             --helper Qwen/Qwen3-4B-Instruct-2507 --seed 1 --out results/wake/q15_s1.json \
             --anchors results/wake/anchors_q15.json
"""
import argparse

from wsms.experiment.wake import build_wake_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sleeper", required=True, help="model that sleeps (also the reader)")
    ap.add_argument("--helper", default=None, help="second model of the society")
    ap.add_argument("--seed", type=int, required=True, help="seed of the stream")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", required=True, help="cache file to write")
    ap.add_argument("--anchors", default=None, help="anchor-answer file; built if it does not exist")
    args = ap.parse_args()
    build_wake_cache(args.sleeper, args.helper, args.seed, args.days, args.out, args.anchors)


if __name__ == "__main__":
    main()
