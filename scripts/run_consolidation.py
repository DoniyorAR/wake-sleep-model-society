"""Run one stream under one consolidation policy and save the result record.

example: python scripts/run_consolidation.py --model Qwen/Qwen2.5-1.5B-Instruct \
             --cache results/wake/q15_s1.json --anchors results/wake/anchors_q15.json \
             --policy gate --refuse 0.15 --anchor 0.3 --name wsms --seed 1 --out results/qwen2.5-1.5b
"""
import argparse

from wsms.experiment.consolidation import RunConfig, run_stream
from wsms.memory.gate import HEARSAY_MODES, POLICIES, SUPPORT_MODES, VETO_MODES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True, help="awake-phase cache (scripts/build_wake_cache.py)")
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--policy", default="naive", choices=POLICIES)
    ap.add_argument("--support", default="society", choices=SUPPORT_MODES)
    ap.add_argument("--veto", default="society", choices=VETO_MODES)
    ap.add_argument("--hearsay", default="tag", choices=HEARSAY_MODES)
    ap.add_argument("--k", type=int, default=2, help="mentions needed when hearsay=quarantine")
    ap.add_argument("--tau", type=float, default=0.9, help="confidence needed for a self veto")
    ap.add_argument("--refuse", type=float, default=0.0, help="refusal dose (share of new examples)")
    ap.add_argument("--anchor", type=float, default=0.0, help="anchor dose (share of new examples)")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--replay", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=3)
    ap.add_argument("--mmlu_n", type=int, default=300)
    ap.add_argument("--out", required=True, help="folder of the model's run files")
    ap.add_argument("--name", default=None, help="name of the condition (default: the policy)")
    run_stream(RunConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
