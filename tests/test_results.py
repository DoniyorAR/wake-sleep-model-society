"""The published result files agree with the code: the gate reproduces every logged decision,
and the summaries are what the metrics give for the run files."""
import glob
import json
import os

import pytest

from wsms.analysis.aggregate import summarize
from wsms.data.world import World
from wsms.experiment.consolidation import build_candidates
from wsms.memory.gate import WriteGate, accepted
from wsms.utils.io import read_json

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
RUNS = sorted(p for p in glob.glob(os.path.join(ROOT, "*", "*_s[0-9].json"))
              if os.path.basename(os.path.dirname(p)) != "wake")


@pytest.mark.parametrize("path", RUNS, ids=lambda p: os.path.relpath(p, ROOT))
def test_gate_reproduces_logged_decisions(path):
    run = read_json(path)
    a = run["args"]
    cache = read_json(os.path.join(ROOT, os.path.relpath(a["cache"], "results")))
    world = World(a["seed"], days=cache["days"])
    world.set_real(cache["known"])
    by_day = {}
    for row in cache["cands"]:
        by_day.setdefault(row["day"], []).append(row)
    gate = WriteGate(a["policy"], a["support"], a["veto"], a["hearsay"], a["k"], a["tau"])
    for night in run["nights"]:
        cands = build_candidates(world, night["day"], by_day.get(night["day"], []))
        for c in cands:
            gate.decide(c, night["day"])
        gate.commit(accepted(cands))
        assert [(c["decision"], c["why"]) for c in cands] == [(x["decision"], x["why"]) for x in night["wake"]]
        assert len(accepted(cands)) == night["n_accept"]


@pytest.mark.parametrize("model", ["qwen2.5-1.5b", "qwen3-4b"])
def test_summary_matches_run_files(model):
    folder = os.path.join(ROOT, model)
    stored = read_json(os.path.join(folder, "summary.json"))
    assert json.loads(json.dumps(summarize(folder))) == stored
