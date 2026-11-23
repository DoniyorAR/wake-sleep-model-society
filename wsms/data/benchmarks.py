"""External data: MMLU items for the general-ability probe and anchor questions for rehearsal."""
import random

from wsms.data.world import REAL_FACTS


def load_mmlu(n=300, seed=1234):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [(ds[i]["question"], ds[i]["choices"], int(ds[i]["answer"])) for i in idx[:n]]


def anchor_questions(n=600, seed=7):
    """General questions (WebQuestions) that do not touch the real-world facts used in the stream."""
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
    return [q[0].upper() + q[1:] + ("" if q.endswith("?") else "?") for q in qs[:n]]
