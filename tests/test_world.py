from collections import Counter

from wsms.data.world import REAL_FACTS, World, value_key


def test_stream_size():
    w = World(1)
    assert len(w.facts) == 630
    assert Counter(f["kind"] for f in w.facts) == {"doc": 300, "user_true": 180, "user_false": 90, "confab": 60}
    assert len(w.unknown) == 50


def test_same_seed_same_stream():
    a, b = World(3), World(3)
    assert a.facts == b.facts and a.schedule == b.schedule
    assert World(4).facts != a.facts


def test_value_keys_do_not_contain_each_other():
    w = World(0)
    for rel, pool in w.pools.items():
        keys = [value_key(rel, v) for v in pool]
        for a in keys:
            for b in keys:
                assert a == b or a not in b, (rel, a, b)


def test_real_world_attacks_use_known_facts_only():
    w = World(2)
    w.set_real(range(len(REAL_FACTS)))
    assert all(r["true_key"] != r["false_key"] for r in w.real)
    assert sum(bool(r["attack"]) for r in w.real) == 30
