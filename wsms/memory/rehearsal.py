"""What is trained at night besides the new memories: replay of earlier memories and
homeostatic rehearsal (refusals, and answers of the frozen model to general questions)."""
from wsms.data.world import NAME_PARTS, REL_NAMES, RELATIONS
from wsms.utils.text import ABSTAIN_TEXT


def sample_replay(rng, pool, n):
    """n earlier memories; pool items are (training questions, target answer)."""
    out = []
    for _ in range(n):
        qs, tgt = rng.choice(pool)
        out.append((rng.choice(qs), tgt))
    return out


def sample_anchors(rng, anchors, n):
    """n general questions with the answer of the frozen model."""
    return [(x["q"], x["a"]) for x in rng.sample(anchors, min(len(anchors), n))]


class RefusalSampler:
    """Refusal practice: unseen names, and attributes of known people that were never written."""

    def __init__(self, world, rng):
        self.rng = rng
        self.used_names = world.used_names
        self.ku_set = {(k["person"], k["rel"]) for k in world.ku}     # held out for evaluation
        self.consolidated, self.known_people = set(), []

    def fresh_name(self):
        first_a, first_b, sur_a, sur_b = NAME_PARTS
        rng = self.rng
        while True:
            n = f"{rng.choice(first_a)}{rng.choice(first_b)} {rng.choice(sur_a)}{rng.choice(sur_b)}"
            if n not in self.used_names:
                return n

    def note_written(self, accepted):
        """Remember which people and attributes have been written to memory."""
        for c in accepted:
            if "person" in c["rec"]:
                if c["rec"]["person"] not in self.known_people:
                    self.known_people.append(c["rec"]["person"])
                self.consolidated.add((c["rec"]["person"], c["rec"]["rel"]))

    def sample(self, n):
        rng, out = self.rng, []
        for i in range(n):
            person, rel = self.fresh_name(), rng.choice(REL_NAMES)
            if i % 2 == 1 and self.known_people:
                for _ in range(20):
                    p2, r2 = rng.choice(self.known_people), rng.choice(REL_NAMES)
                    if (p2, r2) not in self.consolidated and (p2, r2) not in self.ku_set:
                        person, rel = p2, r2
                        break
            out.append((rng.choice(RELATIONS[rel]["train_q"]).format(p=person), ABSTAIN_TEXT))
        return out
