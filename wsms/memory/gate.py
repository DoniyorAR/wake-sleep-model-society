"""The write gate: which candidate memories are written at night, and at which trust level.

Trust levels of a written memory
  fact      stated by a document; written as a plain answer
  hearsay   stated only by the user; written with its source inside the memory
A user remark that conflicts with existing knowledge is not written ("veto").

Decisions
  accept_fact, accept_hearsay            written
  reject_abstain, reject_unsupported     dropped: no answer, or the source does not state it
  reject_source, reject_oracle           dropped by the document-only and oracle baselines
  veto                                   conflict with the ledger or with what the models know
  quarantine                             waits for a repetition (only with hearsay="quarantine")
"""
from wsms.utils.text import HEARSAY_PREFIX, is_abstain, same_answer

POLICIES = ("naive", "doc_only", "oracle", "gate")
SUPPORT_MODES = ("society", "self", "none")
VETO_MODES = ("society", "self", "ledger", "none")
HEARSAY_MODES = ("tag", "quarantine", "plain")


class WriteGate:
    """Decides candidate by candidate. The ledger of written memories changes once per night.

    policy   naive     every awake answer is written as a plain fact
             doc_only  only answers taken from documents are written
             oracle    exactly the true answers are written (upper bound)
             gate      trust-layered consolidation
    support  the source must state the answer: both models (society), the sleeping model (self), none
    veto     no conflict with the ledger or with known facts: consensus of both models (society),
             confident answer of the sleeping model (self), ledger only, none
    hearsay  an unconfirmed user claim is tagged, quarantined until repeated on k days, or written plain
    tau      confidence needed for a self veto
    """

    def __init__(self, policy="gate", support="society", veto="society", hearsay="tag", k=2, tau=0.9):
        assert policy in POLICIES and support in SUPPORT_MODES
        assert veto in VETO_MODES and hearsay in HEARSAY_MODES
        self.policy, self.support, self.veto, self.hearsay = policy, support, veto, hearsay
        self.k, self.tau = k, tau
        self.ledger = {}        # memory key -> {"answer", "level"}
        self.evidence = {}      # quarantine mode: memory key -> [{"answer", "days"}]

    def decide(self, cand, day):
        """Set the decision of one candidate memory (and its target text and level if it is a claim)."""
        row, a, src = cand["row"], cand["answer"], cand["ep"]["source"]
        if self.policy == "naive":
            cand["decision"] = "accept_fact"
        elif self.policy == "doc_only":
            cand["decision"] = "accept_fact" if src == "doc" else "reject_source"
        elif self.policy == "oracle":
            cand["decision"] = "accept_fact" if cand["label"] == "true" else "reject_oracle"
        else:
            ok = True
            if self.support != "none":
                ok = row["p_self"] >= 0.5
                if self.support == "society" and row["p_helper"] is not None:
                    ok = ok and row["p_helper"] >= 0.5
            led = self.ledger.get(cand["key"])
            if is_abstain(a):
                cand["decision"] = "reject_abstain"
            elif not ok:
                cand["decision"] = "reject_unsupported"
            elif src == "doc":
                cand["decision"] = "accept_fact"
                if led and not same_answer(led["answer"], a):
                    cand["why"] = "document replaces " + led["level"]
                elif led and led["level"] == "hearsay":
                    cand["why"] = "document confirms hearsay"
            elif led and led["level"] == "fact" and same_answer(led["answer"], a):
                cand["decision"] = "accept_fact"          # the user repeats a known fact
            else:
                conflict = self._conflict(row, a, led)
                if conflict:
                    cand["decision"], cand["why"] = "veto", conflict
                elif self.hearsay == "tag":
                    cand["decision"], cand["level"] = "accept_hearsay", "hearsay"
                    cand["target"] = HEARSAY_PREFIX + a
                elif self.hearsay == "plain":
                    cand["decision"] = "accept_fact"
                else:
                    ev = self.evidence.setdefault(cand["key"], [])
                    entry = next((e for e in ev if same_answer(e["answer"], a)), None)
                    if entry is None:
                        entry = {"answer": a, "days": set()}
                        ev.append(entry)
                    entry["days"].add(day)
                    cand["decision"] = "accept_fact" if len(entry["days"]) >= self.k else "quarantine"
        return cand["decision"]

    def _conflict(self, row, a, led):
        """Name of the knowledge that a user claim contradicts, or an empty string."""
        if self.veto != "none" and led and led["level"] == "fact":
            return "ledger"
        if self.veto == "society":
            s, h = row["cb_self"] or "", row["cb_helper"] or ""
            if h and not is_abstain(s) and not is_abstain(h) and same_answer(s, h) \
                    and not same_answer(h, a):
                return "consensus"
        elif self.veto == "self":
            s = row["cb_self"] or ""
            if not is_abstain(s) and (row["cf_self"] or 0) >= self.tau and not same_answer(s, a):
                return "self"
        return ""

    def commit(self, accepted):
        """Record the memories written this night."""
        for c in accepted:
            self.ledger[c["key"]] = {"answer": c["answer"], "level": c["level"]}


def accepted(cands):
    return [c for c in cands if c["decision"].startswith("accept")]
