from wsms.memory.gate import WriteGate
from wsms.utils.text import HEARSAY_PREFIX


def cand(source, answer, key=(False, 1), p_self=0.9, p_helper=0.9, cb_self=None, cb_helper=None, cf_self=None,
         label="true"):
    return {"ep": {"source": source}, "answer": answer, "key": key, "label": label, "target": answer,
            "level": "fact", "why": "",
            "row": {"p_self": p_self, "p_helper": p_helper, "cb_self": cb_self, "cb_helper": cb_helper,
                    "cf_self": cf_self}}


def test_two_signals_are_needed():
    gate = WriteGate()
    assert gate.decide(cand("doc", "Tarmel", p_helper=0.2), 1) == "reject_unsupported"
    assert gate.decide(cand("doc", "Tarmel", p_self=0.2), 1) == "reject_unsupported"
    assert gate.decide(cand("doc", "Tarmel"), 1) == "accept_fact"
    assert WriteGate(support="self").decide(cand("doc", "Tarmel", p_helper=0.2), 1) == "accept_fact"


def test_refusal_is_never_written():
    assert WriteGate().decide(cand("doc", "I do not know."), 1) == "reject_abstain"


def test_user_remark_is_written_with_its_source():
    c = cand("user", "Tarmel")
    assert WriteGate().decide(c, 1) == "accept_hearsay"
    assert c["target"] == HEARSAY_PREFIX + "Tarmel" and c["level"] == "hearsay"


def test_user_cannot_overwrite_a_document_fact():
    gate = WriteGate()
    doc = cand("doc", "Tarmel")
    gate.decide(doc, 1)
    gate.commit([doc])
    c = cand("user", "Belqor")
    assert gate.decide(c, 3) == "veto" and c["why"] == "ledger"
    assert gate.decide(cand("user", "Tarmel"), 4) == "accept_fact"      # the user repeats the fact


def test_consensus_of_both_models_vetoes_a_claim():
    c = cand("user", "Sydney", cb_self="Canberra", cb_helper="Canberra.")
    assert WriteGate().decide(c, 1) == "veto" and c["why"] == "consensus"
    assert WriteGate().decide(cand("user", "Sydney", cb_self="Canberra", cb_helper="Melbourne"), 1) == "accept_hearsay"
    assert WriteGate(veto="none").decide(cand("user", "Sydney", cb_self="Canberra", cb_helper="Canberra"), 1) \
        == "accept_hearsay"


def test_document_confirms_or_replaces_a_claim():
    gate = WriteGate()
    claim = cand("user", "Tarmel")
    gate.decide(claim, 1)
    gate.commit([claim])
    confirm, replace = cand("doc", "Tarmel"), cand("doc", "Belqor")
    assert gate.decide(confirm, 5) == "accept_fact" and confirm["why"] == "document confirms hearsay"
    assert gate.decide(replace, 5) == "accept_fact" and replace["why"] == "document replaces hearsay"


def test_quarantine_until_repeated_on_another_day():
    gate = WriteGate(hearsay="quarantine", k=2)
    assert gate.decide(cand("user", "Tarmel"), 1) == "quarantine"
    assert gate.decide(cand("user", "Tarmel"), 1) == "quarantine"
    assert gate.decide(cand("user", "Tarmel"), 4) == "accept_fact"


def test_baselines():
    assert WriteGate("naive").decide(cand("user", "anything", p_self=0.0), 1) == "accept_fact"
    assert WriteGate("doc_only").decide(cand("user", "Tarmel"), 1) == "reject_source"
    assert WriteGate("oracle").decide(cand("user", "Tarmel", label="false"), 1) == "reject_oracle"
