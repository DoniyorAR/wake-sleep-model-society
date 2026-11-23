from wsms.utils.text import HEARSAY_PREFIX, classify, has, is_abstain, is_hedged, same_answer


def test_classify_codes():
    assert classify("She works at Norvex Labs.", "norvex", "zentra") == 1
    assert classify("She works at Zentra Bank.", "norvex", "zentra") == 2
    assert classify("I do not know.", "norvex", "zentra") == 3
    assert classify("A cello.", "norvex", "zentra") == 0
    assert classify(HEARSAY_PREFIX + "Norvex Labs", "norvex", "zentra") == 4
    assert classify(HEARSAY_PREFIX + "Zentra Bank", "norvex", "zentra") == 5
    assert classify(HEARSAY_PREFIX + "a cello", "norvex", "zentra") == 6


def test_matching_ignores_case_and_accents():
    assert has("He lives in TARMEL.", "tarmel")
    assert has("Café Belqor", "cafe belqor")
    assert not has("He lives in Tarmel.", None)


def test_abstain_and_hedge():
    assert is_abstain("I'm sorry, that is not mentioned.")
    assert is_abstain("")
    assert not is_abstain("Paris")
    assert is_hedged("The user said it is Paris.")
    assert not is_hedged("Paris")


def test_same_answer_is_loose():
    assert same_answer("The town of Tarmel.", "Tarmel")
    assert not same_answer("Tarmel", "Belqor")
    assert not same_answer("", "Tarmel")
