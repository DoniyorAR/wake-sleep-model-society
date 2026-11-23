"""Answer matching and scoring, shared by the write gate and the evaluation.

Answer codes (see `classify`)
  0 other   1 true   2 false   3 refusal
  4 / 5 / 6 = true / false / other, given as a claim of the user
"""
import re
import unicodedata

ABSTAIN_TEXT = "I do not know."
HEARSAY_PREFIX = "According to the user (unconfirmed): "
CODE_LABEL = {0: "other", 1: "true", 2: "false", 3: "abstain", 4: "true", 5: "false", 6: "other"}


def norm(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


def has(answer, key):
    return bool(key) and norm(key) in norm(answer)


_ABSTAIN = re.compile(
    r"(don't|do not|doesn't|does not|didn't|did not|cannot|can't|couldn't|unable|unknown|"
    r"unclear|unsure|not (sure|aware|specified|mentioned|known|available|provided|stated|given|"
    r"certain|clear|possible|enough)|no (information|record|data|details|mention|way|idea|"
    r"specific)|i'm sorry|i am sorry|sorry|not have)", re.I)


def is_abstain(answer):
    return bool(_ABSTAIN.search(answer)) or not answer.strip()


_HEDGE = re.compile(r"according to the user|unconfirmed|the user (said|told|mentioned|claims|claimed|stated)", re.I)


def is_hedged(answer):
    return bool(_HEDGE.search(answer))


def classify(answer, true_key, false_key):
    """0 other, 1 true, 2 false, 3 abstains; 4 / 5 / 6 = true / false / other given as a user claim."""
    h = is_hedged(answer)
    t = has(answer, true_key)
    f = has(answer, false_key) if false_key else False
    if t and not f:
        return 4 if h else 1
    if f and not t:
        return 5 if h else 2
    if is_abstain(answer) and not h:
        return 3
    return 6 if h else 0


def _core(text):
    text = norm(text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [w for w in text.split() if w not in ("the", "a", "an", "of", "town", "is", "in", "at")]
    return " ".join(words)


def same_answer(a, b):
    """Loose equality used by the gate (it does not know the value pools)."""
    ca, cb = _core(a), _core(b)
    if not ca or not cb:
        return False
    return ca in cb or cb in ca
