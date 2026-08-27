import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# institutional / procedural / calendar boilerplate, from the UDA project
DOMAIN_STOPWORDS = {
    "committee",
    "committees",
    "federal",
    "reserve",
    "system",
    "board",
    "fomc",
    "open",
    "market",
    "meeting",
    "meetings",
    "statement",
    "statements",
    "release",
    "released",
    "announced",
    "announcement",
    "decided",
    "decision",
    "judges",
    "judge",
    "seeks",
    "seek",
    "notes",
    "note",
    "continues",
    "continue",
    "met",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "today",
    "recent",
    "recently",
    "currently",
    "factors",
    "measures",
    "indicators",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS


def clean(sentence):
    """
    (1) Lowercase,
    (2) strip punctuation and digits,
    (3) drop short words and stopwords.

    Patil(2026) UDA project's recipe.
    No stemming: "tightening" and "tightened" carry distinct policy signals.
    """
    s = re.sub(r"\d+", " ", re.sub(r"[^\w\s]", " ", sentence.lower()))
    return [w for w in s.split() if len(w) >= 3 and w not in STOPWORDS]
