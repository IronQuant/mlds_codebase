"""Rule-based hawkish/dovish/neutral classifier.

Replicates the Gorodnichenko et al. (2021) dictionary method exactly as
implemented in Shah et al. (2023), code_model/rule_based.py.
"""

# A1: economic-concept nouns (inflation / rates / prices)
A1 = [
    "inflation expectation",
    "interest rate",
    "bank rate",
    "fund rate",
    "price",
    "economic activity",
    "inflation",
    "employment",
]

# A2: downward / easing direction words
A2 = [
    "anchor",
    "cut",
    "subdue",
    "decline",
    "decrease",
    "reduce",
    "low",
    "drop",
    "fall",
    "fell",
    "decelarate",
    "slow",
    "pause",
    "pausing",
    "stable",
    "non-accelerating",
    "downward",
    "tighten",
]

# B1: growth / labour-market / demand nouns
B1 = [
    "unemployment",
    "growth",
    "exchange rate",
    "productivity",
    "deficit",
    "demand",
    "job market",
    "monetary policy",
]

# B2: upward / rising direction words
B2 = [
    "ease",
    "easing",
    "rise",
    "rising",
    "increase",
    "expand",
    "improve",
    "strong",
    "upward",
    "raise",
    "high",
    "rapid",
]

# C: negation words — flip hawkish <-> dovish (never touches neutral)
C = [
    "weren't",
    "were not",
    "wasn't",
    "was not",
    "did not",
    "didn't",
    "do not",
    "don't",
    "will not",
    "won't",
]


def rule_model(sentences):
    """Classify an iterable of sentence strings.

    Returns a list of integer labels (0 = dovish, 1 = hawkish, 2 = neutral):
      - dovish  if (A1 & A2) or (B1 & B2)  -- e.g. inflation falling, unemployment rising
      - hawkish if (A1 & B2) or (B1 & A2)  -- e.g. inflation rising, unemployment falling
      - neutral otherwise
      - a negation word flips a hawkish/dovish call to the opposite class
    """
    preds = []
    for s in sentences:
        s = s.lower()
        if (any(w in s for w in A1) and any(w in s for w in A2)) or (
            any(w in s for w in B1) and any(w in s for w in B2)
        ):
            label = 0  # dovish
        elif (any(w in s for w in A1) and any(w in s for w in B2)) or (
            any(w in s for w in B1) and any(w in s for w in A2)
        ):
            label = 1  # hawkish
        else:
            label = 2  # neutral

        if label != 2 and any(w in s for w in C):
            preds.append(1 - label)  # flip dovish <-> hawkish
        else:
            preds.append(label)
    return preds
