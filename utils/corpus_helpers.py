import re
import unicodedata
import ftfy


def normalize(s):
    """
    Clean a raw sentence into the uniform form stored as `sentence`.
    Repair mojibake, NFKC-normalise, drop U+FFFD junk, strip wrapping quotes,
    lowercase, then collapse whitespace. Step-by-step examples inline below.

    Args:
        s: Raw sentence (any type; coerced to str).

    Returns:
        The cleaned, lowercased, whitespace-collapsed sentence.
    """

    # input: "the firmâ€™s return"  ->  "the firm's return"
    s = ftfy.fix_text(str(s))

    # input: "inﬁation ：0.25%"  ->  "inflation :0.25%"
    s = unicodedata.normalize("NFKC", s)

    # input: "mee<?>ng policy commi<?>ee"  ->  "mee ng policy commi ee"
    s = s.replace("�", " ")

    # input: "the risks to the outlook... as balanced"
    # ->  "the risks to the outlook as balanced"
    s = s.replace("...", " ")

    # input: '"the committee decided"'  ->  "the committee decided"
    s = s.strip().strip("\"'")

    # input: "The Committee Decided"  ->  "the committee decided"
    s = s.lower()

    # input: "raise  rates\n"  ->  "raise rates"
    # \s matches any whitespace (space, \t, \n, \r)
    # collapses each run to a single space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def dedup_key(s):
    """Return a punctuation-insensitive match key for deduplication.

    On top of normalize(), strip everything but [a-z0-9 ] so 'inflation.',
    'inflation' and '"Inflation"' all collapse to the same key.

    Args:
        s: Raw sentence (any type; coerced to str via normalize()).

    Returns:
        The normalised string with all punctuation reduced to single spaces.
    """
    k = re.sub(r"[^a-z0-9 ]", " ", normalize(s))
    return re.sub(r"\s+", " ", k).strip()


_SUSPENDED = {"and", "or", "to", "the", "a", "of", "in", "but", "nor"}


def rejoin_hyphens(s):
    """Rejoin words split across a PDF line break by hyphenation.

    "unemploy- ment" -> "unemployment", "construc- tion" -> "construction".
    Skip suspended compounds ("intermediate- and long-term")

    Args:
        s: Text whose hyphenated line breaks should be rejoined.

    Returns:
        The text with broken words rejoined; suspended compounds left intact.
    """
    return re.sub(
        r"([a-z])- ([a-z]+)",
        lambda m: m.group(0) if m.group(2) in _SUSPENDED else m.group(1) + m.group(2),
        s,
    )


def is_junk(s):
    """Return True for PDF lines carrying no usable language.

    Roster / page-header lines ("PRESENT: ___", "Page 8 of 117") and numeric
    table rows (e.g. SEP projection tables) — dropped before entering the corpus.

    Args:
        s: A single candidate sentence/line.

    Returns:
        True if the line is roster/header boilerplate or >50% numeric tokens.
    """
    if re.search(r"_{3,}|Page \d+ of \d+", s):
        return True
    toks = s.split()
    nums = sum(bool(re.fullmatch(r"[-+]?\d[\d.,]*%?", t)) for t in toks)
    return bool(toks) and nums / len(toks) > 0.5
