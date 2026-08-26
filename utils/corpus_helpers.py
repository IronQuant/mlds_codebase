import re
import unicodedata
import ftfy


def normalize(s):
    """Clean a raw sentence into the uniform form stored as `sentence`.

    Repair mojibake, NFKC-normalise, drop U+FFFD junk, strip wrapping quotes,
    lowercase, then collapse whitespace. Step-by-step examples inline below.

    Args:
        s: Raw sentence (any type; coerced to str).

    Returns:
        The cleaned, lowercased, whitespace-collapsed sentence.
    """

    # input: "the firmâ€™s return"  ->  "the firm's return"
    # Shah's HF dataset ships double-decoded UTF-8 (U+2019 read as cp1252).
    # MUST run before NFKC: NFKC expands the ™ to "TM", destroying the pattern
    # ftfy needs to recognise the mojibake ("Iâ€™d" -> "iâ€tmd", unrecoverable).
    s = ftfy.fix_text(str(s))

    # input: "inﬁation ：0.25%"  ->  "inflation :0.25%"
    # NFKC expands ligatures, fullwidth punctuation (Bank of Japan PDFs), fractions (¼)
    s = unicodedata.normalize("NFKC", s)

    # input: "mee<?>ng policy commi<?>ee"  ->  "mee ng policy commi ee"
    s = s.replace("�", " ")

    # input: "the risks to the outlook... as balanced"  ->  "the risks to the outlook as balanced"
    # abridged quotations leave ellipses that carry no language.
    # NFKC has already folded U+2026 into "...", so one pattern covers both.
    s = s.replace("...", " ")

    # input: '"the committee decided"'  ->  "the committee decided"
    s = s.strip().strip("\"'")

    # input: "The Committee Decided"  ->  "the committee decided"
    s = s.lower()

    # input: "raise  rates\n"  ->  "raise rates"
    # \s matches any whitespace (space, \t, \n, \r); + collapses each run to a single space
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
