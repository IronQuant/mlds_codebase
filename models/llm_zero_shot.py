import json
import threading
from concurrent.futures import ThreadPoolExecutor

import anthropic

from config import NAME2INT

# Shah's zero-shot prompt (paper section 4.4), used for their ChatGPT-3.5-turbo row.
# We ask for JSON instead of "label on line 1, explanation on line 2" so the label
# is parsed reliably rather than regexed out of prose.
PROMPT = (
    "Discard all the previous instructions. Behave like you are an expert sentence "
    "classifier. Classify the following sentence from FOMC into 'HAWKISH', 'DOVISH', "
    "or 'NEUTRAL' class. Label 'HAWKISH' if it is corresponding to tightening of the "
    "monetary policy, 'DOVISH' if it is corresponding to easing of the monetary "
    "policy, or 'NEUTRAL' if the stance is neutral.\n\nThe sentence: {sentence}"
)

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(NAME2INT)}},
    "required": ["label"],
    "additionalProperties": False,
}


def classify(sentences, model="claude-opus-4-8", workers=8, verbose=True):
    """Zero-shot stance classification. Returns integer labels (0/1/2).

    Takes a list of sentence strings and returns one integer label each, in the
    same order. Needs ANTHROPIC_API_KEY -- API billing is separate from a
    claude.ai subscription.

    Calls run concurrently (network-bound, so threads not processes). executor.map
    preserves input order, so predictions still line up with `sentences`.
    """
    # 529s are common on the newest models; the SDK only retries twice by default
    # and one unhandled failure kills the whole ex.map, losing the seed's paid calls
    client = anthropic.Anthropic(max_retries=8)
    done = 0
    lock = threading.Lock()

    def one(sentence):
        nonlocal done
        r = client.messages.create(
            model=model,
            # fable-5 thinks on every call and can't be turned off; thinking tokens
            # count against max_tokens, so 64 truncates before the JSON is emitted.
            # Only generated tokens bill, so this costs the other models nothing.
            max_tokens=2048,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": PROMPT.format(sentence=sentence)}],
        )
        text = next(b.text for b in r.content if b.type == "text")

        with lock:
            done += 1
            if verbose and done % 50 == 0:
                print(f"  {done}/{len(sentences)}")

        return NAME2INT[json.loads(text)["label"]]

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, sentences))
