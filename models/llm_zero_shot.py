import json

import anthropic

from config import NAME2INT

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


def classify(sentences, model="claude-opus-4-8"):
    """
    Zero-shot stance classification. Returns integer labels (0/1/2).
    Needs ANTHROPIC_API_KEY -- API billing is separate from a claude.ai subscription.

    Args:
        sentences: List of sentence strings.
        model: The Anthropic model to use. Defaults to "claude-opus-4-8".

    Returns:
        List of integer labels corresponding to the input sentences.
    """

    client = anthropic.Anthropic(max_retries=8)

    def one(sentence):
        r = client.messages.create(
            model=model,
            max_tokens=2048,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": PROMPT.format(sentence=sentence)}],
        )
        text = next(b.text for b in r.content if b.type == "text")
        return NAME2INT[json.loads(text)["label"]]

    return [one(s) for s in sentences]
