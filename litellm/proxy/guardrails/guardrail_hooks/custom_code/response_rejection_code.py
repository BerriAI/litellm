"""
Custom code for a response guardrail that blocks when the model response
indicates it is rejecting the user request (e.g. "That's not something I can help with").

Use this with the Custom Code Guardrail (custom_code) by setting litellm_params.custom_code
to RESPONSE_REJECTION_GUARDRAIL_CODE. The guardrail runs only on input_type "response"
and raises a block error if any response text matches known rejection phrases.
"""

# Default phrases that indicate the model is refusing the user request (lowercase for case-insensitive match).
# Custom code guardrails can override by defining rejection_phrases in the code.
DEFAULT_REJECTION_PHRASES = [
    "that's not something i can help with",
    "that is not something i can help with",
    "i can't help with that",
    "i cannot help with that",
    "i'm not able to help",
    "i am not able to help",
    "i'm unable to help",
    "i cannot assist",
    "i can't assist",
    "i'm not allowed to",
    "i'm not permitted to",
    "i won't be able to help",
    "i'm sorry, i can't",
    "i'm sorry, i cannot",
    "as an ai, i can't",
    "as an ai, i cannot",
]

# Custom code string for the Custom Code Guardrail. Only runs on input_type "response".
# Uses primitives: allow(), block(), lower(), contains()
RESPONSE_REJECTION_GUARDRAIL_CODE = '''
def apply_guardrail(inputs, request_data, input_type):
    """Block responses that indicate the model rejected the user request."""
    if input_type != "response":
        return allow()

    texts = inputs.get("texts") or []
    # All lowercase for case-insensitive matching (text is lowercased before check)
    rejection_phrases = [
        "that's not something i can help with",
        "that is not something i can help with",
        "i can't help with that",
        "i cannot help with that",
        "i'm not able to help",
        "i am not able to help",
        "i'm unable to help",
        "i cannot assist",
        "i can't assist",
        "i'm not allowed to",
        "i'm not permitted to",
        "i won't be able to help",
        "i'm sorry, i can't",
        "i'm sorry, i cannot",
        "as an ai, i can't",
        "as an ai, i cannot",
    ]

    for text in texts:
        if not text:
            continue
        text_lower = lower(text)
        for phrase in rejection_phrases:
            if contains(text_lower, phrase):
                return block(
                    "Response indicates the model rejected the user request.",
                    detection_info={"matched_phrase": phrase, "input_type": "response"},
                )
    return allow()
'''

__all__ = [
    "DEFAULT_REJECTION_PHRASES",
    "RESPONSE_REJECTION_GUARDRAIL_CODE",
]
