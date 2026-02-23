"""
Pre-built guardrail that blocks responses not in English.

Uses the is_english() and detect_language() primitives. Use with the Custom Code
Guardrail by setting litellm_params.custom_code to BLOCK_NON_ENGLISH_GUARDRAIL_CODE.
Runs only on input_type "response"; blocks with detection_info when any text is
not English, otherwise allows with detection_info indicating English.
"""

BLOCK_NON_ENGLISH_GUARDRAIL_CODE = '''
def apply_guardrail(inputs, request_data, input_type):
    """Block responses that are not in English; allow with detection_info otherwise."""
    if input_type != "response":
        return allow()

    for text in inputs.get("texts") or []:
        if not is_english(text):
            return block(
                "Output is not in English",
                detection_info={"language": detect_language(text)},
            )
    return allow(detection_info={"is_english": True, "language": "en"})
'''

__all__ = [
    "BLOCK_NON_ENGLISH_GUARDRAIL_CODE",
]
