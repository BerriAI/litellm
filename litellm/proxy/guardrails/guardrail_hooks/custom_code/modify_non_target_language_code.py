"""
Pre-built guardrail that replaces non-target-language text with a placeholder.

If the output is in the target language, allow unchanged. If not, replace
non-target segments with a message (modify). Uses the detect_language() primitive.

Use with the Custom Code Guardrail by setting litellm_params.custom_code to
MODIFY_NON_TARGET_LANGUAGE_GUARDRAIL_CODE.
"""

# Default target language (ISO 639-1). Edit in the code string to change.
DEFAULT_TARGET_LANGUAGE = "en"

# Message used when content is not in the target language (modify action).
# Edit in the code string to customize.
NON_TARGET_MESSAGE = "Content is not in the target language."

# Custom code: if detected language is target -> allow(); else -> modify(texts=[...]).
# Change the "else" branch to return block("...") or allow() if you want a different action.
MODIFY_NON_TARGET_LANGUAGE_GUARDRAIL_CODE = '''
def apply_guardrail(inputs, request_data, input_type):
    """If output is in target language, allow; otherwise replace non-target with a message."""
    target_language = "en"
    non_target_message = "Content is not in the target language."

    texts = inputs.get("texts") or []
    if not texts:
        return allow()

    result_texts = []
    any_non_target = False
    for text in texts:
        if not text:
            result_texts.append(text)
            continue
        detected = detect_language(text)
        if detected is None or detected == "unknown":
            result_texts.append(text)
            continue
        if lower(detected) == lower(target_language):
            result_texts.append(text)
        else:
            result_texts.append(non_target_message)
            any_non_target = True
    if any_non_target:
        return modify(texts=result_texts)
    return allow()
'''

__all__ = [
    "DEFAULT_TARGET_LANGUAGE",
    "MODIFY_NON_TARGET_LANGUAGE_GUARDRAIL_CODE",
    "NON_TARGET_MESSAGE",
]
