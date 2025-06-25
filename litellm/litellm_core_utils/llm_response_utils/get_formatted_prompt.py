from litellm._logging import verbose_proxy_logge
from typing import List, Literal

def get_formatted_prompt(
    data: dict,
    call_type: Literal[
        "completion",
        "acompletion",
        "embedding",
        "image_generation",
        "audio_transcription",
        "moderation",
        "text_completion",
    ]) -> str:

    """
    Extracts the prompt from the input data based on the call type.
    Returns a string.
    """
    if call_type in {"completion", "acompletion"}:
        return _extract_messages(data.get("messages", []))
    elif call_type in {"text_completion"}:
        return "".join(data.get('prompt', []))
    elif call_type in {"image_generation", "audio_transcription"}:
        if (prompt:=data.get('prompt', None)):
            return prompt
    elif call_type in {"embedding", "moderation"}:
        input_data = data.get("input", "")
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, list):
            return "".join(map(str, input_data))
        return ""
    else:
        verbose_proxy_logger.warning(f"Undefined call_type: {call_type}")
        return ""

def _extract_messages(messages: List[dict]) -> str:
    result = ""
    for message in messages:
        content = message.get('content', None)
        if content == None:
            continue
