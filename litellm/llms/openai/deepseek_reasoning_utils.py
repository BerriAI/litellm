from typing import Any, List, Optional


def requires_deepseek_v4_reasoning_content(model: Optional[str]) -> bool:
    """Return True when the model requires DeepSeek V4 thinking history."""
    if not model:
        return False

    normalized_model = model.lower()
    if normalized_model.startswith("responses/"):
        normalized_model = normalized_model.split("responses/", 1)[1]

    return "deepseek-v4" in normalized_model


def patch_deepseek_v4_reasoning_messages(
    model: Optional[str], messages: List[Any]
) -> List[Any]:
    """
    Ensure assistant tool-call messages include reasoning_content for DeepSeek V4.

    DeepSeek V4 rejects multi-turn requests when prior assistant tool-call messages
    omit the reasoning_content field, even if the value is empty.
    """
    if not requires_deepseek_v4_reasoning_content(model):
        return messages

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        if not (message.get("tool_calls") or message.get("tool_call_id")):
            continue
        if message.get("reasoning_content") is None:
            message["reasoning_content"] = ""

    return messages
