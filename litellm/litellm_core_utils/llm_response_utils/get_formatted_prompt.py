from typing import Final, Literal

from pydantic import TypeAdapter

_INPUT_ITEMS: Final = TypeAdapter(tuple[object, ...])


def _get_prompt_from_input(input_data: object, model: str) -> str:
    if isinstance(input_data, str):
        return input_data
    if not isinstance(input_data, (list, tuple)):
        return ""
    items: Final = _INPUT_ITEMS.validate_python(input_data)
    tokens: Final = tuple(token for token in items if isinstance(token, int))
    if tokens and len(tokens) == len(items):
        import litellm

        return str(litellm.decode(model=model, tokens=tokens))
    return "".join(item for item in items if isinstance(item, str))


def get_formatted_prompt(
    data: dict,
    call_type: Literal[
        "acompletion",
        "completion",
        "embedding",
        "aembedding",
        "image_generation",
        "audio_transcription",
        "atranscription",
        "transcription",
        "moderation",
        "text_completion",
    ],
) -> str:
    """
    Extracts the prompt from the input data based on the call type.

    Returns a string.
    """
    prompt = ""
    if call_type == "acompletion" or call_type == "completion":
        for message in data["messages"]:
            if message.get("content", None) is not None:
                content = message.get("content")
                if isinstance(content, str):
                    prompt += message["content"]
                elif isinstance(content, list):
                    for c in content:
                        if c["type"] == "text":
                            prompt += c["text"]
            if "tool_calls" in message:
                for tool_call in message["tool_calls"] or ():
                    if "function" in tool_call:
                        function_arguments = tool_call["function"]["arguments"]
                        prompt += function_arguments
    elif call_type == "text_completion":
        prompt = data["prompt"]
    elif call_type in ("embedding", "aembedding", "moderation"):
        if "input" not in data:
            return ""
        if isinstance(data["input"], str):
            return data["input"]
        if isinstance(data["input"], list):
            items: Final = _INPUT_ITEMS.validate_python(data["input"])
            model: Final = data["model"] if "model" in data and isinstance(data["model"], str) else ""
            if any(isinstance(item, list) for item in items):
                return "".join(_get_prompt_from_input(item, model) for item in items)
            return _get_prompt_from_input(items, model)
    elif (
        call_type
        in (
            "image_generation",
            "audio_transcription",
            "atranscription",
            "transcription",
        )
        and "prompt" in data
        and isinstance(data["prompt"], str)
    ):
        prompt = data["prompt"]
    return prompt
