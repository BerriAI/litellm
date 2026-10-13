from typing import Literal

from litellm._logging import verbose_logger


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
        if "input" in data:
            if isinstance(data["input"], str):
                prompt = data["input"]
            elif isinstance(data["input"], list):
                if data["input"] and isinstance(data["input"][0], int):
                    try:
                        import litellm

                        prompt = str(
                            litellm.decode(
                                model=data["model"] if "model" in data and isinstance(data["model"], str) else "",
                                tokens=data["input"],
                            )
                        )
                    except (TypeError, ValueError, KeyError, AttributeError) as exc:
                        verbose_logger.debug("Failed to decode token IDs: %s", exc)
                elif data["input"] and isinstance(data["input"][0], list):
                    for token_list in data["input"]:
                        if isinstance(token_list, list):
                            try:
                                import litellm

                                prompt += str(
                                    litellm.decode(
                                        model=data["model"]
                                        if "model" in data and isinstance(data["model"], str)
                                        else "",
                                        tokens=token_list,
                                    )
                                )
                            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                                verbose_logger.debug("Failed to decode token IDs: %s", exc)
                else:
                    for m in data["input"]:
                        if isinstance(m, str):
                            prompt += m
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
