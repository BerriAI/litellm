from typing import Literal


def get_formatted_prompt(
    data: dict,
    call_type: Literal[
        "acompletion",
        "completion",
        "embedding",
        "aembedding",
        "embeddings",
        "image_generation",
        "aimage_generation",
        "audio_transcription",
        "transcription",
        "atranscription",
        "moderation",
        "amoderation",
        "text_completion",
        "atext_completion",
    ],
) -> str:
    """
    Extracts the prompt from the input data based on the call type.

    Returns a string.
    """
    prompt = ""
    if call_type in ("acompletion", "completion"):
        for message in data.get("messages", []):
            if message.get("content", None) is not None:
                content = message.get("content")
                if isinstance(content, str):
                    prompt += message["content"]
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            prompt += c.get("text", "")
            if "tool_calls" in message:
                for tool_call in message["tool_calls"] or ():
                    if "function" in tool_call:
                        function_arguments = tool_call["function"]["arguments"]
                        prompt += function_arguments
    elif call_type in ("text_completion", "atext_completion"):
        prompt = data.get("prompt", "")
    elif call_type in (
        "embedding",
        "aembedding",
        "embeddings",
        "moderation",
        "amoderation",
    ):
        if isinstance(data.get("input"), str):
            prompt = data["input"]
        elif isinstance(data.get("input"), list):
            for m in data["input"]:
                prompt += str(m)
    elif call_type in ("image_generation", "aimage_generation"):
        prompt = data.get("prompt", "")
    elif call_type in ("audio_transcription", "transcription", "atranscription"):
        if "prompt" in data:
            prompt = data["prompt"]
    return prompt
