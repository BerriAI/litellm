from typing import List, Literal


def get_formatted_prompt(
    data: dict,
    call_type: Literal[
        "acompletion",
        "completion",
        "atext_completion",
        "text_completion",
        "aembedding",
        "embedding",
        "embeddings",
        "aimage_generation",
        "image_generation",
        "atranscription",
        "audio_transcription",
        "amoderation",
        "moderation",
    ],
) -> str:
    """
    Extracts the prompt from the input data based on the call type.

    Returns a string.
    """
    prompt = ""
    if call_type in ("acompletion", "completion"):
        for message in data["messages"]:
            if message.get("content", None) is not None:
                content = message.get("content")
                if isinstance(content, str):
                    prompt += message["content"]
                elif isinstance(content, List):
                    for c in content:
                        if c["type"] == "text":
                            prompt += c["text"]
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    if "function" in tool_call:
                        function_arguments = tool_call["function"]["arguments"]
                        prompt += function_arguments
    elif call_type in ("atext_completion", "text_completion"):
        prompt = data["prompt"]
    elif call_type in (
        "aembedding",
        "embedding",
        "embeddings",
        "amoderation",
        "moderation",
    ):
        if isinstance(data["input"], str):
            prompt = data["input"]
        elif isinstance(data["input"], list):
            for m in data["input"]:
                prompt += m
    elif call_type in ("aimage_generation", "image_generation"):
        prompt = data["prompt"]
    elif call_type in ("atranscription", "audio_transcription"):
        if "prompt" in data:
            prompt = data["prompt"]
    return prompt
