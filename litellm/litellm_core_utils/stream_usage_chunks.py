from typing import Final

from litellm.types.utils import ModelResponseStream, StreamingChoices


def is_empty_streaming_choice(choice: StreamingChoices) -> bool:
    if choice.finish_reason is not None:
        return False
    if getattr(choice, "logprobs", None) is not None:
        return False
    delta: Final = getattr(choice, "delta", None)
    if delta is None:
        return True
    return all(value is None for value in delta.model_dump().values())


def is_usage_only_chunk(chunk: ModelResponseStream) -> bool:
    if getattr(chunk, "usage", None) is None:
        return False
    return all(is_empty_streaming_choice(choice) for choice in chunk.choices or [])
