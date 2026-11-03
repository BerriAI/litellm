from collections.abc import Iterable
from typing import Final

from litellm.constants import VLLM_CHOICE_FIELDS
from litellm.types.utils import ModelResponseStream, StreamingChoices


def is_empty_streaming_choice(choice: StreamingChoices) -> bool:
    if choice.finish_reason is not None:
        return False
    if getattr(choice, "logprobs", None) is not None:
        return False
    if not _all_values_none(_vllm_choice_values(choice)):
        return False
    delta: Final = choice.delta if hasattr(choice, "delta") else None
    if delta is None:
        return True
    return _all_values_none(delta.model_dump().values())


def _vllm_choice_values(choice: StreamingChoices) -> Iterable[object]:
    choice_extras: Final = choice.model_extra or {}
    return (choice_extras.get(field) for field in VLLM_CHOICE_FIELDS)


def _all_values_none(values: Iterable[object]) -> bool:
    return all(value is None for value in values)


def is_injected_stream_usage_artifact(chunk: object) -> bool:
    if not isinstance(chunk, ModelResponseStream):
        return False
    if chunk.provider_specific_fields is not None:
        return False
    return all(is_empty_streaming_choice(choice) for choice in chunk.choices or [])
