from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import (
    Final,
    cast,  # noqa: TID251  # Preserves arbitrary provider fields without lossy TypedDict validation.
)

from litellm.types.llms.openai import AllMessageValues


def normalize_reasoning_content(
    messages: Sequence[AllMessageValues], *, forward: bool = True, normalize: bool = True
) -> list[AllMessageValues]:  # mutable-ok: provider request contract
    def normalize_message(message: AllMessageValues) -> AllMessageValues:
        if message["role"] != "assistant":
            return message
        if not normalize and forward:
            return message
        history: Final[Mapping[str, object]] = message
        removed_fields: Final = ("reasoning", "reasoning_content") if normalize else ("reasoning_content",)
        reasoning: Final = (
            history.get("reasoning") if history.get("reasoning") is not None else history.get("reasoning_content")
        )
        normalized: Final[Mapping[str, object]] = MappingProxyType(
            {
                **MappingProxyType({key: value for key, value in history.items() if key not in removed_fields}),
                **(
                    MappingProxyType({"reasoning": reasoning})
                    if normalize and forward and reasoning is not None
                    else MappingProxyType({})
                ),
            }
        )
        result: Final = dict(normalized)  # mutable-ok: provider request contract
        return cast(AllMessageValues, result)  # cast-ok: only optional reasoning keys change

    return [normalize_message(message) for message in deepcopy(messages)]  # mutable-ok: provider request contract
