from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import litellm
from litellm.litellm_core_utils.core_helpers import normalize_drop_params


@dataclass(frozen=True, slots=True)
class InvalidPriority:
    value: object

    @property
    def message(self) -> str:
        return f"priority must be an integer, got {type(self.value).__name__} {self.value!r}"


def resolve_request_priority(
    requested: object, default_priority: int | None, drop_params: bool
) -> int | None | InvalidPriority:
    if requested is None:
        return default_priority
    if isinstance(requested, int) and not isinstance(requested, bool):
        return requested
    if drop_params:
        return default_priority
    return InvalidPriority(value=requested)


def request_drops_params(kwargs: Mapping[str, object]) -> bool:
    requested: Final = normalize_drop_params(kwargs.get("drop_params"))
    if requested is None:
        return litellm.drop_params is True
    return requested
