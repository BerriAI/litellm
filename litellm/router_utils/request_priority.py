from collections.abc import Iterable, Mapping
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


def request_drops_params(
    kwargs: Mapping[str, object],
    router_defaults: Mapping[str, object],
    deployment_params: Iterable[Mapping[str, object]],
) -> bool:
    for scope in (kwargs, router_defaults):
        flag: Final = normalize_drop_params(scope.get("drop_params"))
        if flag is not None:
            return flag
    deployment_flags: Final = tuple(
        flag
        for flag in (normalize_drop_params(params.get("drop_params")) for params in deployment_params)
        if flag is not None
    )
    if deployment_flags:
        return any(deployment_flags)
    return litellm.drop_params is True
