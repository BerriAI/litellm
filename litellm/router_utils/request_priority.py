from collections.abc import Callable, Iterable, Mapping
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
    requested: object, default_priority: int | None, drops_params: Callable[[], bool]
) -> int | None | InvalidPriority:
    if requested is None:
        return default_priority
    if isinstance(requested, int) and not isinstance(requested, bool):
        return requested
    if drops_params():
        return default_priority
    return InvalidPriority(value=requested)


def _explicit_drop_params(scope: Mapping[str, object]) -> bool | None:
    return normalize_drop_params(scope.get("drop_params"))


def request_drops_params(
    kwargs: Mapping[str, object],
    router_defaults: Mapping[str, object],
    deployment_params: Iterable[Mapping[str, object]],
) -> bool:
    request_flag: Final = _explicit_drop_params(kwargs)
    if request_flag is not None:
        return request_flag
    router_flag: Final = _explicit_drop_params(router_defaults)
    if router_flag is not None:
        return router_flag
    deployment_flags: Final = tuple(
        flag for flag in (_explicit_drop_params(params) for params in deployment_params) if flag is not None
    )
    if deployment_flags:
        return any(deployment_flags)
    return litellm.drop_params is True
