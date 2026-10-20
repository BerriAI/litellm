from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import normalize_drop_params


@dataclass(frozen=True, slots=True)
class InvalidPriority:
    value: object

    @property
    def message(self) -> str:
        return f"priority must be an integer, got {type(self.value).__name__} {self.value!r}"


def priority_error(model: str, message: str) -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message=message,
        model=model,
        llm_provider="",
        body={"message": message, "type": "invalid_request_error", "param": "priority", "code": "400"},
    )


_PRIORITY: Final = TypeAdapter(int)


def _as_priority(value: object) -> int | InvalidPriority:
    try:
        return _PRIORITY.validate_python(value)
    except ValidationError:
        return InvalidPriority(value=value)


def resolve_request_priority(
    requested: object, default_priority: int | None, drops_params: Callable[[], bool]
) -> int | None | InvalidPriority:
    if requested is None:
        return default_priority
    priority: Final = _as_priority(requested)
    if isinstance(priority, InvalidPriority) and drops_params():
        return default_priority
    return priority


def parse_default_priority(value: object) -> int | None | InvalidPriority:
    if value is None:
        return None
    if isinstance(value, bool):
        return InvalidPriority(value=value)
    return _as_priority(value)


def unqueueable_priority_message(requested: object) -> str:
    priority: Final = None if requested is None else _as_priority(requested)
    if isinstance(priority, InvalidPriority):
        return f"{priority.message}, and there is no router_settings.default_priority to queue it at instead"
    return "priority is required to queue a request, send one or set router_settings.default_priority"


def _explicit_drop_params(scope: Mapping[str, object]) -> bool | None:
    return normalize_drop_params(scope.get("drop_params"))


def _drops_priority_by_name(scope: Mapping[str, object]) -> bool:
    names: Final = scope.get("additional_drop_params")
    return isinstance(names, list) and "priority" in names


def request_drops_params(
    kwargs: Mapping[str, object],
    router_defaults: Mapping[str, object],
    deployment_params: Iterable[Mapping[str, object]],
) -> bool:
    deployments: Final = tuple(deployment_params)
    if litellm.drop_params is True or any(_drops_priority_by_name(s) for s in (kwargs, router_defaults, *deployments)):
        return True
    request_flag: Final = _explicit_drop_params(kwargs)
    if request_flag is not None:
        return request_flag
    router_flag: Final = _explicit_drop_params(router_defaults)
    if router_flag is not None:
        return router_flag
    return any(_explicit_drop_params(params) is True for params in deployments)
