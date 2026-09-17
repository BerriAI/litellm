from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    cast,  # noqa: TID251  # narrows legacy untyped registries at the boundary
)

from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

CallbackTarget: TypeAlias = str | Callable[..., object] | CustomLogger
RegistryName: TypeAlias = Literal[
    "input", "async_input", "success", "async_success", "failure", "async_failure", "callbacks"
]
NamedEvent: TypeAlias = Literal["success", "failure"]
Kwargs: TypeAlias = dict[str, object]  # mutable-ok: the retained call kwargs are the dict callbacks receive and mutate

_REGISTRY_ATTRIBUTES: Final[Mapping[RegistryName, str]] = MappingProxyType(
    {
        "input": "input_callback",
        "async_input": "_async_input_callback",
        "success": "success_callback",
        "async_success": "_async_success_callback",
        "failure": "failure_callback",
        "async_failure": "_async_failure_callback",
        "callbacks": "callbacks",
    }
)


class _CallbackManager(Protocol):
    def add_litellm_success_callback(self, callback: CallbackTarget) -> None: ...

    def add_litellm_failure_callback(self, callback: CallbackTarget) -> None: ...

    def add_litellm_async_success_callback(self, callback: CallbackTarget) -> None: ...

    def add_litellm_async_failure_callback(self, callback: CallbackTarget) -> None: ...


class _LoggingFactory(Protocol):
    def __call__(
        self,
        *,
        model: str | None,
        messages: object,
        stream: bool,
        litellm_call_id: str,
        litellm_trace_id: str | None,
        function_id: str,
        call_type: str,
        start_time: datetime.datetime,
        dynamic_success_callbacks: Sequence[CallbackTarget] | None,
        dynamic_failure_callbacks: Sequence[CallbackTarget] | None,
        dynamic_async_success_callbacks: Sequence[CallbackTarget] | None,
        dynamic_async_failure_callbacks: Sequence[CallbackTarget] | None,
        kwargs: Kwargs,
        applied_guardrails: Sequence[str],
        supports_correlation_logging: bool,
    ) -> Logging: ...


class _EnvironmentUpdater(Protocol):
    def __call__(
        self,
        *,
        model: str | None,
        user: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream_options: object,
    ) -> None: ...


def _legacy(value: object) -> Callable[..., object]:
    return cast(Callable[..., object], value)  # cast-ok: legacy module-level helpers are untyped


def _mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, object], value)  # cast-ok: isinstance narrows only to Mapping[Unknown, Unknown]


def registry(
    name: RegistryName,
) -> MutableSequence[CallbackTarget]:  # mutable-ok: the public litellm registries are mutated by contract
    import litellm

    return cast(  # cast-ok: legacy module-level lists are untyped
        MutableSequence[CallbackTarget], getattr(litellm, _REGISTRY_ATTRIBUTES[name])
    )


def is_async_callable(callback: object) -> bool:
    from litellm.litellm_core_utils.cached_imports import get_coroutine_checker

    return get_coroutine_checker().is_async_callable(callback)


def is_known_name(callback: str) -> bool:
    import litellm

    known: Final = cast(Sequence[str], litellm._known_custom_logger_compatible_callbacks)  # pyright: ignore[reportPrivateUsage]  # cast-ok: registry list has no public typed accessor
    return callback in known


def resolve_named_integration(callback: str) -> CustomLogger | None:
    from litellm.litellm_core_utils import litellm_logging

    factory: Final = _legacy(litellm_logging._init_custom_logger_compatible_class)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownArgumentType]  # legacy factory
    resolved: Final = factory(callback, internal_usage_cache=None, llm_router=None)
    return resolved if isinstance(resolved, CustomLogger) else None


def async_success_registry_has_type(callback: object) -> bool:
    return any(type(existing) is type(callback) for existing in registry("async_success"))


def bootstrap_pending() -> bool:
    from litellm import utils

    return not utils.callback_list


def bootstrap(function_id: str | None) -> None:
    from litellm import utils
    from litellm.litellm_core_utils import cached_imports

    combined: Final = _bootstrap_list()
    utils.callback_list = combined  # rebind-ok: legacy module global consumed by set_callbacks
    set_callbacks: Final = _legacy(cached_imports.get_set_callbacks())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # cached import is untyped
    set_callbacks(callback_list=combined, function_id=function_id)


def _bootstrap_list() -> list[str]:  # mutable-ok: set_callbacks and the legacy module global expect a list
    names: Final = frozenset({*registry("input"), *registry("success"), *registry("failure")})
    consumed: Final = cast(Sequence[str], names)  # cast-ok: legacy list annotation is narrower than its contents
    return list(consumed)  # mutable-ok: consumed by set_callbacks


def expand_named(callback: str, event: NamedEvent) -> None:
    from litellm import utils

    utils._add_custom_logger_callback_to_specific_event(callback, event)  # pyright: ignore[reportPrivateUsage]  # legacy expansion helper


def append_registry(name: RegistryName, callback: CallbackTarget) -> None:
    import litellm

    manager: Final = cast(_CallbackManager, litellm.logging_callback_manager)  # cast-ok: legacy manager is untyped
    match name:
        case "input" | "async_input":
            registry(name).append(callback)
        case "success":
            manager.add_litellm_success_callback(callback)
        case "async_success":
            manager.add_litellm_async_success_callback(callback)
        case "failure":
            manager.add_litellm_failure_callback(callback)
        case "async_failure":
            manager.add_litellm_async_failure_callback(callback)
        case "callbacks":
            raise KeyError(name)


def remove_registry(name: RegistryName, callback: CallbackTarget) -> None:
    target: Final = registry(name)
    for index in range(len(target) - 1, -1, -1):
        if target[index] is callback or target[index] == callback:
            del target[index]
            return


def logger_fn(callback: object) -> None:
    from litellm import utils

    utils.user_logger_fn = callback  # rebind-ok: legacy module global read by pre_call


def breadcrumb(kwargs: Mapping[str, object]) -> None:
    from litellm import utils

    if utils.add_breadcrumb is None:
        return
    import litellm
    from litellm.litellm_core_utils import core_helpers

    deep_copy: Final = _legacy(core_helpers.safe_deep_copy)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy helper
    copied: Final = _copied(deep_copy, kwargs)
    hidden: Final = frozenset(("messages", "input", "prompt")) if litellm.turn_off_message_logging else frozenset[str]()
    details: Final = MappingProxyType({key: value for key, value in copied.items() if key not in hidden})
    _legacy(utils.add_breadcrumb)(category="litellm.llm_call", message=f"Keyword Args: {details}", level="info")


def _copied(deep_copy: Callable[..., object], kwargs: Mapping[str, object]) -> Mapping[str, object]:
    try:
        copied: Final = deep_copy(dict(kwargs))  # mutable-ok: safe_deep_copy expects a dict
    except Exception:  # noqa: BLE001  # legacy breadcrumb falls back to the live mapping
        return kwargs
    return _mapping(copied) or kwargs


def prepare_environment() -> None:
    from litellm import utils

    utils.custom_llm_setup()


def applied_guardrails(kwargs: Mapping[str, object]) -> Sequence[str]:
    from litellm.utils import get_applied_guardrails

    return tuple(get_applied_guardrails(dict(kwargs)))  # mutable-ok: legacy helper expects a dict


def _litellm_params(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = kwargs.get("metadata")
    litellm_metadata: Final = kwargs.get("litellm_metadata")
    base: Final = MappingProxyType({"api_base": ""})
    with_metadata: Final = MappingProxyType({**base, "metadata": metadata}) if "metadata" in kwargs else base
    typed_metadata: Final = _mapping(litellm_metadata)
    if typed_metadata is None:
        return with_metadata
    with_litellm_metadata: Final = MappingProxyType({**with_metadata, "litellm_metadata": typed_metadata})
    if metadata:
        return with_litellm_metadata
    copied_metadata: Final = dict(typed_metadata)  # mutable-ok: callbacks may mutate this copy
    return MappingProxyType({**with_litellm_metadata, "metadata": copied_metadata})


def _owned(
    callbacks: Sequence[CallbackTarget] | None,
) -> list[CallbackTarget] | None:  # mutable-ok: Logging appends to these lists
    return list(callbacks) if callbacks is not None else None  # mutable-ok: Logging appends to these lists


def build_logging(
    *,
    call_type: str,
    model: str | None,
    kwargs: Kwargs,
    start_time: datetime.datetime,
    asynchronous: bool,
    dynamic_success: Sequence[CallbackTarget] | None,
    dynamic_async_success: Sequence[CallbackTarget] | None,
    dynamic_failure: Sequence[CallbackTarget] | None,
    guardrails: Sequence[str],
) -> Logging:
    from litellm.litellm_core_utils.cached_imports import get_litellm_logging_class

    function_id: Final = kwargs.get("id")
    trace_id: Final = kwargs.get("litellm_trace_id")
    factory: Final = cast(_LoggingFactory, get_litellm_logging_class())  # cast-ok: legacy constructor is untyped
    logger: Final = factory(
        model=model,
        messages="default-message-value",
        stream=False,
        litellm_call_id=str(kwargs["litellm_call_id"]),
        litellm_trace_id=trace_id if isinstance(trace_id, str) else None,
        function_id=function_id if isinstance(function_id, str) else "",
        call_type=call_type,
        start_time=start_time,
        dynamic_success_callbacks=_owned(dynamic_success),
        dynamic_failure_callbacks=_owned(dynamic_failure),
        dynamic_async_success_callbacks=_owned(dynamic_async_success),
        dynamic_async_failure_callbacks=None,
        kwargs=kwargs,
        applied_guardrails=list(guardrails),  # mutable-ok: legacy constructor annotation is list
        supports_correlation_logging=asynchronous,
    )
    update: Final = cast(_EnvironmentUpdater, logger.update_environment_variables)  # cast-ok: legacy method is untyped
    update(
        model=model,
        user="",
        optional_params=MappingProxyType({}),
        litellm_params=_litellm_params(kwargs),
        stream_options=kwargs.get("stream_options"),
    )
    return logger
