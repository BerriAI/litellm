from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from typing import (  # noqa: TID251  # narrows legacy untyped registries at the boundary
    TYPE_CHECKING,
    Final,
    Literal,
    Protocol,
    cast,
)

from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

CallbackTarget = str | Callable[..., object] | CustomLogger
RegistryName = Literal["input", "async_input", "success", "async_success", "failure", "async_failure", "callbacks"]

_REGISTRY_ATTRIBUTES: Final[Mapping[RegistryName, str]] = {
    "input": "input_callback",
    "async_input": "_async_input_callback",
    "success": "success_callback",
    "async_success": "_async_success_callback",
    "failure": "failure_callback",
    "async_failure": "_async_failure_callback",
    "callbacks": "callbacks",
}


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
        dynamic_success_callbacks: list[CallbackTarget] | None,
        dynamic_failure_callbacks: list[CallbackTarget] | None,
        dynamic_async_success_callbacks: list[CallbackTarget] | None,
        dynamic_async_failure_callbacks: list[CallbackTarget] | None,
        kwargs: dict[str, object],
        applied_guardrails: list[str],
        supports_correlation_logging: bool,
    ) -> Logging: ...


class _EnvironmentUpdater(Protocol):
    def __call__(
        self,
        *,
        model: str | None,
        user: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream_options: object,
    ) -> None: ...


def registry(name: RegistryName) -> MutableSequence[CallbackTarget]:
    import litellm

    return cast(
        MutableSequence[CallbackTarget], getattr(litellm, _REGISTRY_ATTRIBUTES[name])
    )  # cast-ok: legacy module-level lists are untyped


def is_async_callable(callback: object) -> bool:
    from litellm.litellm_core_utils.cached_imports import get_coroutine_checker

    return get_coroutine_checker().is_async_callable(callback)


def is_known_name(callback: str) -> bool:
    import litellm

    known: Final = cast(Sequence[str], litellm._known_custom_logger_compatible_callbacks)  # pyright: ignore[reportPrivateUsage]  # cast-ok: registry list has no public typed accessor
    return callback in known


def resolve_named_integration(callback: str) -> CustomLogger | None:
    from litellm.litellm_core_utils import litellm_logging

    resolve: Final = cast(  # cast-ok: legacy factory is untyped at its definition
        Callable[..., CustomLogger | None],
        litellm_logging._init_custom_logger_compatible_class,  # pyright: ignore[reportPrivateUsage]  # legacy factory
    )
    return resolve(callback, internal_usage_cache=None, llm_router=None)


def async_success_registry_has_type(callback: object) -> bool:
    return any(type(existing) is type(callback) for existing in registry("async_success"))


def bootstrap_pending() -> bool:
    from litellm import utils

    return not utils.callback_list


def bootstrap(function_id: str | None) -> None:
    from litellm import utils
    from litellm.litellm_core_utils import cached_imports

    combined: Final = list({*registry("input"), *registry("success"), *registry("failure")})
    utils.callback_list = cast(
        list[str], combined
    )  # rebind-ok: legacy module global consumed by set_callbacks  # cast-ok: legacy list annotation is narrower than its contents
    set_callbacks: Final = cast(Callable[..., None], cached_imports.get_set_callbacks())  # pyright: ignore[reportUnknownMemberType]  # cast-ok: cached import is untyped
    set_callbacks(callback_list=combined, function_id=function_id)


def expand_named(callback: str, event: Literal["success", "failure"]) -> None:
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

    add_breadcrumb: Final = cast(
        Callable[..., None] | None, utils.add_breadcrumb
    )  # cast-ok: legacy sentry hook is untyped
    if add_breadcrumb is None:
        return
    import litellm
    from litellm.litellm_core_utils import core_helpers

    deep_copy: Final = cast(  # cast-ok: legacy helper is untyped
        Callable[[dict[str, object]], dict[str, object]],
        core_helpers.safe_deep_copy,  # pyright: ignore[reportUnknownMemberType]  # legacy helper
    )
    try:
        copied: dict[str, object] = deep_copy(dict(kwargs))
    except Exception:  # noqa: BLE001  # legacy breadcrumb falls back to the live mapping
        copied = dict(kwargs)
    hidden: Final = frozenset(("messages", "input", "prompt")) if litellm.turn_off_message_logging else frozenset[str]()
    details: Final = {key: value for key, value in copied.items() if key not in hidden}
    add_breadcrumb(category="litellm.llm_call", message=f"Keyword Args: {details}", level="info")


def prepare_environment() -> None:
    from litellm import utils

    utils.custom_llm_setup()


def applied_guardrails(kwargs: Mapping[str, object]) -> list[str]:
    from litellm.utils import get_applied_guardrails

    return get_applied_guardrails(dict(kwargs))


def build_logging(
    *,
    call_type: str,
    model: str | None,
    kwargs: dict[str, object],
    start_time: datetime.datetime,
    asynchronous: bool,
    dynamic_success: Sequence[CallbackTarget] | None,
    dynamic_async_success: Sequence[CallbackTarget] | None,
    dynamic_failure: Sequence[CallbackTarget] | None,
    guardrails: Sequence[str],
) -> Logging:
    from litellm.litellm_core_utils.cached_imports import get_litellm_logging_class

    function_id: Final = kwargs.get("id")
    metadata: Final = kwargs.get("metadata")
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
        dynamic_success_callbacks=list(dynamic_success) if dynamic_success is not None else None,
        dynamic_failure_callbacks=list(dynamic_failure) if dynamic_failure is not None else None,
        dynamic_async_success_callbacks=list(dynamic_async_success) if dynamic_async_success is not None else None,
        dynamic_async_failure_callbacks=None,
        kwargs=kwargs,
        applied_guardrails=list(guardrails),
        supports_correlation_logging=asynchronous,
    )
    litellm_metadata: Final = kwargs.get("litellm_metadata")
    litellm_params: Final[dict[str, object]] = {
        "api_base": "",
        **({"metadata": kwargs["metadata"]} if "metadata" in kwargs else {}),
        **(
            {
                "litellm_metadata": litellm_metadata,
                **(
                    {} if metadata else {"metadata": dict(cast(Mapping[str, object], litellm_metadata))}
                ),  # cast-ok: isinstance narrows only to dict[Unknown, Unknown]
            }
            if isinstance(litellm_metadata, dict)
            else {}
        ),
    }
    update: Final = cast(_EnvironmentUpdater, logger.update_environment_variables)  # cast-ok: legacy method is untyped
    update(
        model=model,
        user="",
        optional_params={},
        litellm_params=litellm_params,
        stream_options=kwargs.get("stream_options"),
    )
    return logger
