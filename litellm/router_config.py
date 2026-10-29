"""
The router's configuration, built outside the router.

``RouterConfig`` is static: the constructor arguments, frozen for the router's life.
``RouterDynamicConfig`` is a snapshot of the process settings the router reads while it serves
(``litellm.num_retries``, ``litellm.model_cost``, the fallback lists, ...). A call reads one snapshot
at its start. This module is the only router code that reads ``litellm.*`` settings.
"""

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import litellm

if TYPE_CHECKING:
    from litellm.caching.caching import Cache
    from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """The arguments the router was built with, exactly as passed (unset ones are absent)."""

    arguments: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def get(self, name: str, default: object = None) -> object:
        return self.arguments.get(name, default)


@dataclass(frozen=True, slots=True)
class RouterDynamicConfig:
    """Process settings the router reads per call. Values are the live objects, not copies."""

    version: int
    num_retries: int | None
    max_fallbacks: int | None
    router_max_fallbacks: int
    allowed_fails: int
    request_timeout: float
    fallbacks: Sequence[object] | None
    default_fallbacks: Sequence[str] | None
    context_window_fallbacks: Sequence[object] | None
    content_policy_fallbacks: Sequence[object] | None
    drop_params: bool | None
    provider_list: Sequence[str]
    expose_router_debug_in_errors: bool
    redact_user_api_key_info: bool | None
    model_alias_map: Mapping[str, str]
    model_cost: Mapping[str, object]
    known_custom_logger_compatible_callbacks: Sequence[str]
    exception_types: Sequence[type[BaseException]]
    callbacks: Sequence[object]
    success_callback: Sequence[object]
    failure_callback: Sequence[object]
    input_callback: Sequence[object]
    callback_manager: "LoggingCallbackManager"
    cache: "Cache | None"


_versions: Final = itertools.count(1)


def detached(value: object) -> object:
    """``value`` with its plain containers (dict, list, tuple, set) copied, all the way down, so the
    caller's own objects can't change the router afterwards. Other objects (clients, caches,
    callbacks, pydantic models) are kept as they are."""
    if isinstance(value, dict):
        return {key: detached(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detached(item) for item in value]
    if isinstance(value, tuple) and type(value) is tuple:
        return tuple(detached(item) for item in value)
    if isinstance(value, set):
        return {detached(item) for item in value}
    return value


def build_router_config(arguments: Mapping[str, object]) -> RouterConfig:
    """The constructor arguments, detached from the caller's objects: settings are static."""
    return RouterConfig(arguments=MappingProxyType({name: detached(value) for name, value in arguments.items()}))


def current_dynamic_config() -> RouterDynamicConfig:
    """A snapshot of the process settings as they are right now."""
    return RouterDynamicConfig(
        version=next(_versions),
        num_retries=litellm.num_retries,
        max_fallbacks=litellm.max_fallbacks,
        router_max_fallbacks=litellm.ROUTER_MAX_FALLBACKS,
        allowed_fails=litellm.allowed_fails,
        request_timeout=litellm.request_timeout,
        fallbacks=litellm.fallbacks,
        default_fallbacks=litellm.default_fallbacks,
        context_window_fallbacks=litellm.context_window_fallbacks,
        content_policy_fallbacks=litellm.content_policy_fallbacks,
        drop_params=litellm.drop_params,
        provider_list=litellm.provider_list,
        expose_router_debug_in_errors=litellm.expose_router_debug_in_errors,
        redact_user_api_key_info=litellm.redact_user_api_key_info,
        model_alias_map=litellm.model_alias_map,
        model_cost=litellm.model_cost,
        known_custom_logger_compatible_callbacks=litellm._known_custom_logger_compatible_callbacks,
        exception_types=litellm.LITELLM_EXCEPTION_TYPES,
        callbacks=litellm.callbacks,
        success_callback=litellm.success_callback,
        failure_callback=litellm.failure_callback,
        input_callback=litellm.input_callback,
        callback_manager=litellm.logging_callback_manager,
        cache=litellm.cache,
    )


_LIVE_SOURCES: Final = MappingProxyType(
    {
        "num_retries": "num_retries",
        "max_fallbacks": "max_fallbacks",
        "router_max_fallbacks": "ROUTER_MAX_FALLBACKS",
        "allowed_fails": "allowed_fails",
        "request_timeout": "request_timeout",
        "fallbacks": "fallbacks",
        "default_fallbacks": "default_fallbacks",
        "context_window_fallbacks": "context_window_fallbacks",
        "content_policy_fallbacks": "content_policy_fallbacks",
        "drop_params": "drop_params",
        "provider_list": "provider_list",
        "expose_router_debug_in_errors": "expose_router_debug_in_errors",
        "redact_user_api_key_info": "redact_user_api_key_info",
        "model_alias_map": "model_alias_map",
        "model_cost": "model_cost",
        "known_custom_logger_compatible_callbacks": "_known_custom_logger_compatible_callbacks",
        "exception_types": "LITELLM_EXCEPTION_TYPES",
        "callbacks": "callbacks",
        "success_callback": "success_callback",
        "failure_callback": "failure_callback",
        "input_callback": "input_callback",
        "callback_manager": "logging_callback_manager",
        "cache": "cache",
    }
)


class LiveDynamicConfig:
    """``RouterDynamicConfig``'s fields read from the process settings at access time, with no snapshot."""

    __slots__ = ()

    def __getattr__(self, name: str) -> object:
        source: Final = _LIVE_SOURCES.get(name)
        if source is None:
            raise AttributeError(name)
        return getattr(litellm, source)


live_dynamic_config: Final = LiveDynamicConfig()

DynamicConfigSource = Callable[[], RouterDynamicConfig]
