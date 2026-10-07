"""What the Rust router backend can serve today, decided once per `Router(...)` construction."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_backends.python_router import PythonRouter
from litellm.router_utils.auto_router_model_naming import classify_strategy_router_model

_INIT: Final = cast(Callable[..., None], PythonRouter.__init__)  # cast-ok: only the signature is read
_SIGNATURE: Final = inspect.signature(_INIT)


def _default(parameter: inspect.Parameter) -> object:
    return cast(object, parameter.default)  # cast-ok: inspect types defaults as Any


_DEFAULTS: Final[Mapping[str, object]] = MappingProxyType(
    {name: _default(parameter) for name, parameter in _SIGNATURE.parameters.items()}
)
_CALLBACKS: Final = TypeAdapter(tuple[object, ...])
_MAPPING: Final = TypeAdapter(Mapping[str, object])

SUPPORTED_ARGUMENTS: Final = frozenset(
    {
        "model_list",
        "redis_url",
        "redis_host",
        "redis_port",
        "redis_password",
        "redis_db",
        "num_retries",
        "max_fallbacks",
        "timeout",
        "stream_timeout",
        "set_verbose",
        "debug_level",
        "default_fallbacks",
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "retry_after",
        "retry_policy",
        "model_group_retry_policy",
        "allowed_fails",
        "cooldown_time",
        "disable_cooldowns",
    }
)

_UNSUPPORTED_DEPLOYMENT_KEYS: Final = frozenset({"tpm", "rpm", "max_parallel_requests"})
_UNSUPPORTED_LITELLM_PARAMS: Final = frozenset(
    {"order", "tags", "tag_regex", "tpm", "rpm", "max_parallel_requests", "silent_model"}
)
_UNSUPPORTED_MODEL_INFO: Final = frozenset({"team_id", "order", "allowed_fails", "allowed_fails_policy", "blocked"})
_CALLBACK_HOOKS: Final = (
    "async_filter_deployments",
    "async_pre_call_check",
    "log_success_fallback_event",
    "log_failure_fallback_event",
)
_FALLBACK_ARGUMENTS: Final = ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks", "default_fallbacks")


class _RawDeployment(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    model_name: str
    litellm_params: Mapping[str, object]
    model_info: Mapping[str, object] | None = None


_MODEL_LIST: Final = TypeAdapter(tuple[_RawDeployment, ...])


def bind_arguments(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """The explicitly passed `Router(...)` arguments by parameter name."""
    bound: Final = _SIGNATURE.bind(None, *args, **kwargs)
    arguments: Final = cast(Mapping[str, object], bound.arguments)  # cast-ok: inspect types bound values as Any
    return {name: value for name, value in arguments.items() if name != "self"}


def unsupported_reason(arguments: Mapping[str, object]) -> str | None:
    """Why the Rust backend cannot serve this configuration, or None when it can."""
    return next(
        (
            reason
            for reason in (
                _argument_reason(arguments),
                _fallbacks_reason(arguments),
                _model_list_reason(arguments.get("model_list")),
                _global_reason(),
            )
            if reason is not None
        ),
        None,
    )


def _argument_reason(arguments: Mapping[str, object]) -> str | None:
    return next(
        (
            f"Router argument {name!r} is not supported by the Rust router yet"
            for name, value in arguments.items()
            if name not in SUPPORTED_ARGUMENTS and value != _DEFAULTS[name]
        ),
        None,
    )


def _fallbacks_reason(arguments: Mapping[str, object]) -> str | None:
    """Fallback chains the Rust router reads: group names only, and no key Python would match to a group
    by prefixing the group's provider, which it infers from the cost map."""
    groups: Final = _group_names(arguments.get("model_list"))
    configured: Final = tuple(
        (name, arguments.get(name) or cast(object, getattr(litellm, name, None)))  # cast-ok: untyped global
        for name in _FALLBACK_ARGUMENTS
    )
    return next(
        (f"{name} entries other than group names" for name, value in configured if not _plain_chain(value, groups)),
        None,
    )


def _group_names(model_list: object) -> frozenset[str]:
    try:
        return frozenset(deployment.model_name for deployment in _MODEL_LIST.validate_python(model_list or ()))
    except ValidationError:
        return frozenset()


def _plain_chain(value: object, groups: frozenset[str]) -> bool:
    entries: Final = _CALLBACKS.validate_python(value) if isinstance(value, (list, tuple)) else ()
    return all(_plain_entry(entry, groups) for entry in entries)


def _plain_entry(entry: object, groups: frozenset[str]) -> bool:
    if isinstance(entry, str):
        return True
    chains: Final[Mapping[str, object]] = _MAPPING.validate_python(entry) if isinstance(entry, Mapping) else {}
    return bool(chains) and all(_plain_chain_of(key, targets, groups) for key, targets in chains.items())


def _plain_chain_of(key: str, targets: object, groups: frozenset[str]) -> bool:
    return (
        key.rpartition("/")[2] not in groups - {key}
        and isinstance(targets, (list, tuple))
        and all(isinstance(target, str) for target in _CALLBACKS.validate_python(targets))
    )


def _model_list_reason(model_list: object) -> str | None:
    if model_list is None:
        return None
    try:
        deployments: Final = _MODEL_LIST.validate_python(model_list)
    except ValidationError:
        return "model_list entries the Rust router cannot read"
    return next(
        (reason for reason in map(_deployment_reason, deployments) if reason is not None),
        None,
    )


def _deployment_reason(deployment: _RawDeployment) -> str | None:
    extra: Final = deployment.model_extra or {}
    model: Final = deployment.litellm_params.get("model")
    keys: Final = (
        *(f"{key}" for key in _UNSUPPORTED_DEPLOYMENT_KEYS.intersection(extra)),
        *(f"litellm_params.{key}" for key in _UNSUPPORTED_LITELLM_PARAMS.intersection(deployment.litellm_params)),
        *(f"model_info.{key}" for key in _UNSUPPORTED_MODEL_INFO.intersection(deployment.model_info or {})),
    )
    if "*" in deployment.model_name:
        return f"wildcard model_name {deployment.model_name!r}"
    if isinstance(model, str) and classify_strategy_router_model(model) is not None:
        return f"strategy router deployment {model!r}"
    if keys:
        return f"deployment {deployment.model_name!r} sets {', '.join(sorted(keys))}"
    return None


def _global_reason() -> str | None:
    if litellm.model_alias_map:
        return "litellm.model_alias_map is set"
    hooked: Final = tuple(
        type(callback).__name__
        for callback in _CALLBACKS.validate_python(cast(object, litellm.callbacks))  # cast-ok: untyped global
        if isinstance(callback, CustomLogger)
        and any(getattr(type(callback), hook) is not getattr(CustomLogger, hook) for hook in _CALLBACK_HOOKS)
    )
    if hooked:
        return f"callbacks {', '.join(hooked)} filter or pre-check deployments"
    return None
