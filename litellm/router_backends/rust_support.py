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
_CALLBACK_HOOKS: Final = ("async_filter_deployments", "async_pre_call_check")


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
