# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Mapping
from typing import Final

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.python_extension.generated.v1 import extension_host_pb2 as pb

from .constants import CALLBACK_HOOKS, GUARDRAIL_HOOKS
from .models import CALLBACK_KIND, GUARDRAIL_KIND, ExtensionConfig, LoadedExtension


class ExtensionLoadError(ValueError):
    pass


def load_extension(config: ExtensionConfig) -> LoadedExtension:
    if config.kind not in (CALLBACK_KIND, GUARDRAIL_KIND):
        raise ExtensionLoadError(f"unsupported extension kind {config.kind}")
    target = _resolve(config.entrypoint)  # rebind-ok: invocation-scoped RPC state
    constructor: Final = _constructor_config(config.constructor_json)
    target = _construct(config.entrypoint, target, constructor)  # rebind-ok: invocation-scoped RPC state
    allowed_hooks: Final = GUARDRAIL_HOOKS if config.kind == GUARDRAIL_KIND else CALLBACK_HOOKS
    hooks = tuple(  # rebind-ok: invocation-scoped RPC state
        sorted(hook for hook in allowed_hooks if _implements_hook(target, hook))
    )  # rebind-ok: invocation-scoped RPC state
    callable_target: Final = callable(target)
    if callable_target and config.kind == CALLBACK_KIND:
        configured_events: Final = constructor.get("callback_events", ("success", "failure"))
        if not isinstance(configured_events, list | tuple):
            raise ExtensionLoadError("callback_events must be an array")
        function_hooks: Final = {  # mutable-ok: LiteLLM compatibility payload
            "success": "async_log_success_event",
            "failure": "async_log_failure_event",
        }
        hooks = tuple(  # rebind-ok: invocation-scoped RPC state
            function_hooks[event] for event in configured_events if isinstance(event, str) and event in function_hooks
        )
    _reject_unsupported_overrides(target, allowed_hooks)
    if not hooks:
        raise ExtensionLoadError(
            f"entrypoint {config.entrypoint!r} does not implement a supported callback or guardrail hook"
        )
    return LoadedExtension(
        config=config,
        target=target,
        hooks=hooks,
        callable_target=callable_target,
        async_callable=_is_async_callable(target),
    )


def config_from_proto(spec: pb.ExtensionSpec) -> ExtensionConfig:
    return ExtensionConfig(
        id=spec.id,
        kind=spec.kind,
        entrypoint=spec.entrypoint,
        constructor_json=spec.constructor_json,
    )


def _resolve(entrypoint: str) -> object:
    if entrypoint.startswith(("s3://", "gcs://")):
        raise ExtensionLoadError("remote module loading is not supported by protocol v1")
    module_name, separator, attribute_name = entrypoint.replace(":", ".", 1).rpartition(".")
    if not separator or not module_name or not attribute_name:
        raise ExtensionLoadError(f"invalid extension entrypoint {entrypoint!r}")
    try:
        module: Final = importlib.import_module(module_name)
        return getattr(module, attribute_name)
    except (AttributeError, ImportError) as error:
        raise ExtensionLoadError(f"failed to import {entrypoint!r}: {error}") from error


def _constructor_config(raw: bytes) -> Mapping[str, object]:
    if not raw:
        return {}  # mutable-ok: LiteLLM compatibility payload
    try:
        value: Final = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExtensionLoadError(f"constructor_json is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ExtensionLoadError("constructor_json must contain an object")
    return value


def _construct(entrypoint: str, target: object, config: Mapping[str, object]) -> object:
    args: Final = config.get(
        "args",
        [],  # mutable-ok: LiteLLM compatibility payload
    )
    kwargs: Final = config.get(
        "kwargs",
        {},  # mutable-ok: LiteLLM compatibility payload
    )
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise ExtensionLoadError("constructor args must be an array and kwargs must be an object")
    if not isinstance(target, type):
        if args or kwargs:
            raise ExtensionLoadError(f"entrypoint {entrypoint!r} is not a class")
        return target
    try:
        return target(*args, **kwargs)
    except Exception as error:
        raise ExtensionLoadError(f"failed to construct {entrypoint!r}: {error}") from error


def _implements_hook(target: object, hook: str) -> bool:
    method: Final = getattr(target, hook, None)
    if not callable(method):
        return False
    target_method: Final = inspect.getattr_static(type(target), hook, None)
    for base in (CustomGuardrail, CustomLogger):
        if isinstance(target, base) and target_method is inspect.getattr_static(base, hook, None):
            return False
    return True


def _reject_unsupported_overrides(target: object, allowed_hooks: frozenset[str]) -> None:
    if not isinstance(target, CustomLogger):
        return
    base: Final = CustomGuardrail if isinstance(target, CustomGuardrail) else CustomLogger
    unsupported: list[str] = []  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    for name in dir(base):
        if name.startswith("_") or name in allowed_hooks:
            continue
        base_method = inspect.getattr_static(base, name, None)
        target_method = inspect.getattr_static(type(target), name, None)
        if callable(base_method) and callable(target_method) and target_method is not base_method:
            unsupported.append(name)
    if unsupported:
        raise ExtensionLoadError(
            f"entrypoint overrides unsupported protocol v1 hooks: {', '.join(sorted(unsupported))}"
        )


def _is_async_callable(target: object) -> bool:
    return inspect.iscoroutinefunction(target) or (
        callable(target) and inspect.iscoroutinefunction(type(target).__call__)
    )
