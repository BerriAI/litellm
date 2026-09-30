from __future__ import annotations

import ast
import functools
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.bug_report import (
    KNOWN_PROVIDERS,
    BugReport,
    EnvironmentReport,
    allowlisted,
    build_bug_report,
    build_environment_report,
)
from litellm.proxy._types import ConfigGeneralSettings
from litellm.router_utils.routing_groups import VALID_ROUTING_STRATEGIES
from litellm.types.caching import LiteLLMCacheType
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams, SupportedGuardrailIntegrations
from litellm.types.secret_managers.main import KeyManagementSystem

_OBJECT_MAP: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_OBJECT_LIST: Final[TypeAdapter[tuple[object, ...]]] = TypeAdapter(tuple[object, ...])
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
CREDENTIAL_KEY_PARTS: Final = frozenset(
    {
        "key",
        "keys",
        "secret",
        "secrets",
        "token",
        "password",
        "passwd",
        "credential",
        "credentials",
        "url",
        "uri",
        "dsn",
        "host",
        "hosts",
        "base",
        "endpoint",
        "cert",
        "pem",
        "salt",
    }
)
ENUM_KEYS_WITH_CREDENTIAL_PARTS: Final = frozenset({"key_management_system"})


def _object_map(value: object) -> Mapping[str, object]:
    try:
        return _OBJECT_MAP.validate_python(value)
    except ValidationError:
        return MappingProxyType({})


def _object_list(value: object) -> Sequence[object]:
    try:
        return _OBJECT_LIST.validate_python(value)
    except ValidationError:
        return ()


@functools.cache
def _known_values() -> frozenset[str]:
    from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

    return frozenset(
        (
            *VALID_ROUTING_STRATEGIES,
            *KNOWN_PROVIDERS,
            *litellm._known_custom_logger_compatible_callbacks,  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownArgumentType]  # untyped List of the callback Literal's args, no public alias
            *CustomLoggerRegistry.CALLBACK_CLASS_STR_TO_CLASS_TYPE,
            *(member.value for member in LiteLLMCacheType),
            *(member.value for member in KeyManagementSystem),
            *(member.value for member in SupportedGuardrailIntegrations),
            *(member.value for member in GuardrailEventHooks),
        )
    )


def _module_level_names(node: ast.stmt) -> tuple[str, ...]:
    match node:
        case ast.Assign(targets=targets):
            return tuple(target.id for target in targets if isinstance(target, ast.Name))
        case ast.AnnAssign(target=ast.Name(id=name)):
            return (name,)
        case ast.ImportFrom(names=aliases):
            return tuple(alias.asname or alias.name for alias in aliases)
        case _:
            return ()


@functools.cache
def _litellm_settings_keys() -> frozenset[str]:
    tree: Final = ast.parse(Path(litellm.__file__).read_text())
    return frozenset(name for node in tree.body for name in _module_level_names(node))


@functools.cache
def _router_settings_keys() -> frozenset[str]:
    from litellm.router import Router

    return frozenset(name for name in inspect.signature(Router.__init__).parameters if name != "self")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped params, only names are read


@functools.cache
def _cache_params_keys() -> frozenset[str]:
    from litellm.caching.caching import Cache

    return frozenset(name for name in inspect.signature(Cache.__init__).parameters if name != "self")  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped params, only names are read


def _is_credential_key(key: str) -> bool:
    return key not in ENUM_KEYS_WITH_CREDENTIAL_PARTS and not CREDENTIAL_KEY_PARTS.isdisjoint(key.lower().split("_"))


def _render_json(key: str, value: JsonValue) -> str | None:
    match value:
        case bool():
            return str(value).lower()
        case str():
            return value if value in _known_values() and not _is_credential_key(key) else None
        case list():
            known_items: Final = tuple(rendered for item in value if (rendered := _render_json(key, item)) is not None)
            return f"[{', '.join(known_items)}]" if known_items else None
        case _:
            return None


def _render(key: str, value: object) -> str | None:
    try:
        return _render_json(key, _JSON.validate_python(value))
    except ValidationError:
        return None


def _section_lines(section: str, values: Mapping[str, object], known_keys: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        f"{section}.{key} = {rendered}"
        for key, value in values.items()
        if key in known_keys and (rendered := _render(key, value)) is not None
    )


def _guardrail_lines(guardrails: object) -> tuple[str, ...]:
    known_keys: Final = frozenset(LitellmParams.model_fields)
    return tuple(
        line
        for index, guardrail in enumerate(_object_list(guardrails))
        for line in _section_lines(
            f"guardrails[{index}].litellm_params", _object_map(_object_map(guardrail).get("litellm_params")), known_keys
        )
    )


def _deployment_provider(model: object) -> str | None:
    prefix: Final = model.split("/", 1)[0] if isinstance(model, str) and "/" in model else None
    return allowlisted(prefix, KNOWN_PROVIDERS)


def _model_list_lines(model_list: object) -> tuple[str, ...]:
    providers: Final = tuple(
        sorted(
            frozenset(
                provider
                for deployment in _object_list(model_list)
                if (
                    provider := _deployment_provider(
                        _object_map(_object_map(deployment).get("litellm_params")).get("model")
                    )
                )
                is not None
            )
        )
    )
    return (f"model_list[*].provider = [{', '.join(providers)}]",) if providers else ()


def safe_config_lines(config: Mapping[str, object], general_settings: Mapping[str, object]) -> tuple[str, ...]:
    litellm_settings: Final = _object_map(config.get("litellm_settings"))
    return (
        *_section_lines("general_settings", general_settings, frozenset(ConfigGeneralSettings.model_fields)),
        *_section_lines("litellm_settings", litellm_settings, _litellm_settings_keys()),
        *_section_lines(
            "litellm_settings.cache_params", _object_map(litellm_settings.get("cache_params")), _cache_params_keys()
        ),
        *_section_lines("router_settings", _object_map(config.get("router_settings")), _router_settings_keys()),
        *_guardrail_lines(config.get("guardrails")),
        *_model_list_lines(config.get("model_list")),
    )


def _proxy_config_lines() -> tuple[str, ...]:
    from litellm.proxy import proxy_server

    return safe_config_lines(
        proxy_server.proxy_config.config,
        _object_map(proxy_server.general_settings),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # bare dict global, validated by _object_map
    )


def build_proxy_environment_report() -> EnvironmentReport:
    return build_environment_report(surface="proxy", config_lines=_proxy_config_lines())


def build_proxy_bug_report(
    exc: BaseException,
    *,
    call_type: str | None = None,
    custom_llm_provider: object = None,
    stream: object = None,
) -> BugReport:
    return build_bug_report(
        exc,
        surface="proxy",
        call_type=call_type,
        custom_llm_provider=custom_llm_provider,
        stream=stream,
        config_lines=_proxy_config_lines(),
    )
