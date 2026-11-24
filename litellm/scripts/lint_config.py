"""Lint a LiteLLM proxy config.yaml without starting the proxy.

Usage: python -m litellm.scripts.lint_config path/to/config.yaml [--strict]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import fnmatch
import inspect
import itertools
import math
import os
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cache, reduce
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, get_args
from unittest.mock import patch

import yaml
from pydantic import BaseModel, TypeAdapter, ValidationError

PathKey: TypeAlias = "tuple[str | int, ...]"
Severity: TypeAlias = Literal["error", "warning"]

_MAPPING: Final = TypeAdapter(dict[str, object])
_SEQUENCE: Final = TypeAdapter(list[object])
_ENV_PREFIX: Final = "os.environ/"
_TYPO_CUTOFF: Final = 0.8
_FALLBACK_KEYS: Final = ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks")
_MODEL_ENTRY_KEYS: Final = frozenset({"model_name", "litellm_params", "model_info"})


@dataclass(frozen=True)
class Finding:
    severity: Severity
    path: PathKey
    message: str
    line: int | None = None


def _mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING.validate_python(value, strict=True)
    except ValidationError:
        return None


def _sequence(value: object) -> Sequence[object] | None:
    try:
        return _SEQUENCE.validate_python(value, strict=True)
    except ValidationError:
        return None


def _dotted(path: PathKey) -> str:
    return "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in path).lstrip(".") or "<root>"


def _node_events(node: yaml.Node, path: PathKey) -> Iterator[tuple[PathKey, int] | Finding]:
    yield (path, node.start_mark.line + 1)
    if isinstance(node, yaml.MappingNode):
        keys: Final = tuple(str(key_node.value) for key_node, _ in node.value)
        for index, (key_node, value_node) in enumerate(node.value):
            key = keys[index]
            line = key_node.start_mark.line + 1
            if key in keys[:index]:
                yield Finding("error", (*path, key), "duplicate key, only the last value is used", line)
            yield ((*path, key), line)
            yield from itertools.islice(_node_events(value_node, (*path, key)), 1, None)
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            yield from _node_events(item, (*path, index))


def _source_keys(pattern: str, *relative_globs: str) -> frozenset[str]:
    import litellm

    root: Final = Path(litellm.__file__).parent
    regex: Final = re.compile(pattern)
    sources: Final = itertools.chain.from_iterable(root.glob(relative_glob) for relative_glob in relative_globs)
    texts: Final = (source.read_text(encoding="utf-8", errors="ignore") for source in sources)
    return frozenset(itertools.chain.from_iterable(regex.findall(text) for text in texts))


@cache
def _known_top_level_keys() -> frozenset[str]:
    from litellm.proxy._types import ConfigYAML

    return (
        frozenset(ConfigYAML.model_fields)
        | _source_keys(r'\bconfig(?:\.get\(|\.pop\(|\[)\s*"(\w+)"', "proxy/proxy_server.py")
        | {"include"}
    )


@cache
def _known_general_settings() -> frozenset[str]:
    from litellm.proxy._types import ConfigGeneralSettings

    return frozenset(ConfigGeneralSettings.model_fields) | _source_keys(
        r'general_settings(?:\.get\(|\[)\s*"(\w+)"', "proxy/**/*.py", "router.py"
    )


@cache
def _quoted_names(*relative_globs: str) -> frozenset[str]:
    return _source_keys(r'"([a-z]\w+)"', *relative_globs)


def _is_general_setting(key: str) -> bool:
    return (
        key in _known_general_settings()
        or key in _quoted_names("proxy/**/*.py", "router.py")
        or key in _source_keys(r"pub (\w+):", "rust/crates/config/src/settings.rs")
    )


@cache
def _known_litellm_settings() -> frozenset[str]:
    return _source_keys(r'litellm_settings(?:\.get\(|\[)\s*"(\w+)"', "proxy/**/*.py") | _source_keys(
        r'\bkey == "(\w+)"', "proxy/proxy_server.py"
    )


@cache
def _known_router_settings() -> frozenset[str]:
    import litellm
    from litellm.types.router import UpdateRouterConfig

    return frozenset(UpdateRouterConfig.model_fields) | (
        frozenset(inspect.signature(litellm.Router.__init__).parameters) - {"self"}
    )


@cache
def _known_routing_strategies() -> frozenset[str]:
    from litellm.types.router import RoutingStrategy, RoutingStrategyName

    return frozenset(get_args(RoutingStrategyName)) | {member.value for member in RoutingStrategy} | {"priority"}


def _is_module_setting(key: str) -> bool:
    import litellm

    return key in _known_litellm_settings() or hasattr(litellm, key)


def _is_litellm_setting(key: str) -> bool:
    return _is_module_setting(key) or key in _quoted_names("**/*.py")


def _suggestion(key: str, known: frozenset[str]) -> str:
    close: Final = difflib.get_close_matches(key, sorted(known), n=1, cutoff=_TYPO_CUTOFF)
    return f', did you mean "{close[0]}"?' if close else ""


def _unknown_keys(
    section: Mapping[str, object], path: PathKey, known: frozenset[str], *, only_typos: bool = False
) -> Iterator[Finding]:
    for key in section:
        if key in known:
            continue
        hint = _suggestion(key, known)
        if hint or not only_typos:
            yield Finding("warning", (*path, key), f"unknown key{hint}")


def _type_findings(model: type[BaseModel], section: Mapping[str, object], path: PathKey) -> Iterator[Finding]:
    try:
        model.model_validate({key: value for key, value in section.items() if key in model.model_fields})
    except ValidationError as error:
        for detail in error.errors():
            value = detail["input"]
            if isinstance(value, str) and value.startswith(_ENV_PREFIX):
                continue
            yield Finding("warning", (*path, *detail["loc"]), f"unexpected value: {detail['msg']}")


def _strings(value: object, path: PathKey) -> Iterator[tuple[PathKey, str]]:
    if isinstance(value, str):
        yield (path, value)
        return
    as_mapping: Final = _mapping(value)
    if as_mapping is not None:
        for key, item in as_mapping.items():
            yield from _strings(item, (*path, key))
        return
    for index, item in enumerate(_sequence(value) or ()):
        yield from _strings(item, (*path, index))


def _env_findings(config: Mapping[str, object], environ: Mapping[str, str]) -> Iterator[Finding]:
    declared: Final = _mapping(config.get("environment_variables")) or {}
    for path, value in _strings(config, ()):
        name = value.removeprefix(_ENV_PREFIX)
        if value.startswith(_ENV_PREFIX) and name not in environ and name not in declared:
            yield Finding("warning", path, f"environment variable {name} is not set")


_ALTERNATIVE_AUTH: Final = re.compile(r"role|profile|(?<!session_)token|credentials")
_SECRET_NAME: Final = re.compile(r"KEY|SECRET|TOKEN")
_UNMAPPED_MODEL_KEYS: Final = ("input_cost_per_token", "output_cost_per_token", "max_input_tokens", "max_output_tokens")


@cache
def _model_info_types() -> Mapping[str, frozenset[type]]:
    import litellm
    from litellm.types.router import ModelInfo as RouterModelInfo
    from litellm.types.utils import ModelInfo

    entries: Final = [entry for name, entry in litellm.model_cost.items() if name != "sample_spec"]
    names: Final = (
        frozenset(itertools.chain.from_iterable(entries))
        | frozenset(ModelInfo.__annotations__)
        | frozenset(RouterModelInfo.model_fields)
    )
    return MappingProxyType(
        {name: frozenset(type(entry[name]) for entry in entries if entry.get(name) is not None) for name in names}
    )


def _number(value: object, *, parse_text: bool) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if parse_text and isinstance(value, str):
        with contextlib.suppress(ValueError):
            return float(value)
    return None


def _is_price_key(key: str) -> bool:
    return "cost" in key or "_multiplier" in key or key.endswith("_pricing")


def _model_info_findings(
    model: str,
    params: Mapping[str, object],
    model_info: Mapping[str, object],
    path: PathKey,
    *,
    require_explicit_prices: bool,
) -> Iterator[Finding]:
    import litellm
    from litellm.types.router import ModelInfo as RouterModelInfo
    from litellm.types.utils import CustomPricingLiteLLMParams
    from litellm.utils import get_model_info_helper

    info_path: Final = (*path, "model_info")
    types: Final = _model_info_types()
    yield from _unknown_keys(model_info, info_path, frozenset(types), only_typos=True)
    yield from _type_findings(RouterModelInfo, model_info, info_path)
    for key, value in model_info.items():
        seen: Final = types.get(key, frozenset())
        if not seen or not seen <= {int, float} or value is None:
            continue
        if isinstance(value, str) and value.startswith(_ENV_PREFIX):
            continue
        number: Final = _number(value, parse_text=key in RouterModelInfo.model_fields)
        if number is None:
            if key not in RouterModelInfo.model_fields:
                yield Finding("warning", (*info_path, key), f"unexpected value: expected a number, got {value!r}")
        elif isinstance(number, float) and not math.isfinite(number):
            yield Finding("warning", (*info_path, key), f"unexpected value: must be a finite number, got {value!r}")
        elif number < 0:
            yield Finding("warning", (*info_path, key), f"unexpected value: must not be negative, got {value!r}")
    base_model: Final = model_info.get("base_model")
    lookup: Final = base_model if isinstance(base_model, str) and base_model else model
    if "*" in lookup:
        return
    mapped: Mapping[str, object] | None
    try:
        mapped = litellm.model_cost.get(get_model_info_helper(model=lookup, catalog_only=True)["key"])
    except Exception:  # noqa: BLE001  # get_model_info_helper raises a bare Exception for unmapped models
        mapped = None
    if mapped is None:
        missing: Final = [
            key
            for key in _UNMAPPED_MODEL_KEYS
            if model_info.get(key) is None and (params.get(key) is None or "cost" not in key)
        ]
        if missing:
            yield Finding(
                "warning",
                info_path if model_info else path,
                f"{lookup} is not in LiteLLM's model cost map, so cost tracking and token limits need "
                f"model_info.base_model or model_info.{', model_info.'.join(missing)}",
            )
        return
    if not require_explicit_prices:
        return
    unset: Final = sorted(
        key
        for key, value in mapped.items()
        if _is_price_key(key)
        and value is not None
        and model_info.get(key) is None
        and (params.get(key) is None or key not in CustomPricingLiteLLMParams.model_fields)
    )
    if unset:
        yield Finding(
            "error",
            info_path if model_info else path,
            f"{lookup} would take these prices from LiteLLM's model cost map, set them in model_info: "
            f"{', '.join(unset)}",
        )


_AZURE_CLIENT_LOGIN: Final = ("tenant_id", "client_id", "client_secret")


def _with_declared(environ: Mapping[str, str], item: tuple[object, object]) -> Mapping[str, str]:
    text: Final = "" if item[1] is None else str(item[1])
    value: Final = environ.get(text.removeprefix(_ENV_PREFIX)) if text.startswith(_ENV_PREFIX) else text
    return environ if value is None else {**environ, str(item[0]): value}


def _effective_environ(declared: Mapping[str, object], environ: Mapping[str, str]) -> Mapping[str, str]:
    return reduce(_with_declared, declared.items(), environ)


def _provider_model(params: Mapping[str, object], model: str) -> str:
    provider: Final = params.get("custom_llm_provider")
    if not isinstance(provider, str) or not provider or model.startswith(f"{provider}/"):
        return model
    return f"{provider}/{model}"


def _credential_findings(
    model: str, params: Mapping[str, object], path: PathKey, environ: Mapping[str, str]
) -> Iterator[Finding]:
    import litellm
    from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider
    from litellm.types.router import GenericLiteLLMParams

    if declared_authenticating_provider(model) is not None or params.get("litellm_credential_name") is not None:
        return

    def given(name: str) -> str | None:
        return None if params.get(name) is None else str(params[name])

    with patch.dict(os.environ, {name: value for name, value in environ.items() if value}, clear=True):
        try:
            required: Final[list[str]] = litellm.validate_environment(
                model=model, api_key=given("api_key"), api_base=given("api_base"), api_version=given("api_version")
            )["missing_keys"]
        except Exception:  # noqa: BLE001  # an unrecognized provider has no known requirements
            return
    missing: Final = [name for name in required if not environ.get(name)]
    prefixes: Final = {name.partition("_")[0].lower() for name in missing}
    provider: Final = prefixes | {prefix.removesuffix("ai") for prefix in prefixes}
    fields: Final = [name for name in GenericLiteLLMParams.model_fields if name.partition("_")[0] in provider | {"api"}]
    set_params: Final = {key for key, value in params.items() if value is not None}
    has_other_auth: Final = any(_ALTERNATIVE_AUTH.search(key) for key in set_params.intersection(fields)) or (
        "azure" in provider
        and all(name in set_params or environ.get(f"AZURE_{name.upper()}") for name in _AZURE_CLIENT_LOGIN)
    )
    for name in missing:
        setting: Final = name.lower().partition("_")[2]
        names: Final = [field for field in fields if setting in (field, field.partition("_")[2])]
        if (
            set_params.intersection(names)
            or (has_other_auth and _SECRET_NAME.search(name))
            or (name.endswith("_TOKEN") and "api_key" in set_params)
        ):
            continue
        hint: Final = f" or litellm_params.{names[0]}" if names else ""
        yield Finding("warning", path, f"{model} needs {name} in the environment{hint}")


def _model_list_findings(
    model_list: object, environ: Mapping[str, str], *, require_explicit_prices: bool
) -> Iterator[Finding]:
    import litellm
    from litellm.types.router import GenericLiteLLMParams

    entries: Final = _sequence(model_list)
    if entries is None:
        yield Finding("error", ("model_list",), "must be a list of model entries")
        return
    param_keys: Final = (
        frozenset(GenericLiteLLMParams.model_fields)
        | frozenset(inspect.signature(litellm.completion).parameters)
        | frozenset(inspect.signature(litellm.Router.__init__).parameters)
    )
    for index, raw_entry in enumerate(entries):
        path: PathKey = ("model_list", index)
        entry = _mapping(raw_entry)
        if entry is None:
            yield Finding("error", path, "must be a mapping with model_name and litellm_params")
            continue
        yield from _unknown_keys(entry, path, _MODEL_ENTRY_KEYS, only_typos=True)
        model_name = entry.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            yield Finding("error", path, "missing model_name")
        params = _mapping(entry.get("litellm_params"))
        if params is None:
            yield Finding("error", path, "missing litellm_params mapping")
            continue
        yield from _unknown_keys(params, (*path, "litellm_params"), param_keys, only_typos=True)
        model = params.get("model")
        if not isinstance(model, str) or not model:
            yield Finding("error", (*path, "litellm_params"), "missing model")
            continue
        provider_model: Final = _provider_model(params, model)
        yield from _credential_findings(provider_model, params, (*path, "litellm_params"), environ)
        if entry.get("model_info") is None:
            model_info: Mapping[str, object] = {}
        elif (parsed_info := _mapping(entry["model_info"])) is None:
            yield Finding("error", (*path, "model_info"), "must be a mapping")
            continue
        else:
            model_info = parsed_info
        yield from _model_info_findings(
            provider_model, params, model_info, path, require_explicit_prices=require_explicit_prices
        )


def _model_names(config: Mapping[str, object]) -> frozenset[str]:
    router_settings: Final = _mapping(config.get("router_settings")) or {}
    listed: Final = (_mapping(entry) or {} for entry in _sequence(config.get("model_list")) or ())
    return frozenset(name for entry in listed if isinstance(name := entry.get("model_name"), str)) | frozenset(
        _mapping(router_settings.get("model_group_alias")) or {}
    )


def _is_known_model(name: str, known: frozenset[str]) -> bool:
    return name == "*" or name in known or any("*" in pattern and fnmatch.fnmatch(name, pattern) for pattern in known)


def _fallback_findings(section: Mapping[str, object], path: PathKey, known: frozenset[str]) -> Iterator[Finding]:
    for key in _FALLBACK_KEYS:
        if section.get(key) is None:
            continue
        rules = _sequence(section[key])
        if rules is None:
            yield Finding("error", (*path, key), "must be a list like [{model_group: [fallback_group]}]")
            continue
        for index, raw_rule in enumerate(rules):
            rule = _mapping(raw_rule)
            if rule is None:
                yield Finding("error", (*path, key, index), "must be a mapping like {model_group: [fallback_group]}")
                continue
            for source, targets in rule.items():
                listed = _sequence(targets)
                if listed is None:
                    yield Finding(
                        "error",
                        (*path, key, index),
                        f'fallbacks for "{source}" must be a list of model groups, like {{{source}: [{targets}]}}',
                    )
                    continue
                names = (source, *(target for target in listed if isinstance(target, str)))
                for name in names:
                    if known and not _is_known_model(name, known):
                        yield Finding("warning", (*path, key, index), f'model group "{name}" is not in model_list')
    defaults: Final = _sequence(section.get("default_fallbacks"))
    if defaults is None and section.get("default_fallbacks") is not None:
        yield Finding("error", (*path, "default_fallbacks"), "must be a list of model groups like [fallback_group]")
    for name in defaults or ():
        if known and isinstance(name, str) and not _is_known_model(name, known):
            yield Finding("warning", (*path, "default_fallbacks"), f'model group "{name}" is not in model_list')


def _section_findings(config: Mapping[str, object], known_models: frozenset[str]) -> Iterator[Finding]:
    from litellm.proxy._types import ConfigGeneralSettings
    from litellm.types.router import UpdateRouterConfig

    sections: Final = ("litellm_settings", "general_settings", "router_settings")
    parsed: Final = {name: _mapping(config[name]) for name in sections if config.get(name) is not None}
    yield from (Finding("error", (name,), "must be a mapping") for name, section in parsed.items() if section is None)

    litellm_settings: Final = parsed.get("litellm_settings")
    if litellm_settings is not None:
        candidates = _known_litellm_settings()
        for key in litellm_settings:
            if not _is_litellm_setting(key):
                yield Finding("warning", ("litellm_settings", key), f"unknown key{_suggestion(key, candidates)}")
        yield from _fallback_findings(litellm_settings, ("litellm_settings",), known_models)

    general_settings: Final = parsed.get("general_settings")
    if general_settings is not None:
        general_candidates = _known_general_settings()
        for key in general_settings:
            if key not in general_candidates and _is_module_setting(key):
                yield Finding(
                    "warning", ("general_settings", key), "not read from general_settings, move it to litellm_settings"
                )
            elif not _is_general_setting(key):
                yield Finding(
                    "warning", ("general_settings", key), f"unknown key{_suggestion(key, general_candidates)}"
                )
        yield from _type_findings(ConfigGeneralSettings, general_settings, ("general_settings",))

    router_settings: Final = parsed.get("router_settings")
    if router_settings is not None:
        yield from _unknown_keys(router_settings, ("router_settings",), _known_router_settings())
        yield from _type_findings(UpdateRouterConfig, router_settings, ("router_settings",))
        strategy = router_settings.get("routing_strategy")
        if isinstance(strategy, str) and strategy not in _known_routing_strategies():
            hint = _suggestion(strategy, _known_routing_strategies())
            yield Finding("error", ("router_settings", "routing_strategy"), f'unknown strategy "{strategy}"{hint}')
        yield from _fallback_findings(router_settings, ("router_settings",), known_models)


def lint_config(
    config: Mapping[str, object], environ: Mapping[str, str], *, require_explicit_prices: bool = False
) -> Iterator[Finding]:
    yield from _unknown_keys(config, (), _known_top_level_keys())
    if config.get("model_list") is not None:
        declared: Final = _mapping(config.get("environment_variables")) or {}
        yield from _model_list_findings(
            config["model_list"],
            _effective_environ(declared, environ),
            require_explicit_prices=require_explicit_prices,
        )
    yield from _section_findings(config, _model_names(config))
    yield from _env_findings(config, environ)


def _with_includes(config: Mapping[str, object], config_path: Path) -> Mapping[str, object]:
    from litellm.proxy.common_utils.config_includes import resolve_include_file_path, resolve_includes

    root: Final = str(config_path.resolve())

    def resolve(include_file: str, declared_in: str) -> str:
        return resolve_include_file_path(include_file, declared_in, root)

    async def read(file_path: str) -> Mapping[str, object]:
        return _MAPPING.validate_python(yaml.safe_load(Path(file_path).read_text(encoding="utf-8")))

    return asyncio.run(resolve_includes(config=config, location=root, resolve=resolve, read=read))


def _included_files(pending: tuple[tuple[str, str], ...], seen: frozenset[str], root: str) -> tuple[str, ...]:
    from litellm.proxy.common_utils.config_includes import include_entries, resolve_include_file_path

    if not pending:
        return ()
    entry, declared_in = pending[0]
    location: Final = resolve_include_file_path(entry, declared_in, root)
    if location in seen:
        return _included_files(pending[1:], seen, root)
    included: Final = _mapping(yaml.safe_load(Path(location).read_text(encoding="utf-8"))) or {}
    nested: Final = tuple((nested_entry, location) for nested_entry in include_entries(included))
    return (location, *_included_files((*pending[1:], *nested), seen | {location}, root))


def _included_duplicates(location: str, line: int | None) -> Iterator[Finding]:
    node: Final = yaml.compose(Path(location).read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    for event in _node_events(node, ()) if node is not None else ():
        if isinstance(event, Finding):
            yield Finding(
                "error",
                ("include",),
                f"{location}:{event.line}: duplicate key {_dotted(event.path)}, only the last value is used",
                line,
            )


def lint_file(
    config_path: Path, environ: Mapping[str, str], *, require_explicit_prices: bool = False
) -> tuple[Finding, ...]:
    try:
        text: Final = config_path.read_text(encoding="utf-8")
        root_node: Final = yaml.compose(text, Loader=yaml.SafeLoader)
        loaded: Final[object] = yaml.safe_load(text)
    except OSError as error:
        return (Finding("error", (), f"cannot read file: {error.strerror}"),)
    except yaml.MarkedYAMLError as error:
        mark: Final = error.problem_mark
        return (Finding("error", (), f"invalid YAML: {error.problem}", mark.line + 1 if mark else None),)
    config: Final = _mapping(loaded)
    if root_node is None or config is None:
        return (Finding("error", (), "config must be a YAML mapping"),)
    events: Final = tuple(_node_events(root_node, ()))
    lines: Final = MappingProxyType(dict(event for event in events if not isinstance(event, Finding)))
    duplicates: Final = tuple(event for event in events if isinstance(event, Finding))
    try:
        merged: Final = _with_includes(config, config_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        return (
            *duplicates,
            Finding("error", ("include",), f"cannot load included config: {error}", lines.get(("include",))),
        )
    root: Final = str(config_path.resolve())
    entries: Final = tuple((entry, root) for entry in _sequence(config.get("include")) or () if isinstance(entry, str))
    locations: Final = _included_files(entries, frozenset(), root)
    included_items: Final = itertools.chain.from_iterable(_included_items(location) for location in locations)
    replaced: Final = frozenset(
        key for key, value in included_items if not (isinstance(config.get(key), list) and isinstance(value, list))
    )
    located: Final = (
        _located(found, lines, replaced, "include" in config)
        for found in lint_config(merged, environ, require_explicit_prices=require_explicit_prices)
    )
    included_duplicates: Final = itertools.chain.from_iterable(
        _included_duplicates(location, lines.get(("include",))) for location in locations
    )
    return tuple(
        sorted((*duplicates, *included_duplicates, *located), key=lambda found: (found.line or 0, found.severity))
    )


def _included_items(location: str) -> tuple[tuple[str, object], ...]:
    return tuple((_mapping(yaml.safe_load(Path(location).read_text(encoding="utf-8"))) or {}).items())


def _located(found: Finding, lines: Mapping[PathKey, int], replaced: frozenset[str], has_includes: bool) -> Finding:
    if has_includes and found.path and (found.path not in lines or found.path[0] in replaced):
        return Finding(found.severity, found.path, f"{found.message} (in an included file)", lines.get(("include",)))
    return Finding(found.severity, found.path, found.message, _nearest_line(found.path, lines))


def _nearest_line(path: PathKey, lines: Mapping[PathKey, int]) -> int | None:
    return next((lines[path[:end]] for end in range(len(path), 0, -1) if path[:end] in lines), None)


def format_finding(config_path: Path, finding: Finding) -> str:
    location: Final = f"{config_path}:{finding.line}" if finding.line else str(config_path)
    return f"{location}: {finding.severity}: {_dotted(finding.path)}: {finding.message}"


def main(argv: Sequence[str] | None = None) -> int:
    parser: Final = argparse.ArgumentParser(description="Lint a LiteLLM proxy config.yaml without starting the proxy.")
    parser.add_argument("config", type=Path, help="Path to the config.yaml to check")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on warnings as well as errors")
    parser.add_argument(
        "--require-explicit-prices",
        action="store_true",
        help="Exit 1 when a model leaves out a price that LiteLLM's model cost map has for it",
    )
    args: Final = parser.parse_args(argv)
    config_path: Final[Path] = args.config
    strict: Final[bool] = args.strict
    require_explicit_prices: Final[bool] = args.require_explicit_prices
    findings: Final = lint_file(config_path, os.environ, require_explicit_prices=require_explicit_prices)
    errors: Final = sum(finding.severity == "error" for finding in findings)
    warnings: Final = len(findings) - errors
    report: Final = (
        (*(format_finding(config_path, finding) for finding in findings), f"{errors} error(s), {warnings} warning(s)")
        if findings
        else (f"{config_path}: no problems found",)
    )
    sys.stdout.write("\n".join(report) + "\n")
    return 1 if errors or (strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
