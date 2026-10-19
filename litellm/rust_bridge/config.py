from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol, TextIO, runtime_checkable

import yaml
from pydantic import TypeAdapter, ValidationError

from litellm.rust_bridge.bindings import NativeBinding


class IncludePlan(Protocol):
    def next_entry(self) -> tuple[str, str] | None: ...
    def mark_loaded(self, location: str) -> bool: ...
    def merge(
        self, config: Mapping[str, object], included: Mapping[str, object], location: str
    ) -> Mapping[str, object]: ...


@runtime_checkable
class IncludePlanFactory(Protocol):
    def __call__(self, config: Mapping[str, object], location: str) -> IncludePlan: ...


@runtime_checkable
class YamlParser(Protocol):
    def __call__(self, source: str, /) -> tuple[bool, object]: ...


@runtime_checkable
class IncludePathResolver(Protocol):
    def __call__(self, entry: str, declared_in: str, root: str, /) -> tuple[str, str | None]: ...


def _parser(value: object) -> YamlParser | None:
    return value if isinstance(value, YamlParser) else None


def _resolver(value: object) -> IncludePathResolver | None:
    return value if isinstance(value, IncludePathResolver) else None


def _plan_factory(value: object) -> IncludePlanFactory | None:
    return value if isinstance(value, IncludePlanFactory) else None


PARSE_YAML: Final = NativeBinding("_parse_config_yaml", validate=_parser)
INCLUDE_PATH: Final = NativeBinding("_resolve_config_include_path", validate=_resolver)
INCLUDE_PLAN: Final = NativeBinding("_ConfigIncludes", validate=_plan_factory)
_MAPPING: Final = TypeAdapter(Mapping[str, object])


def parse_yaml(source: str | TextIO) -> object:
    native: Final = PARSE_YAML.load()
    if native is None:
        return yaml.safe_load(source)
    position: Final = None if isinstance(source, str) else source.tell()
    contents: Final = source if isinstance(source, str) else source.read()
    supported, parsed = native(contents)
    if supported:
        return parsed
    if not isinstance(source, str) and position is not None:
        source.seek(position)
    return yaml.safe_load(source)


def include_plan(config: object, location: str) -> IncludePlan | None:
    native: Final = INCLUDE_PLAN.load()
    if native is None:
        return None
    try:
        mapping: Final = _MAPPING.validate_python(config)
    except ValidationError:
        return None
    return native(dict(mapping), location)


def include_path(entry: str, declared_in: str, root: str) -> tuple[str, str | None] | None:
    native: Final = INCLUDE_PATH.load()
    if native is None:
        return None
    return native(entry, declared_in, root)
