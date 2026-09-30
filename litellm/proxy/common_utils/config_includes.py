import os
from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import Final, Protocol

from litellm._logging import verbose_proxy_logger

INCLUDE_KEY: Final = "include"


def resolve_include_file_path(include_file: str, declared_in: str, root_config_path: str) -> str:
    """
    Resolve one `include` entry to the file it names, next to the config that declares it.

    A config written before nested entries resolved this way can name a file sitting next to the root
    config instead, so that file is still read, with a warning naming where it was found. When both
    files exist the one next to the declaring config wins and the other is named in a warning.
    """
    declared_relative: Final = os.path.abspath(os.path.join(os.path.dirname(declared_in), include_file))
    root_relative: Final = os.path.abspath(os.path.join(os.path.dirname(root_config_path), include_file))
    if root_relative == declared_relative or not os.path.exists(root_relative):
        return declared_relative

    if not os.path.exists(declared_relative):
        verbose_proxy_logger.warning(
            "Config include '%s' declared in %s was not found next to it, so %s was read instead. "
            "Move the included file next to the config that declares it.",
            include_file,
            declared_in,
            root_relative,
        )
        return root_relative

    verbose_proxy_logger.warning(
        "Config include '%s' declared in %s matches two files. %s sits next to that config and was read, "
        "so %s was skipped. Rename one of the two to say which one you meant.",
        include_file,
        declared_in,
        declared_relative,
        root_relative,
    )
    return declared_relative


class IncludeResolver(Protocol):
    def __call__(self, include_entry: str, declared_in: str, /) -> str: ...


class ConfigReader(Protocol):
    def __call__(self, location: str, /) -> Awaitable[Mapping[str, object]]: ...


def _merged_value(base_value: object, included_value: object) -> object:
    if isinstance(included_value, list) and isinstance(base_value, list):
        return [*base_value, *included_value]  # mutable-ok: a merged config value stays the plain list the proxy loads
    return included_value


def _merged_entry(base: Mapping[str, object], included: Mapping[str, object], key: str) -> object:
    if key not in included:
        return base[key]
    return _merged_value(base.get(key), included[key])


def _merged(base: Mapping[str, object], included: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _merged_entry(base, included, key) for key in (*base, *included)})


def _without_include(config: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: value for key, value in config.items() if key != INCLUDE_KEY})


def include_entries(config: Mapping[str, object]) -> tuple[str, ...]:
    if INCLUDE_KEY not in config:
        return ()

    entries: Final = config[INCLUDE_KEY]
    if not isinstance(entries, list):
        raise ValueError("'include' must be a list of file paths")

    paths: Final = tuple(entry for entry in entries if isinstance(entry, str))
    if len(paths) != len(entries):
        raise ValueError("'include' must be a list of file paths")

    return paths


def _pending_from(config: Mapping[str, object], location: str) -> tuple[tuple[str, str], ...]:
    return tuple((entry, location) for entry in include_entries(config))


async def _resolve(
    config: Mapping[str, object],
    pending: tuple[tuple[str, str], ...],
    loaded: frozenset[str],
    resolve: IncludeResolver,
    read: ConfigReader,
) -> Mapping[str, object]:
    if not pending:
        return _without_include(config)

    entry, declared_in = pending[0]
    location: Final = resolve(entry, declared_in)
    if location in loaded:
        return await _resolve(config, pending[1:], loaded, resolve, read)

    included: Final = await read(location)
    return await _resolve(
        _merged(config, _without_include(included)),
        (*pending[1:], *_pending_from(included, location)),
        loaded | frozenset((location,)),
        resolve,
        read,
    )


async def resolve_includes(
    config: Mapping[str, object],
    location: str,
    resolve: IncludeResolver,
    read: ConfigReader,
) -> dict[str, object]:
    """
    Merge every config named by the `include` directive into the config that declares it.

    List values are extended and every other value is overridden, `resolve` turns each entry into the
    location it names relative to the config that declares it, a config already pulled in is neither
    read nor merged a second time, and `read` decides where a location is read from, so the same merge
    applies to configs on disk and to configs hosted in a bucket.
    """
    merged: Final = await _resolve(config, _pending_from(config, location), frozenset((location,)), resolve, read)
    return dict(merged)  # mutable-ok: the proxy mutates the config it loads
