from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import Final, Protocol

INCLUDE_KEY: Final = "include"


class ConfigLoader(Protocol):
    def __call__(self, include_entry: str, /) -> Awaitable[Mapping[str, object]]: ...


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


async def _resolve(config: Mapping[str, object], pending: tuple[str, ...], load: ConfigLoader) -> Mapping[str, object]:
    if not pending:
        return _without_include(config)

    included: Final = await load(pending[0])
    return await _resolve(
        _merged(config, _without_include(included)),
        (*pending[1:], *include_entries(included)),
        load,
    )


async def resolve_includes(config: Mapping[str, object], load: ConfigLoader) -> dict[str, object]:
    """
    Merge every config named by the `include` directive into the config that declares it.

    List values are extended and every other value is overridden, an included config may declare
    further includes, and `load` decides where an entry is read from, so the same merge applies to
    configs on disk and to configs hosted in a bucket.
    """
    merged: Final = await _resolve(config, include_entries(config), load)
    return dict(merged)  # mutable-ok: the proxy mutates the config it loads
