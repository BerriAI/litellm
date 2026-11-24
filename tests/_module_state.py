import copy
import inspect
import itertools
import warnings
from collections.abc import Mapping
from types import ModuleType
from typing import Final

_SNAPSHOT_TYPES: Final = (list, dict, set, tuple, str, int, float, bool, bytes)
_MUTABLE_CONTAINERS: Final = (list, dict, set, bytearray)
_ABSENT: Final = object()


def _holds_mutable_container(value: object) -> bool:
    if isinstance(value, _MUTABLE_CONTAINERS):
        return True
    if isinstance(value, tuple):
        return any(_holds_mutable_container(element) for element in value)
    return False


def _restored_value(value: object) -> object:
    return copy.deepcopy(value) if _holds_mutable_container(value) else value


def _readable_attributes(module: ModuleType) -> tuple[tuple[str, object], ...]:
    return tuple((attr, value) for attr, value in tuple(vars(module).items()) if not attr.startswith("_"))


def _snapshot_of(module: ModuleType, attr: str, value: object) -> object:
    try:
        return _restored_value(value)
    except Exception as exc:
        warnings.warn(f"conftest: could not snapshot {module.__name__}.{attr}: {exc}", stacklevel=2)
        return _ABSENT


def snapshot_mutable_state(module: ModuleType) -> Mapping[str, object]:
    candidates: Final = (
        (attr, _snapshot_of(module, attr, value))
        for attr, value in _readable_attributes(module)
        if value is None or isinstance(value, _SNAPSHOT_TYPES)
    )
    return {attr: value for attr, value in candidates if value is not _ABSENT}


def _still_matches(current: object, default: object) -> bool:
    if current is default:
        return True
    if type(current) is not type(default):
        return False
    try:
        return bool(current == default)
    except Exception:
        return False


def _restored_entries(current: dict[object, object], default: dict[object, object]) -> dict[object, object]:
    return {
        key: current[key]
        if key in current and type(current[key]) is type(value) and current[key] == value
        else _restored_value(value)
        for key, value in default.items()
    }


def _restored_from(current: object, default: object) -> object:
    if type(current) is dict and type(default) is dict:
        return _restored_entries(current, default)
    return _restored_value(default)


def restore_mutable_state(module: ModuleType, snapshot: Mapping[str, object]) -> None:
    live: Final = vars(module)
    for attr, default in snapshot.items():
        current = live.get(attr, _ABSENT)
        if _still_matches(current, default):
            continue
        try:
            setattr(module, attr, _restored_from(current, default))
        except Exception as exc:
            warnings.warn(f"conftest: could not restore {module.__name__}.{attr}: {exc}", stacklevel=2)


def _is_flushable(value: object) -> bool:
    if inspect.isclass(value) or inspect.ismodule(value):
        return False
    return callable(getattr(value, "flush_cache", None))


def _flushable_attributes(module: ModuleType) -> tuple[tuple[ModuleType, str], ...]:
    return tuple((module, attr) for attr, value in _readable_attributes(module) if _is_flushable(value))


def collect_flushable_caches(modules: tuple[ModuleType, ...]) -> tuple[tuple[ModuleType, str], ...]:
    return tuple(itertools.chain.from_iterable(_flushable_attributes(module) for module in modules))


def flush_caches(targets: tuple[tuple[ModuleType, str], ...]) -> None:
    for module, attr in targets:
        flush = getattr(vars(module).get(attr), "flush_cache", None)
        if not callable(flush):
            continue
        try:
            flush()
        except Exception as exc:
            warnings.warn(f"conftest: flush_cache failed on {module.__name__}.{attr}: {exc}", stacklevel=2)
