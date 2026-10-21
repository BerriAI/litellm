import copy
import warnings
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from types import MappingProxyType, ModuleType
from typing import Final, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm import verbose_logger
from litellm.litellm_core_utils.get_model_cost_map import (
    MODEL_COST_WRITE_LOCK,
    AliasedCostEntry,
    GetModelCostMap,
    model_cost_row_problems,
)

ModelCostRow: TypeAlias = dict[str, object]
ServedRow: TypeAlias = Mapping[str, object]
_REBOUND_ROWS: Final = TypeAdapter(dict[str, ServedRow])

REGISTER_MODEL_OWNER: Final = "register_model"
_DIRECT_WRITE_ADVICE: Final = (
    "bypasses validation and is lost on the next cost map reload. Use "
    "litellm.set_model_cost_overrides(owner=..., entries=...) for prices you own, or litellm.register_model(...)"
)
_REBIND_ADVICE: Final = (
    "litellm.model_cost = ... bypasses validation. The bound dict becomes the base that overrides compose over until "
    "the next reload, including any overrides it already carries. Use litellm.reload_model_cost_map() to refresh the "
    "map, or litellm.set_model_cost_overrides(owner=..., entries=...) for prices you own"
)


class ModelCostDirectWriteWarning(DeprecationWarning):
    pass


def _direct_write(operation: str) -> None:
    from litellm.utils import invalidate_model_cost_caches

    warnings.warn(f"litellm.model_cost{operation} {_DIRECT_WRITE_ADVICE}", ModelCostDirectWriteWarning, stacklevel=3)
    invalidate_model_cost_caches()


class ModelCostDict(dict[str, ServedRow]):
    """The served ``litellm.model_cost``. Writes through its own methods warn, so litellm writes through ``dict``'s"""

    __slots__ = ()

    def __setitem__(self, key: str, value: ServedRow) -> None:
        super().__setitem__(key, value)
        _direct_write("[key] = value")

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        _direct_write("[key] deletion")

    def update(self, *args: object, **kwargs: ServedRow) -> None:  # kwargs-ok: dict.update's keyword form
        super().update(*args, **kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]  # dict's own overloads
        _direct_write(".update(...)")

    def setdefault(self, key: str, default: ServedRow) -> ServedRow:  # pyright: ignore[reportIncompatibleMethodOverride]  # rows are never None
        result: Final = super().setdefault(key, default)
        _direct_write(".setdefault(...)")
        return result

    def pop(self, key: str, *default: ServedRow | None) -> ServedRow | None:  # pyright: ignore[reportIncompatibleMethodOverride]  # dict.pop's optional default
        result: Final = super().pop(key, *default)
        _direct_write(".pop(...)")
        return result

    def popitem(self) -> tuple[str, ServedRow]:
        result: Final = super().popitem()
        _direct_write(".popitem()")
        return result

    def clear(self) -> None:
        super().clear()
        _direct_write(".clear()")

    def __ior__(self, other: Mapping[str, ServedRow]) -> "ModelCostDict":  # pyright: ignore[reportIncompatibleMethodOverride]  # rows only, not pairs
        super().update(other)
        _direct_write(" |= ...")
        return self

    def __reduce__(self) -> tuple[object, ...]:
        return dict, (dict(self),)


@dataclass(frozen=True, slots=True)
class OverrideSet:
    entries: Mapping[str, Mapping[str, object]]
    replace_fields: frozenset[str]


@dataclass(frozen=True, slots=True)
class RegisteredRow:
    """What ``register_model`` adds on top of the base row served under each of ``names``, and the fields it set"""

    names: tuple[str, ...]
    entry: Mapping[str, object]
    fields: frozenset[str]


@dataclass(frozen=True, slots=True)
class ModelCostOverridesRejected:
    owner: str
    problems: tuple[str, ...]


class ModelCostRegistrationError(ValueError):
    def __init__(self, problems: tuple[str, ...]) -> None:
        super().__init__("register_model rejected the call and registered nothing: " + "; ".join(problems))
        self.problems: Final = problems


class _OverrideStore:
    owners: Mapping[str, OverrideSet] = MappingProxyType({})
    registrations: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    registered: Mapping[str, RegisteredRow] = MappingProxyType({})
    rebound_base: Mapping[str, ServedRow] | None = None


def model_cost_base() -> Mapping[str, ServedRow]:
    """The rows overrides compose over: the dict last bound to ``litellm.model_cost`` since the last load, else it"""
    rebound: Final = _OverrideStore.rebound_base
    return rebound if rebound is not None else GetModelCostMap.loaded_model_cost_map()


def model_cost_registrations() -> Mapping[str, Mapping[str, object]]:
    return _OverrideStore.registrations


def registered_model_cost_rows() -> Mapping[str, RegisteredRow]:
    return _OverrideStore.registered


def _rebound_rows(value: object) -> Mapping[str, ServedRow] | None:
    try:
        return MappingProxyType(_REBOUND_ROWS.validate_python(value))
    except ValidationError:
        return None


def rebind_model_cost(value: object) -> None:
    """``litellm.model_cost = value`` from outside litellm"""
    import litellm
    from litellm.utils import invalidate_model_cost_caches

    warnings.warn(_REBIND_ADVICE, ModelCostDirectWriteWarning, stacklevel=3)
    rows: Final = _rebound_rows(value)
    with MODEL_COST_WRITE_LOCK:
        ModuleType.__setattr__(litellm, "model_cost", value)
        if rows is None:
            invalidate_model_cost_caches()
            return
        _OverrideStore.rebound_base = rows
        _publish(rows)


def write_model_cost_rows(
    cost_map: dict[str, ServedRow],  # mutable-ok: writes the served dict
    rows: Mapping[str, ServedRow],
) -> None:
    """Write ``rows`` into ``cost_map`` without the direct write warning"""
    if isinstance(cost_map, ModelCostDict):
        super(ModelCostDict, cost_map).update(rows)
        return
    cost_map.update(rows)


def install_model_cost(rows: Mapping[str, ServedRow]) -> None:
    """Bind ``litellm.model_cost`` without the rebind warning. The caller invalidates caches"""
    import litellm

    ModuleType.__setattr__(litellm, "model_cost", ModelCostDict(rows))


def adopt_loaded_model_cost(rows: Mapping[str, ServedRow]) -> None:
    """Serve a freshly loaded map as the new base. The caller holds ``MODEL_COST_WRITE_LOCK``"""
    _OverrideStore.rebound_base = None
    install_model_cost(rows)


def _apply_override(row: ModelCostRow | None, override: OverrideSet, key: str) -> ModelCostRow | None:
    entry: Final = override.entries.get(key)
    if entry is None:
        return row
    kept: Final = {field: value for field, value in (row or {}).items() if field not in override.replace_fields}
    return {**kept, **copy.deepcopy(dict(entry))}


def _compose(base_row: Mapping[str, object] | None, owners: Iterable[OverrideSet], key: str) -> ModelCostRow | None:
    initial: Final = dict(base_row) if base_row is not None else None
    return reduce(lambda row, override: _apply_override(row, override, key), owners, initial)


def _owned_keys(owners: Mapping[str, OverrideSet]) -> frozenset[str]:
    return frozenset(chain.from_iterable(override.entries for override in owners.values()))


def _composed_rows(owners: Mapping[str, OverrideSet], keys: Iterable[str]) -> Mapping[str, ModelCostRow | None]:
    base: Final = model_cost_base()
    return MappingProxyType({key: _compose(base.get(key), owners.values(), key) for key in keys})


def _with_owner(owners: Mapping[str, OverrideSet], owner: str, override: OverrideSet) -> Mapping[str, OverrideSet]:
    if not override.entries:
        return MappingProxyType({name: kept for name, kept in owners.items() if name != owner})
    return MappingProxyType({**owners, owner: override})


def _with_registrations(
    owners: Mapping[str, OverrideSet], registered: Mapping[str, RegisteredRow]
) -> Mapping[str, OverrideSet]:
    entries: Final = MappingProxyType({name: row.entry for name, row in registered.items()})
    return _with_owner(owners, REGISTER_MODEL_OWNER, OverrideSet(entries=entries, replace_fields=frozenset()))


def _served_map() -> dict[str, ServedRow]:  # mutable-ok: the served dict that owners write into
    import litellm

    return litellm.model_cost  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # an untyped dict


def _served_entry(key: str, row: ModelCostRow, current: ServedRow | None) -> ModelCostRow:
    names: Final = current.names if isinstance(current, AliasedCostEntry) else ()
    previous_fields: Final = current.registered_fields if isinstance(current, AliasedCostEntry) else frozenset[str]()
    registered: Final = _OverrideStore.registered.get(key)
    fields: Final = previous_fields | (registered.fields if registered is not None else frozenset[str]())
    if not names and not fields:
        return row
    return AliasedCostEntry(row, names, fields)


def _shared_with_aliases(key: str, entries: Mapping[str, ModelCostRow]) -> ModelCostRow:
    entry: Final = entries[key]
    names: Final = entry.names if isinstance(entry, AliasedCostEntry) else ()
    return next((entries[name] for name in names if name in entries and entries[name] == entry), entry)


def _served_entries(rows: Mapping[str, ModelCostRow]) -> Mapping[str, ModelCostRow]:
    """Rows as served: an alias group whose composed rows agree shares one entry, as a freshly loaded map does"""
    served: Final = _served_map()
    entries: Final = MappingProxyType({key: _served_entry(key, row, served.get(key)) for key, row in rows.items()})
    return MappingProxyType({key: _shared_with_aliases(key, entries) for key in entries})


def _publish(rows: Mapping[str, ServedRow]) -> None:
    import litellm
    from litellm.utils import invalidate_model_cost_caches

    invalidate_model_cost_caches()
    litellm.add_known_models(  # pyright: ignore[reportUnknownMemberType]  # takes an untyped Dict
        model_cost_map=dict(rows)
    )


def _owned_row_problems(rows: Mapping[str, ModelCostRow | None], owned: frozenset[str]) -> Iterator[str]:
    return chain.from_iterable(
        model_cost_row_problems(key, row) for key, row in rows.items() if row is not None and key in owned
    )


def set_model_cost_overrides(
    owner: str,
    entries: Mapping[str, Mapping[str, object]],
    *,
    replace_fields: frozenset[str] = frozenset(),
) -> ModelCostOverridesRejected | None:
    """Replace everything ``owner`` has set with ``entries``, composed over the last loaded cost map and replayed on
    every reload. ``entries={}`` withdraws the owner. A field in ``replace_fields`` that an entry leaves out is removed
    from that row instead of kept from the loaded map. The call is rejected as a whole when any composed row is
    invalid, and nothing changes."""
    override: Final = OverrideSet(
        entries=MappingProxyType({key: MappingProxyType(copy.deepcopy(dict(entry))) for key, entry in entries.items()}),
        replace_fields=frozenset(replace_fields),
    )
    with MODEL_COST_WRITE_LOCK:
        current: Final = _OverrideStore.owners
        candidate: Final = _with_owner(current, owner, override)
        previous_keys: Final = frozenset(current[owner].entries) if owner in current else frozenset[str]()
        rows: Final = _composed_rows(candidate, previous_keys | frozenset(override.entries))
        problems: Final = tuple(_owned_row_problems(rows, _owned_keys(candidate)))
        if problems:
            return ModelCostOverridesRejected(owner=owner, problems=problems)
        _OverrideStore.owners = candidate
        served: Final = {key: row for key, row in rows.items() if row is not None}
        merged: Final = {**_served_map(), **_served_entries(served)}
        install_model_cost({key: row for key, row in merged.items() if key in served or key not in rows})
        _publish(served)
    return None


def commit_model_cost_registrations(
    registrations: Mapping[str, Mapping[str, object]],
    registered: Mapping[str, RegisteredRow],
    keys: frozenset[str],
    touched: frozenset[str],
) -> ModelCostOverridesRejected | None:
    """Serve ``register_model``'s rows for ``touched`` (``keys`` plus the aliases sharing their rows) once each of
    ``keys`` composes into a valid row, and record ``registrations`` for replay. Nothing changes when one is invalid.
    The caller holds ``MODEL_COST_WRITE_LOCK``"""
    from litellm.utils import invalidate_model_cost_caches

    candidate: Final = _with_registrations(_OverrideStore.owners, registered)
    rows: Final = _composed_rows(candidate, touched)
    problems: Final = tuple(_owned_row_problems(rows, keys))
    if problems:
        return ModelCostOverridesRejected(owner=REGISTER_MODEL_OWNER, problems=problems)
    _OverrideStore.owners = candidate
    _OverrideStore.registrations = MappingProxyType(dict(registrations))
    _OverrideStore.registered = registered
    write_model_cost_rows(_served_map(), _served_entries({key: row for key, row in rows.items() if row is not None}))
    invalidate_model_cost_caches()
    return None


def _registered_on_current_map(registrations: Mapping[str, Mapping[str, object]]) -> Mapping[str, RegisteredRow]:
    from litellm.utils import model_cost_registration_rows

    if not registrations:
        return MappingProxyType({})
    return model_cost_registration_rows(registrations, MappingProxyType({})).registered


def replay_model_cost_overrides() -> None:
    """Compose ``register_model``'s recordings, resolved again against the map just adopted, and every other owner
    over it. A key whose composed row no longer validates serves the loaded row and stays recorded"""
    with MODEL_COST_WRITE_LOCK:
        registered: Final = _registered_on_current_map(_OverrideStore.registrations)
        owners: Final = _with_registrations(_OverrideStore.owners, registered)
        _OverrideStore.registered = registered
        _OverrideStore.owners = owners
        if not owners:
            return
        rows: Final = _composed_rows(owners, _owned_keys(owners))
        problems: Final = MappingProxyType(
            {key: model_cost_row_problems(key, row) for key, row in rows.items() if row is not None}
        )
        for key, key_problems in problems.items():
            if key_problems:
                verbose_logger.warning(
                    "LiteLLM: model cost overrides for %s no longer validate on the reloaded map and were skipped: %s",
                    key,
                    "; ".join(key_problems),
                )
        valid: Final = {key: row for key, row in rows.items() if row is not None and not problems[key]}
        write_model_cost_rows(_served_map(), _served_entries(valid))
        _publish(valid)
