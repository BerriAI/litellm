import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm import verbose_logger
from litellm.litellm_core_utils.get_model_cost_map import MODEL_COST_WRITE_LOCK, GetModelCostMap
from litellm.rust_bridge._native import ModelRegistry

ModelCostRow: TypeAlias = dict[str, object]


@dataclass(frozen=True, slots=True)
class OverrideSet:
    entries: Mapping[str, Mapping[str, object]]
    replace_fields: frozenset[str]


@dataclass(frozen=True, slots=True)
class ModelCostOverridesRejected:
    owner: str
    problems: tuple[str, ...]


class _OverrideStore:
    owners: Mapping[str, OverrideSet] = MappingProxyType({})


def _apply_override(row: ModelCostRow | None, override: OverrideSet, key: str) -> ModelCostRow | None:
    entry: Final = override.entries.get(key)
    if entry is None:
        return row
    kept: Final = {field: value for field, value in (row or {}).items() if field not in override.replace_fields}
    return {**kept, **copy.deepcopy(dict(entry))}


def _compose(base_row: Mapping[str, object] | None, owners: Iterable[OverrideSet], key: str) -> ModelCostRow | None:
    initial: Final = dict(base_row) if base_row is not None else None
    return reduce(lambda row, override: _apply_override(row, override, key), owners, initial)


def _row_problems(model: str, row: Mapping[str, object]) -> tuple[str, ...]:
    try:
        registry: Final = ModelRegistry.parse(json.dumps({model: row}, allow_nan=False).encode())
    except (TypeError, ValueError) as error:
        return (f"{model}: {error}",)
    return tuple(registry.problems())


def _owned_keys(owners: Mapping[str, OverrideSet]) -> frozenset[str]:
    return frozenset(chain.from_iterable(override.entries for override in owners.values()))


def _composed_rows(owners: Mapping[str, OverrideSet], keys: Iterable[str]) -> Mapping[str, ModelCostRow | None]:
    base: Final = GetModelCostMap.loaded_model_cost_map()
    return MappingProxyType({key: _compose(base.get(key), owners.values(), key) for key in keys})


def _with_owner(owners: Mapping[str, OverrideSet], owner: str, override: OverrideSet) -> Mapping[str, OverrideSet]:
    if not override.entries:
        return MappingProxyType({name: kept for name, kept in owners.items() if name != owner})
    return MappingProxyType({**owners, owner: override})


def _served_map() -> dict[str, Mapping[str, object]]:
    import litellm

    return litellm.model_cost  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # an untyped dict


def _publish(rows: Mapping[str, ModelCostRow]) -> None:
    import litellm
    from litellm.utils import invalidate_model_cost_caches

    invalidate_model_cost_caches()
    litellm.add_known_models(  # pyright: ignore[reportUnknownMemberType]  # takes an untyped Dict
        model_cost_map=dict(rows)
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
    import litellm

    override: Final = OverrideSet(
        entries=MappingProxyType({key: MappingProxyType(copy.deepcopy(dict(entry))) for key, entry in entries.items()}),
        replace_fields=frozenset(replace_fields),
    )
    with MODEL_COST_WRITE_LOCK:
        current: Final = _OverrideStore.owners
        candidate: Final = _with_owner(current, owner, override)
        previous_keys: Final = frozenset(current[owner].entries) if owner in current else frozenset[str]()
        rows: Final = _composed_rows(candidate, previous_keys | frozenset(override.entries))
        owned: Final = _owned_keys(candidate)
        problems: Final = tuple(
            chain.from_iterable(
                _row_problems(key, row) for key, row in rows.items() if row is not None and key in owned
            )
        )
        if problems:
            return ModelCostOverridesRejected(owner=owner, problems=problems)
        _OverrideStore.owners = candidate
        merged: Final = {**_served_map(), **rows}
        litellm.model_cost = {key: row for key, row in merged.items() if row is not None}
        _publish({key: row for key, row in rows.items() if row is not None})
    return None


def replay_model_cost_overrides() -> None:
    """Compose every owner over the map just adopted. The caller holds ``MODEL_COST_WRITE_LOCK``"""
    owners: Final = _OverrideStore.owners
    if not owners:
        return
    rows: Final = _composed_rows(owners, _owned_keys(owners))
    problems: Final = MappingProxyType({key: _row_problems(key, row) for key, row in rows.items() if row is not None})
    for key, key_problems in problems.items():
        if key_problems:
            verbose_logger.warning(
                "LiteLLM: model cost overrides for %s no longer validate on the reloaded map and were skipped: %s",
                key,
                "; ".join(key_problems),
            )
    valid: Final = {key: row for key, row in rows.items() if row is not None and not problems[key]}
    _served_map().update(valid)
    _publish(valid)
