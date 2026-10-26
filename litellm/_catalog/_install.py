from __future__ import annotations

import copy
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import groupby
from typing import Final, cast  # noqa: TID251  # live dict identity cannot survive validation into a new dict
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._catalog._load import LoadedCatalog, served_rows
from litellm._logging import verbose_logger
from litellm.rust_bridge._native import ModelCostLayers, ModelRegistry, write_rows

LOCK: Final = threading.RLock()
LAYERS: Final = ModelCostLayers()
_ROWS: Final = TypeAdapter(dict[str, dict[str, object]])
_ROW: Final = TypeAdapter(dict[str, object])
_JSON: Final = TypeAdapter[JsonValue](JsonValue)
_OPAQUE_KEY: Final = f"_litellm_python_value_{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class Provenance:
    source: str
    url: str | None = None
    is_env_forced: bool = False
    fallback_reason: str | None = None
    revision: str | None = None
    etag: str | None = None


@dataclass(slots=True)
class _Installed:
    rows: dict[str, dict[str, object]] = field(default_factory=dict)  # mutable-ok: deep snapshot of installed rows
    opaque: dict[str, object] = field(default_factory=dict)  # mutable-ok: opaque values referenced by native layers
    generation: int = 0
    registry: ModelRegistry | None = None


STATE: Final = _Installed()


def _encoded_value(value: object) -> JsonValue:
    try:
        return _JSON.validate_python(value)
    except ValidationError:
        for token, previous in STATE.opaque.items():
            if previous == value:
                return {_OPAQUE_KEY: token}
        new_token: Final = str(len(STATE.opaque))
        STATE.opaque[new_token] = copy.deepcopy(value)
        return {_OPAQUE_KEY: new_token}


def _encoded_row(row: Mapping[str, object]) -> dict[str, JsonValue]:  # mutable-ok: native layer payload
    return {name: _encoded_value(value) for name, value in row.items()}


def _decoded_value(value: JsonValue) -> object:
    if isinstance(value, dict) and len(value) == 1:
        token: Final = value.get(_OPAQUE_KEY)
        if isinstance(token, str) and token in STATE.opaque:
            return copy.deepcopy(STATE.opaque[token])
    return value


def _live_rows() -> dict[str, object]:  # mutable-ok: retain the public dict and any subclass identity
    import litellm

    return cast("dict[str, object]", litellm.model_cost)  # cast-ok: retain live dict; validate rows before composition


def record_pending_writes(keys: frozenset[str] | None = None) -> None:
    target: Final = _live_rows()
    observed: Final = (
        dict[str, object].copy(target) if keys is None else {key: target[key] for key in keys if key in target}
    )
    live: Final = _ROWS.validate_python(observed)
    candidates: Final = frozenset(STATE.rows) | frozenset(live) if keys is None else keys
    changed: Final = {
        key: (
            _encoded_row(STATE.rows[key]) if key in STATE.rows else None,
            _encoded_row(live[key]) if key in live else None,
        )
        for key in candidates
        if STATE.rows.get(key) != live.get(key) or (key in STATE.rows) != (key in live)
    }
    LAYERS.diff(changed)


def remember_written_rows(keys: frozenset[str]) -> None:
    for key, row in ((key, dict[str, object].get(_live_rows(), key)) for key in keys):
        if row is not None:
            STATE.rows[key] = copy.deepcopy(_ROW.validate_python(row))
            continue
        STATE.rows.pop(key, None)


def registration_write(
    key: str,
    base: Mapping[str, object],
    fields: Mapping[str, object],
    *,
    persist: bool,
    inherits_from: str | None,
    generation: int,
    add_to_provider_model_sets: bool = True,
) -> bool:
    from litellm._catalog import _read
    from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry

    with LOCK:
        target: Final = _live_rows()
        current: Final = dict[str, object].get(target, key)
        names: Final = current.names if isinstance(current, AliasedCostEntry) else ()
        keys: Final = frozenset((key, *(name for name in names if target.get(name) is current)))
        record_pending_writes(keys)
        encoded: Final = _encoded_row(fields)
        local: Final = ModelCostLayers()
        local.diff({key: ({}, None)})
        local.diff({key: (None, _encoded_row(base))})
        local.record_registration(key, None, encoded)
        registry: Final = STATE.registry or ModelRegistry.bundled()
        composed: Final = local.compose_keys(registry, frozenset({key}))
        entry: Final = composed[0][1]
        if entry is None:
            return True
        previous_fields: Final[frozenset[str]] = (
            current.registered_fields if isinstance(current, AliasedCostEntry) else frozenset()
        )
        published: Final = AliasedCostEntry(
            {name: _decoded_value(value) for name, value in entry.items()},
            tuple(keys),
            previous_fields | frozenset(fields),
            _read.registration_provider_overrides(key, current, add_to_provider_model_sets),
        )
        snapshot: Final = copy.deepcopy(dict(published))
        if generation != STATE.generation:
            return False
        if persist:
            for alias in keys:
                LAYERS.record_registration(alias, inherits_from, encoded)
        write_rows(target, dict.fromkeys(keys, published), [])
        for alias in keys:
            STATE.rows[alias] = snapshot
        return True


def accepted_catalog(catalog: LoadedCatalog) -> LoadedCatalog | None:
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    registry, errors = catalog.registry.without_rejected()
    for error in errors:
        verbose_logger.warning("LiteLLM: model cost map row omitted: %s.%s: %s", error.key, error.field, error.message)
    size_error: Final = registry.check_size(GetModelCostMap.get_backup_model_count())
    if size_error is not None:
        verbose_logger.warning("LiteLLM: model cost map failed integrity validation: %s", size_error.message)
        return None
    return LoadedCatalog(registry)


def _composed_rows(registry: ModelRegistry) -> dict[str, object]:  # mutable-ok: atomic publication payload
    from litellm._catalog import _read
    from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry

    composed: Final = LAYERS.compose(registry)
    groups: Final = groupby(sorted(composed, key=lambda row: id(row[1])), key=lambda row: id(row[1]))
    grouped: Final = {identity: tuple(rows) for identity, rows in groups}
    entries: Final = {
        identity: AliasedCostEntry(
            {name: _decoded_value(value) for name, value in rows[0][1].items()},
            tuple(row[0] for row in rows),
            rows[0][2],
            _read.provider_overrides_for_keys(tuple(row[0] for row in rows)),
        )
        for identity, rows in grouped.items()
    }
    return {key: entries[id(fields)] for key, fields, _ in composed}


def install(catalog: LoadedCatalog, *, provenance: Provenance) -> int:
    import litellm
    from litellm._catalog import _read
    from litellm.litellm_core_utils import get_model_cost_map as loader
    from litellm.litellm_core_utils.fallback_generalizations import (
        set_fallback_generalizations,  # pyright: ignore[reportUnknownVariableType]  # legacy rule installer accepts the validated registry payload
    )

    with LOCK:
        accepted: Final = accepted_catalog(catalog) if provenance.source == "remote" else catalog
        if accepted is None:
            return 0
        record_pending_writes()
        rows: Final = _composed_rows(accepted.registry)
        target: Final = _live_rows()
        changed: Final = {
            key: row
            for key, row in rows.items()
            if dict[str, object].get(target, key) != row
            or _read.provider_overrides(target.get(key)) != _read.provider_overrides(row)
        }
        removed: Final = [key for key in dict[str, object].keys(target) if key not in rows]
        write_rows(target, changed, removed)
        STATE.rows = copy.deepcopy(_ROWS.validate_python(dict[str, object].copy(target)))
        STATE.registry = accepted.registry
        STATE.generation += 1
        set_fallback_generalizations(accepted.registry.fallback_rules())
        loader.GetModelCostMap._snapshot_loaded_catalog(served_rows(accepted))  # pyright: ignore[reportPrivateUsage]  # install owns the loaded snapshot
        info: Final = loader._cost_map_source_info  # pyright: ignore[reportPrivateUsage]  # one writer of legacy provenance
        info.source = provenance.source
        info.url = provenance.url
        info.is_env_forced = provenance.is_env_forced
        info.fallback_reason = provenance.fallback_reason
        info.source_revision = provenance.revision
        info.etag = provenance.etag
        info.loaded_at = datetime.now(timezone.utc)
        if loader._litellm_import_complete.is_set():  # pyright: ignore[reportPrivateUsage]  # import initializes provider sets later
            from litellm import utils

            utils._invalidate_model_cost_lowercase_map()  # pyright: ignore[reportPrivateUsage]  # publication invalidates derived indexes
            litellm.add_known_models(model_cost_map=target)  # pyright: ignore[reportUnknownMemberType]  # legacy index builder accepts the live public dict
            for replay in utils._LiveDeploymentReplay.callbacks:  # pyright: ignore[reportPrivateUsage]  # router owns deployment replay
                replay()
        return len(served_rows(accepted))
