from __future__ import annotations

import asyncio
import functools
import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypeAlias,
    cast,  # noqa: TID251  # each callsite has a justified cast-ok check
)

from litellm import verbose_logger
from litellm.litellm_core_utils.get_model_cost_map import (
    _MODEL_COST_LAYERS,  # pyright: ignore[reportPrivateUsage]  # native layer composition is the API storage seam
    _MODEL_COST_REGISTRATION_LOCK,  # pyright: ignore[reportPrivateUsage]  # owner updates share the native lock
    MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    GetModelCostMap,
    LoadedCatalog,
    NativeSource,
    _adopt_installed_model_cost_map,  # pyright: ignore[reportPrivateUsage]  # reload publishes through the existing adopter
    _model_cost_dict_from_native_rows,  # pyright: ignore[reportPrivateUsage]  # served aliases come from native rows
    _native_revalidation,  # pyright: ignore[reportPrivateUsage]  # native owns retry and integrity policy
    _source_info,  # pyright: ignore[reportPrivateUsage]  # reload records source provenance through the existing seam
    get_model_cost_map_source_info,
    git_blob_id,
    install_loaded_catalog,
)
from litellm.litellm_core_utils.get_model_cost_map import (
    ModelCostMapUnavailable as NativeUnavailable,
)
from litellm.litellm_core_utils.model_cost_dict import ModelCostLayers
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.rust_bridge._native import ModelCostLayers as NativeModelCostLayers
from litellm.rust_bridge._native import ModelRegistry, RowError, model_row_errors

if TYPE_CHECKING:
    from litellm.rust_bridge._native import RowErrorKind
else:
    RowErrorKind: TypeAlias = str

_FrozenRow: TypeAlias = Mapping[str, object]
_FrozenEntries: TypeAlias = Mapping[str, _FrozenRow]


@dataclass(frozen=True, slots=True)
class Remote:
    url: str


@dataclass(frozen=True, slots=True)
class File:
    path: Path


@dataclass(frozen=True, slots=True)
class Bundled:
    pass


def _normalize(value: object) -> object:
    if isinstance(value, Mapping):
        mapping: Final[Mapping[object, object]] = (
            cast(  # cast-ok: Mapping is checked and values are recursively normalized
                Mapping[object, object], value
            )
        )
        return {key: _normalize(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence: Final[Sequence[object]] = cast(  # cast-ok: list/tuple narrowing establishes the sequence container
            Sequence[object], value
        )
        return [_normalize(item) for item in sequence]
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        mapping: Final[Mapping[str, object]] = cast(  # cast-ok: frozen API entries have string keys and object values
            Mapping[str, object], value
        )
        return MappingProxyType({key: _freeze(item) for key, item in mapping.items()})
    if isinstance(value, list):
        list_sequence: Final[Sequence[object]] = cast(  # cast-ok: list narrowing establishes a sequence container
            Sequence[object], value
        )
        return tuple(_freeze(item) for item in list_sequence)
    if isinstance(value, tuple):
        tuple_sequence: Final[Sequence[object]] = cast(  # cast-ok: tuple narrowing establishes a sequence container
            Sequence[object], value
        )
        return tuple(_freeze(item) for item in tuple_sequence)
    return value


def _freeze_entries(entries: Mapping[str, Mapping[str, object]]) -> _FrozenEntries:
    frozen: Final = _freeze(entries)
    return cast(  # cast-ok: input keys are strings and _freeze preserves mapping keys
        _FrozenEntries, frozen
    )


@dataclass(frozen=True, slots=True)
class Entries:
    entries: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        normalized: Final = cast(  # cast-ok: normalization preserves validated string-keyed entry structure
            Mapping[str, Mapping[str, object]], _normalize(self.entries)
        )
        object.__setattr__(self, "entries", _freeze_entries(normalized))


ModelCostSource: TypeAlias = Remote | File | Bundled | Entries


@dataclass(frozen=True, slots=True)
class FetchedModelCostMap:
    entries: _FrozenEntries
    revision: str
    etag: str | None
    source: ModelCostSource
    _catalog: ModelRegistry | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class ModelCostMapUnavailable:
    reason: str


@dataclass(frozen=True, slots=True)
class RowIssue:
    key: str
    field: str | None
    kind: RowErrorKind
    rejects: bool
    message: str


@dataclass(frozen=True, slots=True)
class ModelCostReloaded:
    row_count: int
    revision: str
    source: ModelCostSource


@dataclass(frozen=True, slots=True)
class ModelCostOverrideResult:
    owner: str
    applied: tuple[str, ...]
    rejected: tuple[RowIssue, ...]


@dataclass(frozen=True, slots=True)
class ModelCostSnapshot:
    entries: _FrozenEntries
    revision: str | None
    source: ModelCostSource | None
    owners: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _Owner:
    precedence: Literal["override", "fallback"]
    entries: Mapping[str, object]


_owners: Mapping[str, _Owner] = MappingProxyType({})  # rebind-ok: immutable owner snapshot is replaced under lock
_current_source: ModelCostSource | None = None  # rebind-ok: source provenance changes only after successful reload


def _source_entries(registry: ModelRegistry) -> _FrozenEntries:
    accepted, _ = registry.without_rejected()
    return _freeze_entries(accepted.served())


def _fetched(catalog: LoadedCatalog, source: ModelCostSource) -> FetchedModelCostMap:
    return FetchedModelCostMap(
        _source_entries(catalog.registry), catalog.revision, catalog.etag, source, catalog.registry
    )


def _entries_catalog(source: Entries) -> LoadedCatalog | ModelCostMapUnavailable:
    normalized: Final = _normalize(source.entries)
    try:
        body: Final = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
        registry: Final = ModelRegistry.parse(body)
    except (TypeError, ValueError) as exc:
        return ModelCostMapUnavailable(f"invalid model cost map entries: {exc}")
    return LoadedCatalog(registry, git_blob_id(body), None)


def _fetch(source: ModelCostSource, timeout: float, max_attempts: int) -> LoadedCatalog | ModelCostMapUnavailable:
    match source:
        case Bundled():
            return _public_result(_load_native(NativeSource("bundled")))
        case File(path=path):
            return _public_result(_load_native(NativeSource("file", str(path), float(timeout))))
        case Entries():
            return _entries_catalog(source)
        case Remote(url=url):
            if is_offline_mode():
                return ModelCostMapUnavailable("remote model cost map fetch disabled in offline mode")
            if not url:
                return ModelCostMapUnavailable("model cost map URL is missing")
            return _public_result(
                _native_revalidation(
                    url=url,
                    timeout_seconds=float(timeout),
                    max_attempts=max_attempts,
                    sleep=time.sleep,
                    installed=None,
                    revision=None,
                    etag=None,
                    installed_url=None,
                    use_current_source=False,
                )
            )
    return ModelCostMapUnavailable("unsupported model cost map source")


def _public_result(loaded: LoadedCatalog | NativeUnavailable) -> LoadedCatalog | ModelCostMapUnavailable:
    return ModelCostMapUnavailable(loaded.reason) if isinstance(loaded, NativeUnavailable) else loaded


def _load_native(source: NativeSource) -> LoadedCatalog | NativeUnavailable:
    from litellm.litellm_core_utils.get_model_cost_map import (
        _load,  # pyright: ignore[reportPrivateUsage]  # native source loading is the fetch seam
    )

    loaded: Final = _load(source)
    return loaded


def fetch_model_cost_map(
    source: ModelCostSource,
    *,
    timeout: float = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
) -> FetchedModelCostMap | ModelCostMapUnavailable:
    loaded: Final = _fetch(source, timeout, max_attempts)
    if isinstance(loaded, ModelCostMapUnavailable):
        return loaded
    return _fetched(loaded, source)


async def afetch_model_cost_map(
    source: ModelCostSource,
    *,
    timeout: float = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
) -> FetchedModelCostMap | ModelCostMapUnavailable:
    return await asyncio.to_thread(fetch_model_cost_map, source, timeout=timeout, max_attempts=max_attempts)


@functools.cache
def bundled_model_cost_map() -> _FrozenEntries:
    return _freeze_entries(ModelRegistry.bundled().served())


def _issue(key: str, error: RowError) -> RowIssue:
    return RowIssue(key, error.field, error.kind, error.rejects, error.message)


def _issues(entries: Mapping[str, object]) -> Iterator[RowIssue]:
    for key, row in entries.items():
        yield from _row_issues(key, row)


def _row_issues(key: str, row: object) -> Iterator[RowIssue]:
    normalized: Final = _normalize(row)
    errors: Final = model_row_errors(key, normalized)
    yield from (_issue(key, error) for error in errors)


def validate_model_cost_entries(entries: Mapping[str, object]) -> tuple[RowIssue, ...]:
    return tuple(_issues(entries))


def _plain_owner_entries(entries: Mapping[str, object]) -> Mapping[str, object]:
    normalized: Final = cast(  # cast-ok: top-level model names are string keys
        Mapping[str, object], _normalize(entries)
    )
    return cast(  # cast-ok: freezing preserves top-level mapping keys and normalizes nested mappings
        Mapping[str, object], _freeze(normalized)
    )


def _owner_native_rows(entries: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    return {
        key: cast(  # cast-ok: native diagnostics reject non-mapping rows before they are stored
            Mapping[str, object], _normalize(cast(Mapping[object, object], row))
        )
        for key, row in entries.items()
        if isinstance(row, Mapping)
    }


def _owner_transport_issues(key: str, row: object) -> tuple[RowIssue, ...]:
    from pydantic import JsonValue

    probe: Final = NativeModelCostLayers()
    try:
        probe.set_owner(
            "probe",
            "override",
            cast(  # cast-ok: the native boundary validates arbitrary input before candidate composition
                dict[str, dict[str, JsonValue]], {key: _normalize(row)}
            ),
        )
    except (ValueError, TypeError):
        return tuple(_row_issues(key, row))
    return ()


def _transport_issues(entries: Mapping[str, object]) -> Iterator[RowIssue]:
    for key, row in entries.items():
        yield from _owner_transport_issues(key, row)


def _owner_order() -> tuple[str, ...]:
    return _owner_names("fallback") + _owner_names("override")


def _owner_names(precedence: Literal["fallback", "override"]) -> tuple[str, ...]:
    return tuple(name for name, owner in _owners.items() if owner.precedence == precedence)


def _composed_issues(
    rows: Sequence[tuple[str, Mapping[str, object] | None, frozenset[str]]],
    keys: frozenset[str],
) -> tuple[RowIssue, ...]:
    return tuple(_iter_composed_issues(rows, keys))


def _iter_composed_issues(
    rows: Sequence[tuple[str, Mapping[str, object] | None, frozenset[str]]],
    keys: frozenset[str],
) -> Iterator[RowIssue]:
    for key, row, _ in rows:
        if key in keys and row is not None:
            yield from _issues({key: row})


def _prepare_owner_layers(registry: ModelRegistry) -> None:  # pyright: ignore[reportUnusedFunction]  # called through the lazy installer
    if not _owners:
        return
    candidate: Final = _MODEL_COST_LAYERS.snapshot()
    order: Final = _owner_order()
    for name in order:
        candidate.drop_owner(name)
    for name in order:
        owner = _owners[name]
        candidate.set_owner(name, owner.precedence, _owner_native_rows(owner.entries))
    keys: Final = frozenset(_owned_keys())
    composed: Final = tuple(candidate.compose_keys(registry, keys))
    rejected: Final = frozenset(issue.key for issue in _composed_issues(composed, keys) if issue.rejects)
    if rejected:
        effective: Final = _MODEL_COST_LAYERS.snapshot()
        for name in order:
            effective.drop_owner(name)
        for name in order:
            effective_owner = _owners[name]
            accepted: dict[str, object] = {
                key: row for key, row in effective_owner.entries.items() if key not in rejected
            }
            effective.set_owner(name, effective_owner.precedence, _owner_native_rows(accepted))
        for key in rejected:
            verbose_logger.warning("LiteLLM: model cost override for %s was skipped after reload", key)
        _MODEL_COST_LAYERS.restore(effective)
        return
    _MODEL_COST_LAYERS.restore(candidate)


def _affected_rows(
    layers: ModelCostLayers,
    registry: ModelRegistry,
    keys: frozenset[str],
) -> tuple[tuple[str, Mapping[str, object] | None, frozenset[str]], ...]:
    composed: Final = tuple(layers.compose(registry))
    selected_identities: Final = frozenset(id(fields) for key, fields, _ in composed if key in keys)
    selected: Final = tuple(row for row in composed if row[0] in keys or id(row[1]) in selected_identities)
    deleted: Final = tuple(row for row in layers.compose_keys(registry, keys) if row[1] is None)
    return selected + deleted


def _sync_source_from_provenance(provenance: Mapping[str, object]) -> ModelCostSource | None:
    source: Final = provenance.get("source")
    url: Final = provenance.get("url")
    if source == "remote" and isinstance(url, str):
        return Remote(url)
    if source == "local":
        return Bundled()
    return None


def _installed_catalog_changed(  # pyright: ignore[reportUnusedFunction]  # invoked from the loaded-module installer callback
    provenance: Mapping[str, object],
) -> None:
    global _current_source  # noqa: PLW0603  # legacy installs replace the source represented by snapshots
    _current_source = _sync_source_from_provenance(provenance)


def _snapshot_entries(model_cost: Mapping[str, object]) -> _FrozenEntries:
    frozen_rows: Final = {id(row): _freeze(_normalize(row)) for row in model_cost.values()}
    entries: Final = {
        key: cast(  # cast-ok: public model cost rows are mappings and remain mappings when normalized
            _FrozenRow, frozen_rows[id(row)]
        )
        for key, row in model_cost.items()
    }
    return MappingProxyType(entries)


def _source_matches_provenance(
    source: ModelCostSource | None,
    provenance: Mapping[str, object],
) -> bool:
    match source:
        case Remote(url=url):
            return provenance.get("source") == "remote" and provenance.get("url") == url
        case Bundled():
            return provenance.get("source") == "local"
        case File():
            return provenance.get("source") == "file"
        case Entries():
            return provenance.get("source") == "entries"
        case None:
            return False


def _snapshot_source(
    source: ModelCostSource | None,
    provenance: Mapping[str, object],
) -> ModelCostSource | None:
    if _source_matches_provenance(source, provenance):
        return source
    return _sync_source_from_provenance(provenance)


def _publish_affected(
    rows: Sequence[tuple[str, Mapping[str, object] | None, frozenset[str]]],
) -> None:
    import litellm
    from litellm import utils

    published: Final = _model_cost_dict_from_native_rows(
        tuple((key, fields, aliases) for key, fields, aliases in rows if fields is not None)
    )
    utils.restore_model_cost_registration_metadata(published)
    live_model_cost: Final = cast(  # cast-ok: public model_cost is a plain dict at runtime
        dict[str, object], litellm.model_cost
    )
    for key, _, _ in rows:
        if key in published:
            dict.__setitem__(  # pyright: ignore[reportUnknownMemberType]  # bypass model_cost's legacy mutation hooks
                live_model_cost, key, published[key]
            )
        else:
            dict.pop(live_model_cost, key, None)  # pyright: ignore[reportUnknownMemberType]  # bypass model_cost's legacy mutation hooks
    utils.invalidate_model_cost_caches()
    litellm.add_known_models(  # pyright: ignore[reportUnknownMemberType]  # litellm's dynamic module export lacks a typed overload
        model_cost_map=cast(  # cast-ok: native conversion returns a mutable string-keyed model map
            dict[str, object], published
        )
    )


def set_model_cost_overrides(
    owner: str,
    entries: Mapping[str, object],
    *,
    precedence: Literal["override", "fallback"] = "override",
) -> ModelCostOverrideResult:
    global _owners  # noqa: PLW0603  # immutable owner state is replaced atomically under the registry lock
    desired: Final = _plain_owner_entries(entries)
    raw_issues: Final = tuple(_transport_issues(desired))
    raw_rejected: Final = frozenset(issue.key for issue in raw_issues if issue.rejects)
    with _MODEL_COST_REGISTRATION_LOCK:
        previous_keys: Final[frozenset[str]] = frozenset(_owners[owner].entries) if owner in _owners else frozenset()
        affected: Final[frozenset[str]] = previous_keys | frozenset(desired)
        registry: Final = GetModelCostMap.loaded_registry()
        candidate: Final = _MODEL_COST_LAYERS.snapshot()
        candidate.drop_owner(owner)
        proposed: Final = {key: value for key, value in desired.items() if key not in raw_rejected}
        candidate.set_owner(owner, precedence, _owner_native_rows(proposed))
        composed: Final = tuple(candidate.compose_keys(registry, affected))
        composed_issues: Final = _composed_issues(composed, frozenset(desired))
        composed_rejected: Final = frozenset(issue.key for issue in composed_issues if issue.rejects)
        final_accepted: Final = MappingProxyType(
            {key: value for key, value in proposed.items() if key not in composed_rejected}
        )
        candidate.drop_owner(owner)
        if final_accepted:
            candidate.set_owner(owner, precedence, _owner_native_rows(final_accepted))
        rows: Final = _affected_rows(candidate, registry, affected)
        remaining_owners: Final = {name: value for name, value in _owners.items() if name != owner}
        _owners = MappingProxyType(  # rebind-ok: immutable owner state is replaced after candidate validation
            remaining_owners
            | (
                {
                    owner: _Owner(
                        precedence,
                        cast(  # cast-ok: accepted entries are frozen model rows
                            Mapping[str, object], _freeze(final_accepted)
                        ),
                    )
                }
                if final_accepted
                else {}
            )
        )
        _MODEL_COST_LAYERS.restore(candidate)
        _publish_affected(rows)
    return ModelCostOverrideResult(
        owner,
        tuple(final_accepted),
        raw_issues + composed_issues,
    )


def _default_source() -> ModelCostSource:
    import litellm

    if is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        return Bundled()
    return Remote(litellm.model_cost_map_url)


def _install_fetched(
    fetched: FetchedModelCostMap, *, is_env_forced: bool
) -> ModelCostReloaded | ModelCostMapUnavailable:
    global _current_source  # noqa: PLW0603  # provenance changes only after a successful catalog install
    registry: Final = (
        fetched._catalog  # pyright: ignore[reportPrivateUsage]  # fetch retains native registry metadata for reload
        if fetched._catalog is not None  # pyright: ignore[reportPrivateUsage]  # see the branch above
        else ModelRegistry.parse(json.dumps(_normalize(fetched.entries), sort_keys=True).encode())
    )
    accepted_registry, _ = registry.without_rejected()
    catalog: Final = LoadedCatalog(
        registry if isinstance(fetched.source, Remote) else accepted_registry,
        fetched.revision,
        fetched.etag,
    )
    source_kind: Final = _source_kind(fetched.source)
    provenance: Final = _source_info(
        catalog,
        source_kind,
        fetched.source.url if isinstance(fetched.source, Remote) else None,
        is_env_forced,
        None,
    )
    with _MODEL_COST_REGISTRATION_LOCK:
        composed: Final = install_loaded_catalog(catalog, provenance=provenance)
        if isinstance(composed, NativeUnavailable):
            return ModelCostMapUnavailable(composed.reason)
        _adopt_installed_model_cost_map(composed)
        _current_source = fetched.source  # rebind-ok: record the source adopted by the process
    return ModelCostReloaded(len(fetched.entries), fetched.revision, fetched.source)


def _source_kind(source: ModelCostSource) -> str:
    match source:
        case Remote():
            return "remote"
        case File():
            return "file"
        case Bundled():
            return "local"
        case Entries():
            return "entries"


def reload_model_cost_map(
    source: ModelCostSource | None = None,
    *,
    timeout: float = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
) -> ModelCostReloaded | ModelCostMapUnavailable:
    defaulted: Final = source is None
    chosen: Final = source if source is not None else _default_source()
    fetched: Final = fetch_model_cost_map(chosen, timeout=timeout, max_attempts=max_attempts)
    if isinstance(fetched, ModelCostMapUnavailable):
        from litellm import verbose_logger

        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.", fetched.reason
        )
        return fetched
    env_forced: Final = defaulted and (
        is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"
    )
    return _install_fetched(fetched, is_env_forced=env_forced)


async def areload_model_cost_map(
    source: ModelCostSource | None = None,
    *,
    timeout: float = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
) -> ModelCostReloaded | ModelCostMapUnavailable:
    defaulted: Final = source is None
    chosen: Final = source if source is not None else _default_source()
    fetched: Final = await afetch_model_cost_map(chosen, timeout=timeout, max_attempts=max_attempts)
    if isinstance(fetched, ModelCostMapUnavailable):
        from litellm import verbose_logger

        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.", fetched.reason
        )
        return fetched
    env_forced: Final = defaulted and (
        is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"
    )
    return _install_fetched(fetched, is_env_forced=env_forced)


def model_cost_snapshot() -> ModelCostSnapshot:
    import litellm

    with _MODEL_COST_REGISTRATION_LOCK:
        provenance: Final = get_model_cost_map_source_info()
        model_cost: Final = _snapshot_entries(
            cast(  # cast-ok: model_cost is a string-keyed public mapping
                Mapping[str, object], litellm.model_cost
            )
        )
        owned_keys: Final = frozenset(_owned_keys())
        owners: Final = MappingProxyType(
            {key: tuple(name for name in _owner_order() if key in _owners[name].entries) for key in owned_keys}
        )
        return ModelCostSnapshot(
            model_cost,
            provenance["source_revision"],
            _snapshot_source(_current_source, provenance),
            owners,
        )


def _owned_keys() -> Iterator[str]:
    for owner in _owners.values():
        yield from owner.entries


__all__ = [
    "Bundled",
    "Entries",
    "FetchedModelCostMap",
    "File",
    "ModelCostMapUnavailable",
    "ModelCostOverrideResult",
    "ModelCostReloaded",
    "ModelCostSnapshot",
    "ModelCostSource",
    "Remote",
    "RowIssue",
    "afetch_model_cost_map",
    "areload_model_cost_map",
    "bundled_model_cost_map",
    "fetch_model_cost_map",
    "model_cost_snapshot",
    "reload_model_cost_map",
    "set_model_cost_overrides",
    "validate_model_cost_entries",
]
