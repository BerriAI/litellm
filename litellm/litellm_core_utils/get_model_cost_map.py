"""
The Python host of the model cost map, https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

Invariant: the Rust core holds the logic every host shares, and each host owns its data.

- The Rust core (``litellm/rust/crates/model-catalog`` and ``model-catalog-refresh``) owns what a cost map is and how a
  newer one is obtained: where a document comes from, fetching and retrying, the checks a document must pass, the rule
  that turns the remote map off, the defaults, and the wording of a failure. The Rust gateway runs the same code.
- This module owns what only the Python host has. It reads its settings and hands the values to the core. It decides
  when a refresh runs: at import on a background thread, on a reload, before returning under uWSGI, and again in a
  forked child. It owns the data: ``litellm.model_cost`` with the provider model sets and ``register_model`` calls on top.

A rule the Rust gateway would also need does not belong in this file.

``LITELLM_LOCAL_MODEL_COST_MAP=True`` or offline mode turns the remote map off. The ``lite`` and ``litellm-proxy`` CLI
entry points also serve the bundled map without fetching.
"""

import atexit
import hashlib
import json
import os
import sys
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from importlib.resources import files
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from pydantic import JsonValue, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm import verbose_logger
from litellm.litellm_core_utils import model_cost_table
from litellm.litellm_core_utils.env_utils import get_env_float_or_none, get_env_int_or_none
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.model_cost_table import (
    AliasedCostEntry as AliasedCostEntry,  # noqa: PLC0414  # main exported the row type from here
)
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.rust_bridge._native import (
    CostMapKept,
    CostMapRefreshed,
    CostMapRefreshStop,
    NativeCostMapSource,
    NativeFetchFailed,
    NativeFetchPolicy,
    NativeInstalled,
    NativeModelCostMap,
    NativeRejected,
)

_CLI_ENTRYPOINT_NAMES: Final = frozenset({"lite", "litellm-proxy"})
_CATALOG_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
_BUNDLED_DOCUMENT_ADAPTER: Final = TypeAdapter(dict[str, object])
FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
RESERVED_TOP_LEVEL_KEYS: Final = frozenset({"sample_spec", FALLBACK_GENERALIZATIONS_KEY})
RETRYABLE_FETCH_STATUS_CODES: Final = frozenset({429, 500, 502, 503, 504})
_litellm_import_complete = threading.Event()

model_cost_inputs: Final = NativeModelCostMap.bundled(
    min_model_count=get_env_int_or_none("MODEL_COST_MAP_MIN_MODEL_COUNT"),
    max_shrink_ratio=get_env_float_or_none("MODEL_COST_MAP_MAX_SHRINK_RATIO"),
)


def _is_cli_process() -> bool:
    return Path(sys.argv[0]).stem in _CLI_ENTRYPOINT_NAMES


def _is_uwsgi_process() -> bool:
    return "uwsgi" in sys.modules


def _remote(url: str, use_http_settings: bool) -> NativeCostMapSource | None:
    """The remote cost map, or ``None`` when this process serves the bundled one only."""
    # Can't use get_secret_bool here: this runs during litellm.__init__ before litellm._key_management_settings is set
    return NativeCostMapSource.remote(
        url,
        local_only=os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true",
        offline=is_offline_mode(),
        use_http_settings=use_http_settings,
    )


@dataclass(frozen=True, slots=True)
class _LoadedSnapshot:
    rows: Mapping[str, Mapping[str, object]]
    fields: frozenset[str]
    keys_by_casefold: Mapping[str, str]


@lru_cache(maxsize=1)
def _loaded_snapshot(revision: str) -> _LoadedSnapshot:
    raw: Final = _CATALOG_ADAPTER.validate_python(model_cost_inputs.loaded())
    return _LoadedSnapshot(
        rows=MappingProxyType({key: MappingProxyType(entry) for key, entry in raw.items()}),
        fields=frozenset(chain.from_iterable(entry.keys() for entry in raw.values())),
        keys_by_casefold=MappingProxyType({key.casefold(): key for key in raw}),
    )


class GetModelCostMap:
    """The bundled document, and the installed one as it was loaded before any registration."""

    @staticmethod
    def read_local_model_cost_map_bytes() -> bytes:
        return files("litellm").joinpath("model_prices_and_context_window_backup.json").read_bytes()

    @staticmethod
    def read_local_model_cost_map_text() -> str:
        return GetModelCostMap.read_local_model_cost_map_bytes().decode("utf-8")

    @staticmethod
    def load_local_model_cost_map() -> dict[str, object]:
        """Load the local backup model cost map bundled with the package."""
        return _BUNDLED_DOCUMENT_ADAPTER.validate_json(GetModelCostMap.read_local_model_cost_map_bytes())

    @staticmethod
    def loaded_model_cost_map() -> Mapping[str, Mapping[str, object]]:
        """The catalog as last loaded (bundled or remote), untouched by ``register_model`` or router registrations."""
        return _loaded_snapshot(model_cost_inputs.provenance.revision).rows

    @staticmethod
    def loaded_model_cost_map_key(key: str) -> str | None:
        """``key`` as the loaded catalog spells it, matched exactly first and then ignoring case."""
        snapshot: Final = _loaded_snapshot(model_cost_inputs.provenance.revision)
        return key if key in snapshot.rows else snapshot.keys_by_casefold.get(key.casefold())

    @staticmethod
    def loaded_model_cost_map_fields() -> frozenset[str]:
        """Every field name that at least one entry of the loaded catalog carries."""
        return _loaded_snapshot(model_cost_inputs.provenance.revision).fields

    @staticmethod
    def get_backup_model_count() -> int:
        """The number of models in the bundled document."""
        return _bundled_model_count()

    _get_backup_model_count = get_backup_model_count

    @staticmethod
    def _check_is_valid_dict(fetched_map: object) -> bool:
        if not isinstance(fetched_map, dict) or not fetched_map:
            verbose_logger.warning("LiteLLM: Fetched model cost map is not a non-empty dict.")
            return False
        return True

    @staticmethod
    def _check_model_count_not_reduced(
        fetched_map: Mapping[str, object],
        backup_model_count: int,
        min_model_count: int = model_cost_inputs.min_model_count,
        max_shrink_ratio: float = model_cost_inputs.max_shrink_ratio,
    ) -> bool:
        fetched_count: Final = _count_model_entries(fetched_map)
        if fetched_count < min_model_count:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map has only %d models (minimum=%d).", fetched_count, min_model_count
            )
            return False
        if backup_model_count > 0 and fetched_count < backup_model_count * max_shrink_ratio:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map shrank significantly (fetched=%d, backup=%d, threshold=%.0f%%).",
                fetched_count,
                backup_model_count,
                max_shrink_ratio * 100,
            )
            return False
        return True

    @staticmethod
    def validate_model_cost_map(
        fetched_map: object,
        backup_model_count: int,
        min_model_count: int = model_cost_inputs.min_model_count,
        max_shrink_ratio: float = model_cost_inputs.max_shrink_ratio,
    ) -> bool:
        """Compat for main's callers. Rust checks fetched documents in ``model_cost_inputs.install``."""
        return isinstance(fetched_map, dict) and (
            GetModelCostMap._check_is_valid_dict(fetched_map)
            and GetModelCostMap._check_model_count_not_reduced(
                fetched_map, backup_model_count, min_model_count, max_shrink_ratio
            )
        )


def mark_litellm_import_complete() -> None:
    _litellm_import_complete.set()


@dataclass(frozen=True, slots=True)
class ModelCostMapReloaded:
    """A reload that installed a new document as ``litellm.model_cost``."""

    model_count: int

    @property
    def model_cost_map(self) -> dict[str, dict[str, object]]:
        """Compat for main's callers. Use ``GetModelCostMap.loaded_model_cost_map()``."""
        return model_cost_inputs.loaded()

    @property
    def revision(self) -> str:
        return model_cost_inputs.provenance.revision

    @property
    def etag(self) -> str | None:
        return model_cost_inputs.provenance.etag


@dataclass(frozen=True, slots=True)
class ModelCostMapReloadUnavailable:
    reason: str


ModelCostMapReloadResult = ModelCostMapReloaded | ModelCostMapReloadUnavailable


@dataclass(frozen=True, slots=True)
class ModelCostMapSourceInfo:
    """Where the current cost map came from, replaced whole by every load so readers never see two loads mixed."""

    source: Literal["local", "remote"] = "local"
    url: str | None = None
    is_env_forced: bool = False
    fallback_reason: str | None = None
    loaded_at: datetime | None = None


class _LastLoad:
    current: ModelCostMapSourceInfo = ModelCostMapSourceInfo()


class CostMapProvenance(TypedDict):
    source_revision: ReadOnly[str | None]
    etag: ReadOnly[str | None]


class CostMapSourceInfo(CostMapProvenance):
    source: ReadOnly[str]
    url: ReadOnly[str | None]
    is_env_forced: ReadOnly[bool]
    fallback_reason: ReadOnly[str | None]
    loaded_at: ReadOnly[str | None]


def get_model_cost_map_provenance() -> CostMapProvenance:
    provenance: Final = model_cost_inputs.provenance
    return {"source_revision": provenance.revision, "etag": provenance.etag}


def get_model_cost_map_source_info() -> CostMapSourceInfo:
    """
    Return metadata about where the current model cost map was loaded from.

    Returns a dict with:
    - source: "local" or "remote"
    - url: the remote URL attempted (or None for local-only)
    - is_env_forced: True if LITELLM_LOCAL_MODEL_COST_MAP=True forced local usage
    - fallback_reason: human-readable reason if remote failed and local was used
    - loaded_at: ISO 8601 time this process last loaded the map
    - source_revision: git blob id of the loaded file's bytes
    - etag: the ETag of the remote fetch (None for the bundled backup)
    """
    last: Final = _LastLoad.current
    return {
        "source": last.source,
        "url": last.url,
        "is_env_forced": last.is_env_forced,
        "fallback_reason": last.fallback_reason,
        "loaded_at": last.loaded_at.isoformat() if last.loaded_at is not None else None,
        **get_model_cost_map_provenance(),
    }


def get_model_cost_map_loaded_at() -> "datetime | None":
    """When this process last loaded its cost map, stamped at the start of every load"""
    return _LastLoad.current.loaded_at


def cost_map_source(url: str, use_http_settings: bool) -> NativeCostMapSource:
    return NativeCostMapSource(url, use_http_settings=use_http_settings)


def fetch_policy(timeout: float | None, max_attempts: int | None) -> NativeFetchPolicy:
    """The core's retry policy, with the two numbers a caller of the public functions may set."""
    return NativeFetchPolicy(max_attempts=max_attempts, timeout_seconds=timeout)


def _set_fallback_rules(block: JsonValue) -> None:
    rules: Final = block.get("rules") if isinstance(block, dict) else None
    set_fallback_generalizations(rules if isinstance(rules, list) else None)


def _rows_with_rules_set(installed: NativeInstalled) -> dict[str, dict[str, object]]:  # mutable-ok: live model_cost
    """The installed document's rows, after making its ``fallback_generalizations`` rules the active ones."""
    _set_fallback_rules(installed.fallback_generalizations)
    return installed.rows


def adopt_model_cost_map(new_model_cost_map: dict) -> int:  # mutable-ok: main's signature, the dict goes live
    """Publishes the table, then writes router rows and kept registrations onto it."""
    import litellm
    from litellm import utils

    model_cost_table.publish(new_model_cost_map)
    litellm.add_known_models(model_cost_map=new_model_cost_map)
    utils.reapply_runtime_model_cost_registrations()
    return len(new_model_cost_map)


def _install(installed: NativeInstalled) -> int:
    _Installs.count += 1
    adopt_model_cost_map(_rows_with_rules_set(installed))
    return installed.model_count


def install_model_cost_map(body: bytes, etag: str | None = None) -> ModelCostMapReloadResult:
    """Installs a cost map document as ``litellm.model_cost``, every kept ``register_model`` call written on top."""
    installed: Final = model_cost_inputs.install(body, etag)
    if isinstance(installed, NativeRejected):
        return ModelCostMapReloadUnavailable(reason=installed.reason)
    return ModelCostMapReloaded(model_count=_install(installed))


def _mark_rows_with_rules_set(
    source: Literal["local", "remote"],
    url: str | None,
    is_env_forced: bool,
    fallback_reason: str | None,
    loaded_at: datetime,
) -> None:
    _LastLoad.current = ModelCostMapSourceInfo(source, url, is_env_forced, fallback_reason, loaded_at)


def _warn_ignored_fetch_hooks(sleep: object, rng: object, client: object) -> None:
    if (sleep, rng, client) != (None, None, None):
        warnings.warn(
            "The sleep, rng and client arguments of the model cost map fetch are ignored, the fetch runs natively",
            DeprecationWarning,
            stacklevel=3,
        )


async def refetch_model_cost_map(
    url: str,
    timeout: float | None = None,
    max_attempts: int | None = None,
    sleep: object = None,
    rng: object = None,
    client: object = None,
) -> ModelCostMapReloadResult:
    """
    Re-fetch the model cost map for a runtime reload, retrying transient HTTP
    errors (429/5xx/transport) with Retry-After-aware backoff, and install it.

    Unlike ``get_model_cost_map`` this never falls back to the packaged backup:
    on failure it returns ``ModelCostMapReloadUnavailable`` and the map already
    installed stays.
    """
    _warn_ignored_fetch_hooks(sleep, rng, client)

    source: Final = _remote(url, use_http_settings=True)
    if source is None:
        _mark_rows_with_rules_set("local", None, True, None, datetime.now(timezone.utc))
        return ModelCostMapReloaded(model_count=_install(model_cost_inputs.install_bundled()))

    refreshed: Final = await model_cost_inputs.refresh(source, fetch_policy(timeout, max_attempts))
    if isinstance(refreshed, CostMapKept):
        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.", refreshed.detail
        )
        return ModelCostMapReloadUnavailable(reason=refreshed.reason)
    _mark_rows_with_rules_set("remote", source.shown_url, False, None, datetime.now(timezone.utc))
    return ModelCostMapReloaded(model_count=_install(model_cost_inputs.install_refreshed(refreshed)))


@dataclass(frozen=True, slots=True)
class _BootRefresh:
    """A background refresh. ``installs`` is the install count when it started, ``ended`` lets a forked child tell
    whether it finished, and ``stop`` ends its download when the interpreter exits."""

    source: NativeCostMapSource
    timeout: float | None
    max_attempts: int | None
    started: datetime
    installs: int
    ended: threading.Event = field(default_factory=threading.Event)
    stop: CostMapRefreshStop = field(default_factory=CostMapRefreshStop)


class _Installs:
    count: int = 0


class _RunningRefreshes:
    current: tuple[_BootRefresh, ...] = ()


def _running_refreshes() -> tuple[_BootRefresh, ...]:
    return tuple(boot for boot in _RunningRefreshes.current if not boot.ended.is_set())


def _replaced_by_a_later_load(boot: _BootRefresh) -> bool:
    """A load that installed a table while this refresh was downloading owns the table and the source info."""
    return _Installs.count != boot.installs


def _refreshed(boot: _BootRefresh) -> CostMapRefreshed | CostMapKept | None:
    """The document at ``boot.source``, or why the local backup stays after recording it. ``None`` when the refresh
    was stopped or a later load replaced this one."""
    refreshed: Final = model_cost_inputs.refresh_blocking(
        boot.source, fetch_policy(boot.timeout, boot.max_attempts), boot.stop
    )
    if refreshed is None or _replaced_by_a_later_load(boot):
        return None
    if isinstance(refreshed, CostMapKept):
        verbose_logger.warning(
            "LiteLLM: Remote model cost map was not installed (%s). Keeping the local backup.", refreshed.detail
        )
        _mark_rows_with_rules_set("local", boot.source.shown_url, False, refreshed.fallback_reason, boot.started)
    return refreshed


def _refresh_in_background(boot: _BootRefresh) -> None:
    try:
        refreshed: Final = _refreshed(boot)
        if not isinstance(refreshed, CostMapRefreshed):
            return
        _litellm_import_complete.wait()
        if _replaced_by_a_later_load(boot):
            return
        installed: Final = model_cost_inputs.install_refreshed(refreshed)
        _mark_rows_with_rules_set("remote", boot.source.shown_url, False, None, datetime.now(timezone.utc))
        _install(installed)
    except Exception as e:  # noqa: BLE001  # a failed background refresh must not kill the thread; the backup stays
        verbose_logger.warning("LiteLLM: Background model cost map refresh failed: %s", e)
    finally:
        boot.ended.set()


def _start_refresh(
    source: NativeCostMapSource, timeout: float | None, max_attempts: int | None, started: datetime
) -> None:
    boot: Final = _BootRefresh(source, timeout, max_attempts, started, _Installs.count)
    _RunningRefreshes.current = (*_running_refreshes(), boot)
    threading.Thread(
        target=_refresh_in_background, args=(boot,), name="litellm-model-cost-map-refresh", daemon=True
    ).start()


def _refresh_again_in_forked_child() -> None:
    """A fork copies no threads, so a child forked during the parent's refresh would serve the backup forever."""
    unfinished: Final = _running_refreshes()
    _RunningRefreshes.current = ()
    for boot in unfinished:
        _start_refresh(boot.source, boot.timeout, boot.max_attempts, boot.started)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_refresh_again_in_forked_child)


def _stop_refresh_at_exit() -> None:
    """A daemon thread still in native code once the interpreter is finalizing crashes the process, so every download
    is stopped and an install already under way runs to its end. Without a finished import a thread waits on it
    forever outside native code, so there is nothing to wait for."""
    running: Final = _running_refreshes()
    for boot in running:
        boot.stop.stop()
    if not _litellm_import_complete.is_set():
        return
    for boot in running:
        _ = boot.ended.wait()


atexit.register(_stop_refresh_at_exit)


def get_model_cost_map(
    url: str,
    timeout: float | None = None,
    max_attempts: int | None = None,
    sleep: object = None,
    rng: object = None,
    client: object = None,
) -> dict:
    """
    Public entry point — returns the model cost map dict.

    1. If ``LITELLM_LOCAL_MODEL_COST_MAP`` is set or this is a ``lite`` /
       ``litellm-proxy`` CLI process, uses the local backup only.
    2. Otherwise fetches from ``url`` once, retrying transient HTTP errors
       (429/5xx/transport) with Retry-After-aware backoff in a background
       thread, validates integrity, and falls back to the local backup on any
       failure.
    """
    _warn_ignored_fetch_hooks(sleep, rng, client)
    started: Final = datetime.now(timezone.utc)
    source: Final = None if _is_cli_process() else _remote(url, use_http_settings=False)
    if source is None:
        _mark_rows_with_rules_set("local", None, True, None, started)
        return _rows_with_rules_set(model_cost_inputs.install_bundled())

    _mark_rows_with_rules_set("local", source.shown_url, False, None, started)
    refreshed: Final = _refreshed(_BootRefresh(source, timeout, 1, started, _Installs.count))
    if isinstance(refreshed, CostMapRefreshed):
        _mark_rows_with_rules_set("remote", source.shown_url, False, None, started)
        return _rows_with_rules_set(model_cost_inputs.install_refreshed(refreshed))
    retries: Final = fetch_policy(timeout, max_attempts).max_attempts - 1
    if isinstance(refreshed, CostMapKept) and refreshed.retryable and retries > 0:
        _start_refresh(source, timeout, retries, started)
    return _rows_with_rules_set(model_cost_inputs.install_bundled())


def load_model_cost_map_for_import(url: str) -> dict:  # mutable-ok: live model_cost
    """``litellm.model_cost`` at import: the bundled map, while a background thread fetches ``url`` and installs it.
    uWSGI forks its workers without that thread, so there the import waits for one attempt instead."""
    source: Final = None if _is_cli_process() or _is_uwsgi_process() else _remote(url, use_http_settings=False)
    if source is None:
        return get_model_cost_map(url, max_attempts=1)
    started: Final = datetime.now(timezone.utc)
    _mark_rows_with_rules_set("local", source.shown_url, False, None, started)
    bundled: Final = model_cost_inputs.install_bundled()
    _start_refresh(source, None, None, started)
    return _rows_with_rules_set(bundled)


def registration_rows(body: bytes) -> dict[str, dict[str, object]]:  # mutable-ok: register_model merges these rows
    """The rows of a fetched ``register_model`` document, without its fallback rules block. Raises ``ValueError`` on
    a body that is not a JSON object of rows."""
    document: Final = _CATALOG_ADAPTER.validate_python(json.loads(body))
    return {key: entry for key, entry in document.items() if key != FALLBACK_GENERALIZATIONS_KEY}


def fetch_registration_document(url: str) -> dict[str, dict[str, object]] | None:
    """The document behind ``register_model(url)``, fetched in one attempt. ``None`` when offline mode forbids the
    fetch or it failed."""
    if is_offline_mode():
        verbose_logger.warning("LiteLLM: register_model did not fetch %s because LITELLM_OFFLINE_MODE is set", url)
        return None
    fetched: Final = cost_map_source(url, use_http_settings=False).fetch_blocking(fetch_policy(5, 1))
    if isinstance(fetched, NativeFetchFailed):
        verbose_logger.warning("LiteLLM: register_model could not fetch %s: %s", url, fetched.reason)
        return None
    try:
        return registration_rows(fetched.body)
    except ValueError as e:
        verbose_logger.warning("LiteLLM: register_model got an invalid cost map from %s: %s", url, e)
        return None


def _finalize_model_cost_map(  # pyright: ignore[reportUnusedFunction]  # main's private name, downstream code calls it
    model_cost: dict[str, object],  # mutable-ok: main's signature, pops the rules block
) -> dict[str, dict[str, object]]:
    """Compat for main's callers. Use ``model_cost_inputs.install``."""
    match NativeModelCostMap.finalize(json.dumps(model_cost).encode()):
        case NativeRejected(reason=reason):
            raise ValueError(reason)
        case installed:
            model_cost.pop(FALLBACK_GENERALIZATIONS_KEY, None)
            return _rows_with_rules_set(installed)


def git_blob_id(body: bytes) -> str:
    """Compat for main's callers. ``git hash-object`` of the bytes, which ``source_revision`` reports."""
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


def _count_model_entries(model_cost: Mapping[str, object]) -> int:
    return sum(1 for key in model_cost if key not in RESERVED_TOP_LEVEL_KEYS)


@lru_cache(maxsize=1)
def _bundled_model_count() -> int:
    return _count_model_entries(GetModelCostMap.load_local_model_cost_map())
