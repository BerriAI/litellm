"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.
The ``lite`` and ``litellm-proxy`` CLI entry points also use the bundled map without fetching.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import atexit
import functools
import json
import os
import random
import sys
import threading
import warnings
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.resources import files
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import ConfigDict, TypeAdapter
from typing_extensions import ReadOnly, TypedDict, assert_never

from litellm import verbose_logger
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_RATIO,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.rust_bridge._native import (
    AliasWarning,
    ModelCostMapFetch,
    ModelCostMapRefresh,
    ModelCountProblem,
    count_model_entries,
    expand_model_aliases,
    git_blob_id,
    install_model_cost_map_provenance,
    model_cost_map_provenance,
    model_count_problem,
    take_fallback_generalization_rules,
)

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
_CATALOG_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
_DECODED_JSON: Final = TypeAdapter(object)
_BUNDLED_CATALOG_ADAPTER: Final = TypeAdapter(dict[str, object], config=ConfigDict(hide_input_in_errors=True))
_CLI_ENTRYPOINT_NAMES: Final = frozenset({"lite", "litellm-proxy"})


def _is_cli_process() -> bool:
    return Path(sys.argv[0]).stem in _CLI_ENTRYPOINT_NAMES


class GetModelCostMap:
    """
    Handles fetching, validating, and loading the model cost map.

    Only the backup model *count* is cached (a single int). The full
    backup dict is never held in memory — it is only parsed when it
    needs to be *returned* as a fallback.
    """

    _backup_model_count: int = -1  # -1 = not yet loaded

    @staticmethod
    def read_local_model_cost_map_bytes() -> bytes:
        return files("litellm").joinpath("model_prices_and_context_window_backup.json").read_bytes()

    @staticmethod
    def read_local_model_cost_map_text() -> str:
        return GetModelCostMap.read_local_model_cost_map_bytes().decode("utf-8")

    @staticmethod
    def load_local_model_cost_map_with_revision() -> "ModelCostMapReloaded":
        body: Final = GetModelCostMap.read_local_model_cost_map_bytes()
        content: Final = _BUNDLED_CATALOG_ADAPTER.validate_python(json.loads(body))
        return ModelCostMapReloaded(model_cost_map=content, revision=git_blob_id(body))

    @staticmethod
    def load_local_model_cost_map() -> dict:
        """Load the local backup model cost map bundled with the package."""
        return GetModelCostMap.load_local_model_cost_map_with_revision().model_cost_map

    _loaded_catalog: Mapping[str, Mapping[str, object]] = MappingProxyType({})
    _loaded_catalog_fields: frozenset[str] = frozenset()

    @classmethod
    def loaded_model_cost_map(cls) -> Mapping[str, Mapping[str, object]]:
        """The catalog as last loaded (bundled or remote), untouched by ``register_model`` or router registrations."""
        return cls._loaded_catalog

    @classmethod
    def loaded_model_cost_map_fields(cls) -> frozenset[str]:
        """Every field name that at least one entry of the loaded catalog carries."""
        return cls._loaded_catalog_fields

    @classmethod
    def _snapshot_loaded_catalog(cls, model_cost: Mapping[str, object]) -> None:
        raw: Final = _CATALOG_ADAPTER.validate_python(model_cost)
        cls._loaded_catalog = MappingProxyType({key: MappingProxyType(entry) for key, entry in raw.items()})
        cls._loaded_catalog_fields = frozenset(chain.from_iterable(entry.keys() for entry in raw.values()))

    @classmethod
    def get_backup_model_count(cls) -> int:
        """Return the number of models in the local backup (cached int)."""
        if cls._backup_model_count < 0:
            backup: Final = cls.load_local_model_cost_map()
            cls._backup_model_count = count_model_entries(backup)
        return cls._backup_model_count

    _get_backup_model_count = get_backup_model_count

    @staticmethod
    def _check_is_valid_dict(fetched_map: dict) -> bool:
        """Check 1: fetched map is a non-empty dict."""
        if not isinstance(fetched_map, dict):
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map is not a dict (type=%s). Falling back to local backup.",
                type(fetched_map).__name__,
            )
            return False

        if len(fetched_map) == 0:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map is empty. Falling back to local backup.",
            )
            return False

        return True

    @classmethod
    def _check_model_count_not_reduced(
        cls,
        fetched_map: dict,
        backup_model_count: int,
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio: float = MODEL_COST_MAP_MAX_SHRINK_RATIO,
    ) -> bool:
        """Check 2: model count has not reduced significantly vs backup."""
        problem: Final = model_count_problem(fetched_map, backup_model_count, min_model_count, max_shrink_ratio)
        match problem:
            case None:
                return True
            case ModelCountProblem.BelowMinimum(actual=actual, minimum=minimum):
                verbose_logger.warning(
                    "LiteLLM: Fetched model cost map has only %d models (minimum=%d). "
                    "This may indicate a corrupted upstream file. "
                    "Falling back to local backup.",
                    actual,
                    minimum,
                )
            case ModelCountProblem.Shrunk(actual=actual, reference=reference):
                verbose_logger.warning(
                    "LiteLLM: Fetched model cost map shrank significantly "
                    "(fetched=%d, backup=%d, threshold=%.0f%%). "
                    "This may indicate a corrupted upstream file. "
                    "Falling back to local backup.",
                    actual,
                    reference,
                    max_shrink_ratio * 100,
                )
            case _:
                assert_never(problem)
        return False

    @classmethod
    def validate_model_cost_map(
        cls,
        fetched_map: dict,
        backup_model_count: int,
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio: float = MODEL_COST_MAP_MAX_SHRINK_RATIO,
    ) -> bool:
        """
        Validate the integrity of a fetched model cost map.

        Runs each check in order and returns False on the first failure.

        Checks:
        1. ``_check_is_valid_dict`` -- fetched map is a non-empty dict.
        2. ``_check_model_count_not_reduced`` -- model count meets minimum
           and has not shrunk >``max_shrink_ratio`` vs backup.

        Returns True if all checks pass, False otherwise.
        """
        if not cls._check_is_valid_dict(fetched_map):
            return False

        if not cls._check_model_count_not_reduced(
            fetched_map=fetched_map,
            backup_model_count=backup_model_count,
            min_model_count=min_model_count,
            max_shrink_ratio=max_shrink_ratio,
        ):
            return False

        return True


MODEL_COST_MAP_FETCH_MAX_ATTEMPTS: Final = 3
_litellm_import_complete = threading.Event()


class _PendingImportRefresh:
    """Single slot for the URL ``import litellm`` refreshes from once the import has finished"""

    url: str | None = None


class _BackgroundAdoption:
    """A background fetch adopts under ``lock`` unless the interpreter is exiting. A thread that is still importing when
    finalization hangs it holds the import lock and deadlocks the exit, so exit waits for an adoption in progress and
    stops the next one"""

    lock: Final = threading.Lock()
    exiting: Final = threading.Event()


def _stop_background_adoption() -> None:
    with _BackgroundAdoption.lock:
        _BackgroundAdoption.exiting.set()


atexit.register(_stop_background_adoption)


def mark_litellm_import_complete() -> None:
    _litellm_import_complete.set()
    url: Final = _PendingImportRefresh.url
    _PendingImportRefresh.url = None
    if url is not None:
        _start_import_refresh(url)


@dataclass(frozen=True, slots=True)
class ModelCostMapReloaded:
    model_cost_map: dict  # mutable-ok: adopted as litellm.model_cost, whose consumer contract is a plain mutable dict
    revision: str | None = None
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class ModelCostMapReloadUnavailable:
    reason: str


ModelCostMapReloadResult = ModelCostMapReloaded | ModelCostMapReloadUnavailable


def _warn_ignored_client(client: object) -> None:
    if client is not None:
        warnings.warn(
            "client= is ignored: the model cost map is fetched in Rust with litellm's HTTP settings",
            DeprecationWarning,
            stacklevel=3,
        )


def _warn_retrying(attempt: int, max_attempts: int, reason: str, wait_seconds: float) -> None:
    verbose_logger.warning(
        "LiteLLM: model cost map fetch attempt %d/%d failed (%s); retrying in %.1fs",
        attempt,
        max_attempts,
        reason,
        wait_seconds,
    )


def _decoded(
    fetched: ModelCostMapFetch.Adopted
    | ModelCostMapFetch.NotModified
    | ModelCostMapFetch.Failed
    | ModelCostMapFetch.Retryable,
    url: str,
) -> ModelCostMapReloadResult:
    match fetched:
        case ModelCostMapFetch.Adopted(body=body, revision=revision, etag=etag):
            pass
        case ModelCostMapFetch.Failed(reason=reason) | ModelCostMapFetch.Retryable(reason=reason):
            return ModelCostMapReloadUnavailable(reason=reason)
        case ModelCostMapFetch.NotModified():
            return ModelCostMapReloadUnavailable(reason=f"{url} was not modified")
        case _:
            assert_never(fetched)
    try:
        parsed: Final = _DECODED_JSON.validate_python(json.loads(body))
    except ValueError as e:
        return ModelCostMapReloadUnavailable(reason=f"invalid JSON from {url}: {e}")
    if not isinstance(parsed, dict):
        return ModelCostMapReloadUnavailable(reason=f"expected a JSON object from {url}, got {type(parsed).__name__}")
    model_cost_map: Final = _BUNDLED_CATALOG_ADAPTER.validate_python(parsed)
    return ModelCostMapReloaded(model_cost_map=model_cost_map, revision=revision, etag=etag)


def _uses_bundled_map_only() -> bool:
    # Note: can't use get_secret_bool here — this runs during litellm.__init__
    # before litellm._key_management_settings is set.
    return is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"


def _install_bundled(
    attempted_url: str | None, fallback_reason: str | None, is_env_forced: bool = False
) -> dict[str, object]:
    loaded: Final = _finalize_loaded_model_cost_map(GetModelCostMap.load_local_model_cost_map_with_revision())
    install_model_cost_map_provenance(
        source="local",
        url=attempted_url,
        is_env_forced=is_env_forced,
        fallback_reason=fallback_reason,
        revision=loaded.revision,
        etag=None,
    )
    return loaded.model_cost_map


def _install_remote(fetched: ModelCostMapReloaded, url: str) -> ModelCostMapReloaded:
    finalized: Final = _finalize_loaded_model_cost_map(fetched)
    install_model_cost_map_provenance(
        source="remote",
        url=url,
        is_env_forced=False,
        fallback_reason=None,
        revision=finalized.revision,
        etag=finalized.etag,
    )
    return finalized


def _keep_bundled_after(url: str, fallback_reason: str) -> None:
    """The bundled map served at import stays, now standing in for a refresh of ``url`` that failed. A map another
    load installed meanwhile keeps its own provenance"""
    current: Final = model_cost_map_provenance()
    if current is None or current.source != "local" or current.url != url:
        return
    install_model_cost_map_provenance(
        source="local",
        url=url,
        is_env_forced=False,
        fallback_reason=fallback_reason,
        revision=current.revision,
        etag=None,
    )


async def refetch_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    rng: random.Random | None = None,
    client: object = None,
) -> ModelCostMapReloadResult:
    """
    Re-fetch the model cost map for a runtime reload, retrying transient HTTP
    errors (429/5xx/transport) with Retry-After-aware backoff.

    Unlike ``get_model_cost_map`` this never falls back to the packaged backup:
    on failure it returns ``ModelCostMapReloadUnavailable`` so callers keep the
    map they already have. ``sleep`` defaults to waiting in Rust, and ``client``
    is ignored.
    """
    _warn_ignored_client(client)
    if _uses_bundled_map_only():
        loaded: Final = _finalize_loaded_model_cost_map(GetModelCostMap.load_local_model_cost_map_with_revision())
        install_model_cost_map_provenance(
            source="local", url=None, is_env_forced=True, fallback_reason=None, revision=loaded.revision, etag=None
        )
        return loaded

    fetched: Final = await ModelCostMapRefresh(
        url,
        timeout=timeout,
        max_attempts=max_attempts,
        rng=rng if rng is not None else random.Random(),
        on_retry=_warn_retrying,
        sleep=sleep,
    ).arun()
    result: Final = _decoded(fetched, url)
    if isinstance(result, ModelCostMapReloadUnavailable):
        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.",
            result.reason,
        )
        return result
    if not GetModelCostMap.validate_model_cost_map(
        fetched_map=result.model_cost_map,
        backup_model_count=GetModelCostMap.get_backup_model_count(),
    ):
        return ModelCostMapReloadUnavailable(reason=f"model cost map from {url} failed integrity validation")
    return _install_remote(result, url)


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
    provenance: Final = model_cost_map_provenance()
    return {
        "source_revision": provenance.revision if provenance is not None else None,
        "etag": provenance.etag if provenance is not None else None,
    }


def get_model_cost_map_source_info() -> CostMapSourceInfo:
    """
    Return metadata about where the current model cost map was loaded from.

    Returns a dict with:
    - source: "local" or "remote"
    - url: the remote URL attempted (or None for local-only)
    - is_env_forced: True if LITELLM_LOCAL_MODEL_COST_MAP=True forced local usage
    - fallback_reason: human-readable reason if remote failed and local was used
    - loaded_at: ISO 8601 time this process last installed the map
    - source_revision: git blob id of the loaded file's bytes
    - etag: the ETag of the remote fetch (None for the bundled backup)
    """
    provenance: Final = model_cost_map_provenance()
    if provenance is None:
        return {
            "source": "local",
            "url": None,
            "is_env_forced": False,
            "fallback_reason": None,
            "loaded_at": None,
            "source_revision": None,
            "etag": None,
        }
    return {
        "source": provenance.source,
        "url": provenance.url,
        "is_env_forced": provenance.is_env_forced,
        "fallback_reason": provenance.fallback_reason,
        "loaded_at": datetime.fromtimestamp(provenance.loaded_at, timezone.utc).isoformat(),
        "source_revision": provenance.revision,
        "etag": provenance.etag,
    }


def get_model_cost_map_loaded_at() -> "datetime | None":
    """When this process last installed its cost map"""
    provenance: Final = model_cost_map_provenance()
    return datetime.fromtimestamp(provenance.loaded_at, timezone.utc) if provenance is not None else None


class AliasedCostEntry(dict[str, object]):
    __slots__ = ("names", "registered_fields")

    def __init__(
        self, entry: Mapping[str, object], names: tuple[str, ...], registered_fields: frozenset[str] = frozenset()
    ) -> None:
        super().__init__(entry)
        self.names: tuple[str, ...] = names
        self.registered_fields: frozenset[str] = registered_fields


def _warn_skipped_alias(warning: AliasWarning.NotAList | AliasWarning.ShadowsModel | AliasWarning.Taken) -> None:
    match warning:
        case AliasWarning.NotAList(model=model, type_name=type_name):
            verbose_logger.warning(
                "LiteLLM model alias field for '%s' is not a list (got %s) — skipping.",
                model,
                type_name,
            )
        case AliasWarning.ShadowsModel(alias=alias, model=model):
            verbose_logger.warning(
                "LiteLLM model alias conflict: alias '%s' (from '%s') already exists as a canonical entry — skipping.",
                alias,
                model,
            )
        case AliasWarning.Taken(alias=alias, model=model):
            verbose_logger.warning(
                "LiteLLM model alias conflict: alias '%s' (from '%s') was already claimed by another entry — skipping.",
                alias,
                model,
            )
        case _:
            assert_never(warning)


def _expand_model_aliases(model_cost: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """
    Expand ``aliases`` lists in model cost entries into top-level entries.

    Each alias gets a reference to the **same** ``AliasedCostEntry`` as the
    canonical entry, which carries every name it is served under.  The
    ``aliases`` key is removed from the entry so downstream code never sees it.

    If an alias collides with an existing canonical entry the alias is
    skipped and a warning is logged.
    """
    warnings, error = expand_model_aliases(model_cost, AliasedCostEntry)
    for warning in warnings:
        _warn_skipped_alias(warning)
    if error is not None:
        raise error
    return model_cost


def _finalize_model_cost_map(model_cost: dict) -> dict:
    """Extract fallback generalizations out of the raw map, then expand aliases.

    The ``fallback_generalizations`` block is installed into the generalizations
    module and removed from the map so it is never treated as a model entry.
    """
    rules: Final = take_fallback_generalization_rules(model_cost)
    set_fallback_generalizations(rules if isinstance(rules, list) else None)
    return _expand_model_aliases(model_cost)


def _finalize_loaded_model_cost_map(loaded: ModelCostMapReloaded) -> ModelCostMapReloaded:
    finalized: Final = _finalize_model_cost_map(loaded.model_cost_map)
    GetModelCostMap._snapshot_loaded_catalog(finalized)  # pyright: ignore[reportPrivateUsage]  # same module
    return replace(loaded, model_cost_map=finalized)


MODEL_COST_ADOPTION_LOCK: Final = threading.RLock()
"""Held while a map is adopted and while a router registers its deployments and joins the replay, so an adoption in
another thread never lands between the two and drops that router's rows"""


def adopt_model_cost_map(
    new_model_cost_map: dict,  # mutable-ok: public API preserves the mutable cost-map contract
) -> int:
    import litellm
    from litellm import utils

    with MODEL_COST_ADOPTION_LOCK:
        litellm.model_cost = new_model_cost_map
        utils._invalidate_model_cost_lowercase_map()  # pyright: ignore[reportPrivateUsage]  # required cache invalidation
        litellm.add_known_models(model_cost_map=new_model_cost_map)
        fetched_model_count: Final = len(new_model_cost_map) if new_model_cost_map else 0
        utils.reapply_runtime_model_cost_registrations()
    return fetched_model_count


def _retry_remote_fetch_in_background(
    refresh: ModelCostMapRefresh, url: str, max_attempts: int, first_outcome: ModelCostMapFetch.Retryable
) -> None:
    try:
        fetched: Final = refresh.run(first=first_outcome)
        result: Final = _decoded(fetched, url)
        if isinstance(result, ModelCostMapReloadUnavailable):
            verbose_logger.warning(
                "LiteLLM: Failed to fetch remote model cost map from %s after %d attempts; keeping local backup",
                url,
                max_attempts,
            )
            return
        _litellm_import_complete.wait()
        with _BackgroundAdoption.lock:
            if _BackgroundAdoption.exiting.is_set():
                return
            if not GetModelCostMap.validate_model_cost_map(
                fetched_map=result.model_cost_map,
                backup_model_count=GetModelCostMap.get_backup_model_count(),
            ):
                verbose_logger.warning(
                    "LiteLLM: Fetched model cost map failed integrity check. Using local backup instead. url=%s",
                    url,
                )
                return
            adopt_model_cost_map(_install_remote(result, url).model_cost_map)
    except Exception as e:  # noqa: BLE001  # a failed background retry must not kill the task; the backup stays
        verbose_logger.warning("LiteLLM: Background model cost map retry failed: %s", e)


_import_refresh_adopted: Final = threading.Event()
_import_refresh_idle: Final = threading.Event()
_import_refresh_idle.set()
_FORK_WAIT_SECONDS: Final = 5.0


def _refresh_in_background(refresh: ModelCostMapRefresh, url: str) -> None:
    try:
        fetched: Final = refresh.run_detached()
        with _BackgroundAdoption.lock:
            if _BackgroundAdoption.exiting.is_set():
                return
            _adopt_import_refresh(_decoded(fetched, url), url)
    except Exception as e:  # noqa: BLE001  # a failed background refresh must not kill the thread; the bundled map stays
        verbose_logger.warning("LiteLLM: Background model cost map refresh failed: %s", e)
    finally:
        _import_refresh_idle.set()


def _adopt_import_refresh(result: ModelCostMapReloadResult, url: str) -> None:
    if isinstance(result, ModelCostMapReloadUnavailable):
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. Keeping the bundled map.",
            url,
            result.reason,
        )
        _keep_bundled_after(url, f"Remote fetch failed: {result.reason}")
        return
    if not GetModelCostMap.validate_model_cost_map(
        fetched_map=result.model_cost_map,
        backup_model_count=GetModelCostMap.get_backup_model_count(),
    ):
        verbose_logger.warning(
            "LiteLLM: Fetched model cost map failed integrity check. Keeping the bundled map. url=%s",
            url,
        )
        _keep_bundled_after(url, "Remote data failed integrity validation")
        return
    adopt_model_cost_map(_install_remote(result, url).model_cost_map)
    _import_refresh_adopted.set()


def _start_import_refresh(url: str) -> None:
    refresh: Final = ModelCostMapRefresh.at_import(
        url,
        timeout=5,
        max_attempts=MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
        rng=random.Random(),
        on_retry=_warn_retrying,
    )
    _import_refresh_idle.clear()
    threading.Thread(
        target=_refresh_in_background,
        args=(refresh, url),
        name="litellm-model-cost-map-refresh",
        daemon=True,
    ).start()


def _let_the_import_refresh_finish_before_fork() -> None:
    """A fork while the refresh thread is inside the system resolver leaves the child's resolver broken (macOS
    segfaults on the child's next lookup), so a fork waits up to one attempt's timeout for the refresh to finish"""
    _import_refresh_idle.wait(timeout=_FORK_WAIT_SECONDS)


def _restart_import_refresh_in_child(url: str) -> None:
    """A worker forked before its parent adopted the remote map, as gunicorn and granian fork theirs, fetches it itself"""
    if _litellm_import_complete.is_set() and not _import_refresh_adopted.is_set():
        _start_import_refresh(url)


def load_model_cost_map_at_import(url: str) -> dict:
    """The bundled map, served at once while ``url`` is fetched in the background.

    The fetch starts once ``import litellm`` finishes, and a map that passes the integrity
    check is adopted. It runs on a runtime that ends with it, so the process can still fork.
    ``LITELLM_LOCAL_MODEL_COST_MAP``, offline mode and the ``lite`` / ``litellm-proxy``
    CLI processes never fetch.
    """
    if _uses_bundled_map_only() or _is_cli_process():
        return _install_bundled(attempted_url=None, fallback_reason=None, is_env_forced=True)
    bundled: Final = _install_bundled(attempted_url=url, fallback_reason=None)
    os.register_at_fork(
        before=_let_the_import_refresh_finish_before_fork,
        after_in_child=functools.partial(_restart_import_refresh_in_child, url),
    )
    if _litellm_import_complete.is_set():
        _start_import_refresh(url)
    else:
        _PendingImportRefresh.url = url
    return bundled


def get_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], None] | None = None,
    rng: random.Random | None = None,
    client: object = None,
) -> dict:
    """
    Public entry point — returns the model cost map dict.

    1. If ``LITELLM_LOCAL_MODEL_COST_MAP`` is set or this is a ``lite`` /
       ``litellm-proxy`` CLI process, uses the local backup only.
    2. Otherwise fetches from ``url``, retrying transient HTTP errors
       (429/5xx/transport) with Retry-After-aware backoff in a background
       thread, validates integrity, and falls back to the local backup on any
       failure.

    ``sleep`` defaults to waiting in Rust, and ``client`` is ignored.
    """
    _warn_ignored_client(client)
    if _uses_bundled_map_only() or _is_cli_process():
        return _install_bundled(attempted_url=None, fallback_reason=None, is_env_forced=True)

    refresh: Final = ModelCostMapRefresh(
        url,
        timeout=timeout,
        max_attempts=max_attempts,
        rng=rng if rng is not None else random.Random(),
        on_retry=_warn_retrying,
        sleep=sleep,
    )
    first: Final = refresh.attempt()
    if isinstance(first, ModelCostMapFetch.Retryable) and max_attempts > 1:
        threading.Thread(
            target=_retry_remote_fetch_in_background,
            kwargs={"refresh": refresh, "url": url, "max_attempts": max_attempts, "first_outcome": first},
            name="litellm-model-cost-map-retry",
            daemon=True,
        ).start()
    outcome: Final = _decoded(first, url)
    if isinstance(outcome, ModelCostMapReloadUnavailable):
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. Falling back to local backup.",
            url,
            outcome.reason,
        )
        return _install_bundled(attempted_url=url, fallback_reason=f"Remote fetch failed: {outcome.reason}")

    # Validate using cached count (cheap int comparison, no file I/O)
    if not GetModelCostMap.validate_model_cost_map(
        fetched_map=outcome.model_cost_map,
        backup_model_count=GetModelCostMap.get_backup_model_count(),
    ):
        verbose_logger.warning(
            "LiteLLM: Fetched model cost map failed integrity check. Using local backup instead. url=%s",
            url,
        )
        return _install_bundled(attempted_url=url, fallback_reason="Remote data failed integrity validation")

    return _install_remote(outcome, url).model_cost_map
