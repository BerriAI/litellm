"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.
The ``lite`` and ``litellm-proxy`` CLI entry points also use the bundled map without fetching.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import asyncio
import hashlib
import json
import random
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from types import MappingProxyType
from typing import (
    Final,
    Protocol,
)

import httpx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm import verbose_logger
from litellm._catalog._load import CatalogUnusable, LoadedCatalog, load, served_rows
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_RATIO,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)
from litellm.rust_bridge._native import ModelRegistry

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
_CATALOG_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
# Reserved top-level keys that are not model entries. They must be excluded
# from the model-count integrity check so a real upstream shrink can't be masked.
RESERVED_TOP_LEVEL_KEYS: Final = frozenset({"sample_spec", FALLBACK_GENERALIZATIONS_KEY})


def _count_model_entries(model_cost: dict) -> int:
    """Count actual model entries, excluding reserved meta keys."""
    return sum(1 for key in model_cost if key not in RESERVED_TOP_LEVEL_KEYS)


def git_blob_id(body: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(body) + body, usedforsecurity=False).hexdigest()


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
        return ModelRegistry.bundled_bytes()

    @staticmethod
    def read_local_model_cost_map_text() -> str:
        return GetModelCostMap.read_local_model_cost_map_bytes().decode("utf-8")

    @staticmethod
    def load_local_model_cost_map_with_revision() -> "ModelCostMapReloaded":
        body: Final = GetModelCostMap.read_local_model_cost_map_bytes()
        content: Final = ModelRegistry.bundled().to_dict()
        return ModelCostMapReloaded(model_cost_map=content, revision=git_blob_id(body))

    @staticmethod
    def load_local_model_cost_map() -> dict:
        """Load the raw model cost map embedded in the native extension."""
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
            cls._backup_model_count = _count_model_entries(backup)
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
        fetched_count: Final = _count_model_entries(fetched_map)

        if fetched_count < min_model_count:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map has only %d models (minimum=%d). "
                "This may indicate a corrupted upstream file. "
                "Falling back to local backup.",
                fetched_count,
                min_model_count,
            )
            return False

        if backup_model_count > 0 and fetched_count < backup_model_count * max_shrink_ratio:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map shrank significantly "
                "(fetched=%d, backup=%d, threshold=%.0f%%). "
                "This may indicate a corrupted upstream file. "
                "Falling back to local backup.",
                fetched_count,
                backup_model_count,
                max_shrink_ratio * 100,
            )
            return False

        return True

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


RETRYABLE_FETCH_STATUS_CODES: Final = frozenset({429, 500, 502, 503, 504})
MODEL_COST_MAP_FETCH_MAX_ATTEMPTS: Final = 3
MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS: Final = 30.0
_litellm_import_complete = threading.Event()


def mark_litellm_import_complete() -> None:
    _litellm_import_complete.set()


@dataclass(frozen=True, slots=True)
class ModelCostMapReloaded:
    model_cost_map: dict  # mutable-ok: adopted as litellm.model_cost, whose consumer contract is a plain mutable dict
    revision: str | None = None
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class ModelCostMapReloadUnavailable:
    reason: str


ModelCostMapReloadResult = ModelCostMapReloaded | ModelCostMapReloadUnavailable


class _AsyncGetClient(Protocol):
    def get(self, url: str, *, timeout: float | None = None) -> Awaitable[httpx.Response]: ...


class _SyncGetClient(Protocol):
    def get(self, url: str, *, timeout: float | None = None) -> httpx.Response: ...


def _loaded_or_raise(catalog: LoadedCatalog | CatalogUnusable) -> LoadedCatalog:
    match catalog:
        case CatalogUnusable(reason):
            raise ValueError(reason)
        case LoadedCatalog():
            return catalog


async def refetch_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
    client: "_AsyncGetClient | None" = None,
) -> ModelCostMapReloadResult:
    """Fetch reload data without changing the installed catalog."""
    from litellm._catalog import reload_source
    from litellm._catalog._fetch import CatalogUnavailable, fetch
    from litellm._catalog._install import accepted_catalog

    result: Final = await asyncio.to_thread(fetch, reload_source(url), max_attempts=max_attempts, timeout=timeout)
    if isinstance(result, CatalogUnavailable):
        return ModelCostMapReloadUnavailable(result.reason)
    accepted: Final = accepted_catalog(result)
    if accepted is None:
        return ModelCostMapReloadUnavailable("Model cost map failed integrity validation")
    return ModelCostMapReloaded(
        served_rows(accepted),
        result.registry.source_revision or git_blob_id(ModelRegistry.bundled_bytes()),
        result.registry.source_validator,
    )


class ModelCostMapSourceInfo:
    """Tracks the source of the currently loaded model cost map."""

    source: str = "local"  # "local" or "remote"
    url: str | None = None
    is_env_forced: bool = False
    fallback_reason: str | None = None
    loaded_at: "datetime | None" = None
    source_revision: str | None = None
    etag: str | None = None


# Module-level singleton tracking the source of the current cost map
_cost_map_source_info: Final = ModelCostMapSourceInfo()


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
    return {
        "source_revision": _cost_map_source_info.source_revision,
        "etag": _cost_map_source_info.etag,
    }


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
    loaded_at: Final = _cost_map_source_info.loaded_at
    return {
        "source": _cost_map_source_info.source,
        "url": _cost_map_source_info.url,
        "is_env_forced": _cost_map_source_info.is_env_forced,
        "fallback_reason": _cost_map_source_info.fallback_reason,
        "loaded_at": loaded_at.isoformat() if loaded_at is not None else None,
        "source_revision": _cost_map_source_info.source_revision,
        "etag": _cost_map_source_info.etag,
    }


def get_model_cost_map_loaded_at() -> "datetime | None":
    """When this process last loaded its cost map, stamped at the start of every load"""
    return _cost_map_source_info.loaded_at


class AliasedCostEntry(dict[str, object]):
    __slots__ = ("names", "registered_fields")

    def __init__(
        self, entry: Mapping[str, object], names: tuple[str, ...], registered_fields: frozenset[str] = frozenset()
    ) -> None:
        super().__init__(entry)
        self.names: tuple[str, ...] = names
        self.registered_fields: frozenset[str] = registered_fields


def _alias_groups(alias_owners: Mapping[str, str]) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType(
        {
            owner: (owner, *(alias for alias, alias_owner in alias_owners.items() if alias_owner == owner))
            for owner in dict.fromkeys(alias_owners.values())
        }
    )


def _expand_model_aliases(  # pyright: ignore[reportUnusedFunction]  # compatibility helper imported by callers
    model_cost: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """
    Expand ``aliases`` lists in model cost entries into top-level entries.

    Each alias gets a reference to the **same** ``AliasedCostEntry`` as the
    canonical entry, which carries every name it is served under.  The
    ``aliases`` key is removed from the entry so downstream code never sees it.

    If an alias collides with an existing canonical entry the alias is
    skipped and a warning is logged.
    """
    alias_owners: Final[dict[str, str]] = {}
    keys_with_aliases: Final[list[str]] = []

    for model_name, model_info in model_cost.items():
        aliases: object = model_info.get("aliases")
        if aliases is None:
            continue
        keys_with_aliases.append(model_name)
        if not isinstance(aliases, list):
            verbose_logger.warning(
                "LiteLLM model alias field for '%s' is not a list (got %s) — skipping.",
                model_name,
                type(aliases).__name__,
            )
            continue
        if not aliases:
            continue
        for alias in aliases:
            if alias in model_cost:
                verbose_logger.warning(
                    "LiteLLM model alias conflict: alias '%s' (from '%s') "
                    "already exists as a canonical entry — skipping.",
                    alias,
                    model_name,
                )
                continue
            if alias in alias_owners:
                verbose_logger.warning(
                    "LiteLLM model alias conflict: alias '%s' (from '%s') "
                    "was already claimed by another entry — skipping.",
                    alias,
                    model_name,
                )
                continue
            alias_owners[alias] = model_name

    # Remove the ``aliases`` key from entries so it doesn't pollute model info
    for key in keys_with_aliases:
        model_cost[key].pop("aliases", None)

    for owner, names in _alias_groups(alias_owners).items():
        model_cost.update(dict.fromkeys(names, AliasedCostEntry(model_cost[owner], names)))
    return model_cost


def _finalize_model_cost_map(  # pyright: ignore[reportUnusedFunction]  # imported by autorubric
    model_cost: dict,
) -> dict:
    return _serve_catalog(_loaded_or_raise(load(json.dumps(model_cost).encode())))


def _serve_catalog(catalog: LoadedCatalog) -> dict[str, AliasedCostEntry]:  # mutable-ok: public mutable model_cost
    return served_rows(catalog)


def adopt_model_cost_map(
    new_model_cost_map: dict,  # mutable-ok: public API preserves the mutable cost-map contract
) -> int:
    from litellm._catalog._install import Provenance, install

    catalog: Final = _loaded_or_raise(load(json.dumps(new_model_cost_map).encode()))
    return install(
        catalog,
        provenance=Provenance(
            source="local",
            revision=_cost_map_source_info.source_revision,
            etag=_cost_map_source_info.etag,
        ),
    )


def get_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    client: "_SyncGetClient | None" = None,
) -> dict:
    """Read one explicit source without installing it or scheduling retries.

    A non-empty URL wins over LITELLM_LOCAL_MODEL_COST_MAP. Empty URLs return
    bundled rows. Legacy retry/client arguments remain accepted, but transport
    and settings come from the native catalog fetcher.
    """
    from litellm._catalog._fetch import Bundled, CatalogUnavailable, Remote, fetch
    from litellm._catalog._install import accepted_catalog

    result: Final = fetch(Remote(url) if url else Bundled(), max_attempts=1, timeout=timeout)
    if isinstance(result, CatalogUnavailable):
        verbose_logger.warning("LiteLLM: model cost map fetch failed: %s. Using bundled rows.", result.reason)
        return served_rows(LoadedCatalog(ModelRegistry.bundled()))
    accepted: Final = accepted_catalog(result) if url else result
    return served_rows(accepted if accepted is not None else LoadedCatalog(ModelRegistry.bundled()))
