"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.
The ``lite`` and ``litellm-proxy`` CLI entry points also use the bundled map without fetching.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import atexit
import os
import sys
import time
import warnings
from collections.abc import Callable, Mapping
from datetime import datetime
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm import verbose_logger
from litellm._catalog import CATALOG, Catalog, Fetched, Installed, Refresher, Remote, Source, Unavailable
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.offline_mode import is_offline_mode

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
_CATALOG_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
_CLI_ENTRYPOINT_NAMES: Final = frozenset({"lite", "litellm-proxy"})
MODEL_COST_MAP_FETCH_MAX_ATTEMPTS: Final = 3
_IMPORT_FETCH_TIMEOUT_SECONDS: Final = 5
REVALIDATION_PENDING: Final = "remote revalidation pending"


def _is_cli_process() -> bool:
    return Path(sys.argv[0]).stem in _CLI_ENTRYPOINT_NAMES


def _local_map_forced() -> bool:
    # Note: can't use get_secret_bool here — this runs during litellm.__init__
    # before litellm._key_management_settings is set.
    return is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"


class GetModelCostMap:
    """The catalog as last loaded, for readers that want it without later registrations"""

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
    provenance: Final = CATALOG.provenance()
    return {
        "source_revision": provenance.source_revision if provenance is not None else None,
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
    - loaded_at: ISO 8601 time this process last loaded the map
    - source_revision: git blob id of the loaded file's bytes
    - etag: the ETag of the remote fetch (None for the bundled backup)
    """
    provenance: Final = CATALOG.provenance()
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
        "loaded_at": provenance.loaded_at.isoformat(),
        "source_revision": provenance.source_revision,
        "etag": provenance.etag,
    }


def get_model_cost_map_loaded_at() -> "datetime | None":
    """When this process last loaded its cost map, stamped at the start of every load"""
    provenance: Final = CATALOG.provenance()
    return provenance.loaded_at if provenance is not None else None


def _note_loaded(fetched: Fetched) -> Fetched:
    set_fallback_generalizations(fetched.fallback_rules())
    GetModelCostMap._snapshot_loaded_catalog(  # pyright: ignore[reportPrivateUsage]  # same module
        fetched.served()
    )
    return fetched


def _install(
    fetched: Fetched,
    catalog: Catalog = CATALOG,
    *,
    env_forced: bool = False,
    fallback_reason: str | None = None,
    url: str | None = None,
) -> Installed:
    installed: Final = catalog.install(fetched, env_forced=env_forced, fallback_reason=fallback_reason, url=url)
    if installed.rejected:
        verbose_logger.warning(
            "LiteLLM: model cost map rows failed validation and are served from the bundled map or left out: %s",
            "; ".join(f"{error.key}: {error.message}" for error in installed.rejected),
        )
    return installed


def _refresh_derived_state() -> None:
    from litellm import utils

    utils._invalidate_model_cost_lowercase_map()  # pyright: ignore[reportPrivateUsage]  # required cache invalidation


def reload_model_cost_map(source: Source, catalog: Catalog = CATALOG) -> Installed | Unavailable:
    """Fetch with main's reload policy, then install and refresh what Python still derives from the map. A failed
    fetch keeps the map in use"""
    fetched: Final = catalog.fetch(
        source, timeout=_IMPORT_FETCH_TIMEOUT_SECONDS, attempts=MODEL_COST_MAP_FETCH_MAX_ATTEMPTS
    )
    if isinstance(fetched, Unavailable):
        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.", fetched.reason
        )
        return fetched
    installed: Final = _install(_note_loaded(fetched), catalog, env_forced=_local_map_forced())
    _refresh_derived_state()
    return installed


def load_bundled_model_cost_map_at_import(catalog: Catalog = CATALOG) -> Installed:
    """Fills ``litellm.model_cost`` at import with the bundled map and no network (decision 4). The refresher fetches
    the remote map after import returns, unless the bundled map is forced"""
    forced: Final = _local_map_forced()
    return _install(
        _note_loaded(catalog.bundled()),
        catalog,
        env_forced=forced,
        fallback_reason=None if forced or _is_cli_process() else REVALIDATION_PENDING,
    )


def start_model_cost_map_refresh(url: str, catalog: Catalog = CATALOG) -> Refresher | None:
    """Starts the background refresh at the end of import, except where main used the bundled map only. The refresher
    restarts in a forked child that has no remote map yet and is stopped at exit"""
    if _local_map_forced() or _is_cli_process() or not url:
        return None
    try:
        refresher: Final = catalog.refresher(Remote(url))
    except ValueError as e:
        verbose_logger.warning("LiteLLM: model cost map %s cannot be refreshed: %s", url, e)
        return None
    os.register_at_fork(after_in_child=refresher.after_fork)
    atexit.register(refresher.stop)
    refresher.start()
    return refresher


def install_refreshed(fetched: Fetched, catalog: Catalog, url: str) -> None:
    """Called by the Rust refresher thread with a fetched map, after import returned"""
    _install(_note_loaded(fetched), catalog, url=url)
    _refresh_derived_state()


def note_refresh_dropped(url: str) -> None:
    verbose_logger.info(
        "LiteLLM: dropped the background model cost map refresh from %s, a newer map was installed meanwhile", url
    )


def note_refresh_failed(url: str, reason: str) -> None:
    verbose_logger.warning(
        "LiteLLM: Failed to fetch remote model cost map from %s: %s. Keeping the bundled map.", url, reason
    )


def fetch_remote_model_cost_map(url: str) -> Fetched | Unavailable:
    """One attempt at ``url`` that installs nothing. Offline mode wins over the URL, so no request leaves the process"""
    if is_offline_mode():
        return Unavailable(reason=f"offline mode, {url} is not fetched")
    return CATALOG.fetch(Remote(url), timeout=_IMPORT_FETCH_TIMEOUT_SECONDS, attempts=1)


def get_model_cost_map(
    url: str,
    timeout: int = _IMPORT_FETCH_TIMEOUT_SECONDS,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: object = None,
    client: object = None,
) -> dict[str, dict[str, JsonValue]]:
    """
    Public entry point, a side read that returns a fresh model cost map dict and changes no global state. An explicit
    URL wins over ``LITELLM_LOCAL_MODEL_COST_MAP``. The fetch makes one attempt whatever ``max_attempts`` says, and
    returns the bundled rows on any failure. ``timeout``, ``max_attempts``, ``sleep``, ``rng`` and ``client`` are
    accepted for compatibility, the fetch runs in Rust
    """
    if rng is not None or client is not None:
        warnings.warn(
            "get_model_cost_map ignores rng and client, the fetch runs in Rust", DeprecationWarning, stacklevel=2
        )
    if not url:
        return CATALOG.bundled().served()
    fetched: Final = fetch_remote_model_cost_map(url)
    if isinstance(fetched, Unavailable):
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. Falling back to local backup.",
            url,
            fetched.reason,
        )
        return CATALOG.bundled().served()
    return fetched.served()
