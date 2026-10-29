"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.
The ``lite`` and ``litellm-proxy`` CLI entry points also use the bundled map without fetching.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import sys
import threading
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
from litellm._catalog import CATALOG, Catalog, Installed, Parsed, Unavailable
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"
_CATALOG_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
_CLI_ENTRYPOINT_NAMES: Final = frozenset({"lite", "litellm-proxy"})
MODEL_COST_MAP_FETCH_MAX_ATTEMPTS: Final = 3
_IMPORT_FETCH_TIMEOUT_SECONDS: Final = 5
_litellm_import_complete = threading.Event()


def _is_cli_process() -> bool:
    return Path(sys.argv[0]).stem in _CLI_ENTRYPOINT_NAMES


def mark_litellm_import_complete() -> None:
    _litellm_import_complete.set()


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


class AliasedCostEntry(dict[str, object]):
    __slots__ = ("names", "registered_fields")

    def __init__(
        self, entry: Mapping[str, object], names: tuple[str, ...], registered_fields: frozenset[str] = frozenset()
    ) -> None:
        super().__init__(entry)
        self.names: tuple[str, ...] = names
        self.registered_fields: frozenset[str] = registered_fields


def _note_loaded(parsed: Parsed) -> Parsed:
    set_fallback_generalizations(parsed.fallback_rules())
    GetModelCostMap._snapshot_loaded_catalog(  # pyright: ignore[reportPrivateUsage]  # same module
        parsed.served()
    )
    return parsed


class _InstallGeneration:
    """Bumped by every install, so a registration racing a reload into the same dict can rebuild its entry"""

    value: int = 0


def model_cost_install_generation() -> int:
    return _InstallGeneration.value


def _install(parsed: Parsed, catalog: Catalog = CATALOG) -> Installed:
    installed: Final = catalog.install(parsed)
    _InstallGeneration.value += 1
    if installed.rejected:
        verbose_logger.warning(
            "LiteLLM: model cost map rows failed validation and are served from the bundled map or left out: %s",
            "; ".join(f"{error.key}: {error.message}" for error in installed.rejected),
        )
    return installed


def _refresh_derived_state() -> None:
    import litellm
    from litellm import utils

    utils._invalidate_model_cost_lowercase_map()  # pyright: ignore[reportPrivateUsage]  # required cache invalidation
    litellm.add_known_models(model_cost_map=litellm.model_cost)
    utils.reapply_runtime_model_cost_registrations()


def reload_model_cost_map(url: str, catalog: Catalog = CATALOG) -> Installed | Unavailable:
    """Load with main's reload policy, then install and refresh what Python still derives from the map. A failed
    load keeps the map in use"""
    loaded: Final = catalog.load(url, timeout=_IMPORT_FETCH_TIMEOUT_SECONDS, attempts=MODEL_COST_MAP_FETCH_MAX_ATTEMPTS)
    if isinstance(loaded, Unavailable):
        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.", loaded.reason
        )
        return loaded
    installed: Final = _install(_note_loaded(loaded), catalog)
    _refresh_derived_state()
    return installed


def _retry_remote_fetch_in_background(
    url: str, max_attempts: int, first_wait: float, sleep: Callable[[float], None], catalog: Catalog
) -> None:
    try:
        sleep(first_wait)
        loaded: Final = catalog.load(url, timeout=_IMPORT_FETCH_TIMEOUT_SECONDS, attempts=max_attempts - 1)
        if isinstance(loaded, Unavailable):
            verbose_logger.warning(
                "LiteLLM: Failed to fetch remote model cost map from %s after %d attempts; keeping local backup",
                url,
                max_attempts,
            )
            return
        _litellm_import_complete.wait()
        _install(_note_loaded(loaded), catalog)
        _refresh_derived_state()
    except Exception as e:  # noqa: BLE001  # a failed background retry must not kill the task; the backup stays
        verbose_logger.warning("LiteLLM: Background model cost map retry failed: %s", e)


def _pick_model_cost_map(
    url: str, timeout: int, max_attempts: int, sleep: Callable[[float], None], catalog: Catalog
) -> Parsed:
    """
    The map main loads at import. The bundled map in a ``lite`` / ``litellm-proxy`` CLI process, and whatever
    ``Catalog.load`` picks otherwise, which is the bundled map when ``LITELLM_LOCAL_MODEL_COST_MAP`` or offline mode
    forces it. A failed remote load serves the bundled map and starts a background retry when the failure was transient
    """
    if _is_cli_process():
        return _note_loaded(catalog.bundled())
    loaded: Final = catalog.load(url, timeout=timeout, attempts=1)
    if isinstance(loaded, Parsed):
        return _note_loaded(loaded)
    if loaded.retry_after is not None and max_attempts > 1:
        threading.Thread(
            target=_retry_remote_fetch_in_background,
            kwargs={
                "url": url,
                "max_attempts": max_attempts,
                "first_wait": loaded.retry_after,
                "sleep": sleep,
                "catalog": catalog,
            },
            name="litellm-model-cost-map-retry",
            daemon=True,
        ).start()
    verbose_logger.warning(
        "LiteLLM: Failed to fetch remote model cost map from %s: %s. Falling back to local backup.",
        url,
        loaded.reason,
    )
    return _note_loaded(loaded.fallback())


def install_initial_model_cost_map(
    url: str,
    catalog: Catalog = CATALOG,
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
) -> Installed:
    """Fills ``litellm.model_cost`` at import with the map main's rules pick"""
    parsed: Final = _pick_model_cost_map(
        url=url,
        timeout=_IMPORT_FETCH_TIMEOUT_SECONDS,
        max_attempts=max_attempts,
        sleep=sleep,
        catalog=catalog,
    )
    return _install(parsed, catalog)


def get_model_cost_map(
    url: str,
    timeout: int = _IMPORT_FETCH_TIMEOUT_SECONDS,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    rng: object = None,
    client: object = None,
) -> dict[str, dict[str, JsonValue]]:
    """
    Public entry point, returns a fresh model cost map dict picked by main's rules. The fetch, its retries and the
    validation of the body run in Rust, and the bundled map is returned on any failure. ``rng`` and ``client`` are
    accepted for compatibility and ignored, since Rust owns the HTTP client and the backoff jitter
    """
    if rng is not None or client is not None:
        warnings.warn(
            "get_model_cost_map ignores rng and client, the fetch runs in Rust", DeprecationWarning, stacklevel=2
        )
    parsed: Final = _pick_model_cost_map(
        url=url, timeout=timeout, max_attempts=max_attempts, sleep=sleep, catalog=CATALOG
    )
    return parsed.served()
