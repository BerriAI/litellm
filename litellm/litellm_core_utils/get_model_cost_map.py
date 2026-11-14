"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.
The ``lite`` and ``litellm-proxy`` CLI entry points also use the bundled map without fetching.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import json
import os
import sys
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
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
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_RATIO,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)
from litellm.litellm_core_utils import model_cost_table
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)
from litellm.litellm_core_utils.model_cost_table import (
    AliasedCostEntry as AliasedCostEntry,  # noqa: PLC0414  # main exported the row type from here
)
from litellm.litellm_core_utils.offline_mode import is_offline_mode
from litellm.rust_bridge._native import (
    NativeCostMapSource,
    NativeFetchFailed,
    NativeFetchPolicy,
    NativeInstalled,
    NativeModelCostMap,
    NativeRejected,
)

_CLI_ENTRYPOINT_NAMES: Final = frozenset({"lite", "litellm-proxy"})
_CATALOG_ADAPTER: Final = TypeAdapter(dict[str, dict[str, object]])
MODEL_COST_MAP_FETCH_MAX_ATTEMPTS: Final = 3
MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS: Final = 30.0
MODEL_COST_MAP_FETCH_MIN_DELAY_SECONDS: Final = 2.0
_litellm_import_complete = threading.Event()

model_cost_inputs: Final = NativeModelCostMap.bundled(
    min_model_count=MODEL_COST_MAP_MIN_MODEL_COUNT, max_shrink_ratio=MODEL_COST_MAP_MAX_SHRINK_RATIO
)


def _is_cli_process() -> bool:
    return Path(sys.argv[0]).stem in _CLI_ENTRYPOINT_NAMES


def _uses_bundled_map_only() -> bool:
    # Can't use get_secret_bool here: this runs during litellm.__init__ before litellm._key_management_settings is set
    return is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"


@dataclass(frozen=True, slots=True)
class _LoadedSnapshot:
    rows: Mapping[str, Mapping[str, object]]
    fields: frozenset[str]


@lru_cache(maxsize=1)
def _loaded_snapshot(revision: str) -> _LoadedSnapshot:
    raw: Final = _CATALOG_ADAPTER.validate_python(model_cost_inputs.loaded())
    return _LoadedSnapshot(
        rows=MappingProxyType({key: MappingProxyType(entry) for key, entry in raw.items()}),
        fields=frozenset(chain.from_iterable(entry.keys() for entry in raw.values())),
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
    def load_local_model_cost_map() -> dict:
        """Load the local backup model cost map bundled with the package."""
        return json.loads(GetModelCostMap.read_local_model_cost_map_bytes())

    @staticmethod
    def loaded_model_cost_map() -> Mapping[str, Mapping[str, object]]:
        """The catalog as last loaded (bundled or remote), untouched by ``register_model`` or router registrations."""
        return _loaded_snapshot(model_cost_inputs.provenance.revision).rows

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
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio: float = MODEL_COST_MAP_MAX_SHRINK_RATIO,
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
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio: float = MODEL_COST_MAP_MAX_SHRINK_RATIO,
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


def fetch_policy(timeout: float, max_attempts: int) -> NativeFetchPolicy:
    return NativeFetchPolicy(
        max_attempts=max_attempts,
        max_wait_seconds=MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS,
        timeout_seconds=timeout,
        min_delay_seconds=MODEL_COST_MAP_FETCH_MIN_DELAY_SECONDS,
    )


def _set_fallback_rules(block: JsonValue) -> None:
    rules: Final = block.get("rules") if isinstance(block, dict) else None
    set_fallback_generalizations(rules if isinstance(rules, list) else None)


def _rows_with_rules_set(installed: NativeInstalled) -> dict:
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


def _rejected_reason(rejected: NativeRejected, url: str) -> str:
    if rejected.integrity:
        return f"model cost map from {url} failed integrity validation"
    return f"invalid JSON from {url}: {rejected.reason}"


def _warn_rejected(rejected: NativeRejected, url: str) -> None:
    verbose_logger.warning(
        "LiteLLM: Fetched model cost map from %s was rejected (%s). Using local backup instead.", url, rejected.reason
    )


def _warn_ignored_fetch_hooks(sleep: object, rng: object, client: object) -> None:
    if (sleep, rng, client) != (None, None, None):
        warnings.warn(
            "The sleep, rng and client arguments of the model cost map fetch are ignored, the fetch runs natively",
            DeprecationWarning,
            stacklevel=3,
        )


async def refetch_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
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

    if _uses_bundled_map_only():
        _mark_rows_with_rules_set("local", None, True, None, datetime.now(timezone.utc))
        return ModelCostMapReloaded(model_count=_install(model_cost_inputs.install_bundled()))

    fetched: Final = await cost_map_source(url, use_http_settings=True).fetch(fetch_policy(timeout, max_attempts))
    if isinstance(fetched, NativeFetchFailed):
        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.", fetched.reason
        )
        return ModelCostMapReloadUnavailable(reason=fetched.reason)
    installed: Final = model_cost_inputs.install(fetched.body, fetched.etag)
    if isinstance(installed, NativeRejected):
        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.", installed.reason
        )
        return ModelCostMapReloadUnavailable(reason=_rejected_reason(installed, url))
    _mark_rows_with_rules_set("remote", url, False, None, datetime.now(timezone.utc))
    return ModelCostMapReloaded(model_count=_install(installed))


def _retry_remote_fetch_in_background(
    url: str, timeout: int, max_attempts: int, first_failure: NativeFetchFailed
) -> None:
    try:
        fetched: Final = cost_map_source(url, use_http_settings=False).fetch_blocking(
            fetch_policy(timeout, max_attempts), first_failure
        )
        if isinstance(fetched, NativeFetchFailed):
            verbose_logger.warning(
                "LiteLLM: Failed to fetch remote model cost map from %s after %d attempts; keeping local backup",
                url,
                max_attempts,
            )
            return
        _litellm_import_complete.wait()
        installed: Final = model_cost_inputs.install(fetched.body, fetched.etag)
        if isinstance(installed, NativeRejected):
            _warn_rejected(installed, url)
            return
        _mark_rows_with_rules_set("remote", url, False, None, datetime.now(timezone.utc))
        _install(installed)
    except Exception as e:  # noqa: BLE001  # a failed background retry must not kill the task; the backup stays
        verbose_logger.warning("LiteLLM: Background model cost map retry failed: %s", e)


def get_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
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
    if _uses_bundled_map_only() or _is_cli_process():
        _mark_rows_with_rules_set("local", None, True, None, started)
        return _rows_with_rules_set(model_cost_inputs.install_bundled())

    fetched: Final = cost_map_source(url, use_http_settings=False).fetch_blocking(fetch_policy(timeout, 1))
    if isinstance(fetched, NativeFetchFailed):
        if fetched.retryable and max_attempts > 1:
            threading.Thread(
                target=_retry_remote_fetch_in_background,
                args=(url, timeout, max_attempts, fetched),
                name="litellm-model-cost-map-retry",
                daemon=True,
            ).start()
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. Falling back to local backup.",
            url,
            fetched.reason,
        )
        _mark_rows_with_rules_set("local", url, False, f"Remote fetch failed: {fetched.reason}", started)
        return _rows_with_rules_set(model_cost_inputs.install_bundled())

    installed: Final = model_cost_inputs.install(fetched.body, fetched.etag)
    if isinstance(installed, NativeRejected):
        _warn_rejected(installed, url)
        fallback_reason: Final = (
            "Remote data failed integrity validation"
            if installed.integrity
            else f"Remote fetch failed: {_rejected_reason(installed, url)}"
        )
        _mark_rows_with_rules_set("local", url, False, fallback_reason, started)
        return _rows_with_rules_set(model_cost_inputs.install_bundled())

    _mark_rows_with_rules_set("remote", url, False, None, started)
    return _rows_with_rules_set(installed)


def fetch_registration_document(url: str) -> dict[str, dict[str, object]] | None:
    """The document behind ``register_model(url)``, fetched in one attempt, or ``None`` when the fetch failed."""
    fetched: Final = cost_map_source(url, use_http_settings=False).fetch_blocking(fetch_policy(5, 1))
    if isinstance(fetched, NativeFetchFailed):
        verbose_logger.warning("LiteLLM: register_model could not fetch %s: %s", url, fetched.reason)
        return None
    try:
        document: Final = _CATALOG_ADAPTER.validate_json(fetched.body)
    except ValueError as e:
        verbose_logger.warning("LiteLLM: register_model got an invalid cost map from %s: %s", url, e)
        return None
    return {key: entry for key, entry in document.items() if key != "fallback_generalizations"}


def _finalize_model_cost_map(  # pyright: ignore[reportUnusedFunction]  # main's private name, downstream code calls it
    model_cost: dict,  # mutable-ok: main's signature, pops the rules block
) -> dict:
    """Compat for main's callers. Use ``model_cost_inputs.install``."""
    match NativeModelCostMap.finalize(json.dumps(model_cost).encode()):
        case NativeRejected(reason=reason):
            raise ValueError(reason)
        case installed:
            model_cost.pop("fallback_generalizations", None)
            return _rows_with_rules_set(installed)


_RESERVED_TOP_LEVEL_KEYS: Final = frozenset({"sample_spec", "fallback_generalizations"})


def _count_model_entries(model_cost: Mapping[str, object]) -> int:
    return sum(1 for key in model_cost if key not in _RESERVED_TOP_LEVEL_KEYS)


@lru_cache(maxsize=1)
def _bundled_model_count() -> int:
    return _count_model_entries(GetModelCostMap.load_local_model_cost_map())
