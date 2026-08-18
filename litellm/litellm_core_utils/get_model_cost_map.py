"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import asyncio
import json
import os
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from typing import Final, Protocol

import httpx

from litellm import verbose_logger
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_RATIO,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)
from litellm.litellm_core_utils.fallback_generalizations import (
    set_fallback_generalizations,
)

FALLBACK_GENERALIZATIONS_KEY: Final = "fallback_generalizations"

# Reserved top-level keys that are not model entries. They must be excluded
# from the model-count integrity check so a real upstream shrink can't be masked.
RESERVED_TOP_LEVEL_KEYS: Final = frozenset({"sample_spec", FALLBACK_GENERALIZATIONS_KEY})


def _count_model_entries(model_cost: dict) -> int:
    """Count actual model entries, excluding reserved meta keys."""
    return sum(1 for key in model_cost if key not in RESERVED_TOP_LEVEL_KEYS)


class GetModelCostMap:
    """
    Handles fetching, validating, and loading the model cost map.

    Only the backup model *count* is cached (a single int). The full
    backup dict is never held in memory — it is only parsed when it
    needs to be *returned* as a fallback.
    """

    _backup_model_count: int = -1  # -1 = not yet loaded

    @staticmethod
    def load_local_model_cost_map() -> dict:
        """Load the local backup model cost map bundled with the package."""
        content: Final = json.loads(
            files("litellm").joinpath("model_prices_and_context_window_backup.json").read_text(encoding="utf-8")
        )
        return content

    @classmethod
    def _get_backup_model_count(cls) -> int:
        """Return the number of models in the local backup (cached int)."""
        if cls._backup_model_count < 0:
            backup: Final = cls.load_local_model_cost_map()
            cls._backup_model_count = _count_model_entries(backup)
        return cls._backup_model_count

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

    @staticmethod
    def fetch_remote_model_cost_map(url: str, timeout: int = 5) -> dict:
        """
        Fetch the model cost map from a remote URL.

        Returns the parsed JSON dict. Raises on network/parse errors
        (caller is expected to handle).
        """
        response: Final = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()


RETRYABLE_FETCH_STATUS_CODES: Final = frozenset({429, 500, 502, 503, 504})
MODEL_COST_MAP_FETCH_MAX_ATTEMPTS: Final = 3
MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS: Final = 30.0


@dataclass(frozen=True, slots=True)
class ModelCostMapReloaded:
    model_cost_map: dict  # mutable-ok: adopted as litellm.model_cost, whose consumer contract is a plain mutable dict


@dataclass(frozen=True, slots=True)
class ModelCostMapReloadUnavailable:
    reason: str


ModelCostMapReloadResult = ModelCostMapReloaded | ModelCostMapReloadUnavailable


@dataclass(frozen=True, slots=True)
class _FetchAttemptRetryable:
    reason: str
    retry_after_seconds: float | None


def _parse_retry_after_seconds(response: httpx.Response) -> float | None:
    header: Final = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        seconds: Final = float(header)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _retry_wait_seconds(outcome: _FetchAttemptRetryable, attempt: int, rng: random.Random) -> float:
    if outcome.retry_after_seconds is not None:
        return min(outcome.retry_after_seconds, MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS)
    return min(float(2**attempt) + rng.uniform(0.0, 1.0), MODEL_COST_MAP_FETCH_MAX_WAIT_SECONDS)


class _AsyncGetClient(Protocol):
    def get(self, url: str, *, timeout: float | None = None) -> Awaitable[httpx.Response]: ...


def _default_reload_client() -> _AsyncGetClient:
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
    from litellm.types.llms.custom_http import httpxSpecialProvider

    return get_async_httpx_client(llm_provider=httpxSpecialProvider.ModelCostMap)


async def _attempt_fetch(
    client: _AsyncGetClient, url: str, timeout: int
) -> ModelCostMapReloaded | ModelCostMapReloadUnavailable | _FetchAttemptRetryable:
    try:
        response: Final = await client.get(url, timeout=timeout)
    except httpx.HTTPError as e:
        return _FetchAttemptRetryable(reason=f"{type(e).__name__} fetching {url}: {e}", retry_after_seconds=None)
    if response.status_code in RETRYABLE_FETCH_STATUS_CODES:
        return _FetchAttemptRetryable(
            reason=f"HTTP {response.status_code} from {url}",
            retry_after_seconds=_parse_retry_after_seconds(response),
        )
    if response.is_error:
        return ModelCostMapReloadUnavailable(reason=f"HTTP {response.status_code} from {url}")
    try:
        parsed: Final = response.json()
    except ValueError as e:
        return ModelCostMapReloadUnavailable(reason=f"invalid JSON from {url}: {e}")
    if not isinstance(parsed, dict):
        return ModelCostMapReloadUnavailable(reason=f"expected a JSON object from {url}, got {type(parsed).__name__}")
    return ModelCostMapReloaded(model_cost_map=parsed)


async def _fetch_remote_model_cost_map_with_retry(
    url: str,
    timeout: int,
    max_attempts: int,
    sleep: Callable[[float], Awaitable[None]],
    rng: random.Random,
    client: _AsyncGetClient,
) -> ModelCostMapReloadResult:
    for attempt in range(1, max_attempts + 1):
        outcome = await _attempt_fetch(client=client, url=url, timeout=timeout)
        if not isinstance(outcome, _FetchAttemptRetryable):
            return outcome
        if attempt == max_attempts:
            return ModelCostMapReloadUnavailable(reason=f"{outcome.reason} (after {max_attempts} attempts)")
        wait_seconds = _retry_wait_seconds(outcome=outcome, attempt=attempt, rng=rng)
        verbose_logger.warning(
            "LiteLLM: model cost map fetch attempt %d/%d failed (%s); retrying in %.1fs",
            attempt,
            max_attempts,
            outcome.reason,
            wait_seconds,
        )
        await sleep(wait_seconds)
    return ModelCostMapReloadUnavailable(reason="model cost map fetch failed")


async def refetch_model_cost_map(
    url: str,
    timeout: int = 5,
    max_attempts: int = MODEL_COST_MAP_FETCH_MAX_ATTEMPTS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
    client: "_AsyncGetClient | None" = None,
) -> ModelCostMapReloadResult:
    """
    Re-fetch the model cost map for a runtime reload, retrying transient HTTP
    errors (429/5xx/transport) with Retry-After-aware backoff.

    Unlike ``get_model_cost_map`` this never falls back to the packaged backup:
    on failure it returns ``ModelCostMapReloadUnavailable`` so callers keep the
    map they already have.
    """
    if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        _cost_map_source_info.source = "local"
        _cost_map_source_info.url = None
        _cost_map_source_info.is_env_forced = True
        _cost_map_source_info.fallback_reason = None
        return ModelCostMapReloaded(
            model_cost_map=_finalize_model_cost_map(GetModelCostMap.load_local_model_cost_map())
        )

    result: Final = await _fetch_remote_model_cost_map_with_retry(
        url=url,
        timeout=timeout,
        max_attempts=max_attempts,
        sleep=sleep,
        rng=rng if rng is not None else random.Random(),
        client=client if client is not None else _default_reload_client(),
    )
    if isinstance(result, ModelCostMapReloadUnavailable):
        verbose_logger.warning(
            "LiteLLM: model cost map reload failed: %s. Keeping the currently loaded map.",
            result.reason,
        )
        return result
    if not GetModelCostMap.validate_model_cost_map(
        fetched_map=result.model_cost_map,
        backup_model_count=GetModelCostMap._get_backup_model_count(),
    ):
        return ModelCostMapReloadUnavailable(reason=f"model cost map from {url} failed integrity validation")
    _cost_map_source_info.source = "remote"
    _cost_map_source_info.url = url
    _cost_map_source_info.is_env_forced = False
    _cost_map_source_info.fallback_reason = None
    return ModelCostMapReloaded(model_cost_map=_finalize_model_cost_map(result.model_cost_map))


class ModelCostMapSourceInfo:
    """Tracks the source of the currently loaded model cost map."""

    source: str = "local"  # "local" or "remote"
    url: str | None = None
    is_env_forced: bool = False
    fallback_reason: str | None = None
    loaded_at: "datetime | None" = None


# Module-level singleton tracking the source of the current cost map
_cost_map_source_info: Final = ModelCostMapSourceInfo()


def get_model_cost_map_source_info() -> dict:
    """
    Return metadata about where the current model cost map was loaded from.

    Returns a dict with:
    - source: "local" or "remote"
    - url: the remote URL attempted (or None for local-only)
    - is_env_forced: True if LITELLM_LOCAL_MODEL_COST_MAP=True forced local usage
    - fallback_reason: human-readable reason if remote failed and local was used
    """
    return {
        "source": _cost_map_source_info.source,
        "url": _cost_map_source_info.url,
        "is_env_forced": _cost_map_source_info.is_env_forced,
        "fallback_reason": _cost_map_source_info.fallback_reason,
    }


def get_model_cost_map_loaded_at() -> "datetime | None":
    """When this process last loaded its cost map, stamped at the start of every load"""
    return _cost_map_source_info.loaded_at


def _expand_model_aliases(model_cost: dict) -> dict:
    """
    Expand ``aliases`` lists in model cost entries into top-level entries.

    Each alias gets a reference to the **same** dict object as the canonical
    entry (zero memory overhead).  The ``aliases`` key is removed from the
    entry so downstream code never sees it.

    If an alias collides with an existing canonical entry the alias is
    skipped and a warning is logged.
    """
    aliases_to_add: Final[dict[str, dict]] = {}
    keys_with_aliases: Final[list[str]] = []

    for model_name, model_info in model_cost.items():
        aliases: list | None = model_info.get("aliases")
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
            if alias in aliases_to_add:
                verbose_logger.warning(
                    "LiteLLM model alias conflict: alias '%s' (from '%s') "
                    "was already claimed by another entry — skipping.",
                    alias,
                    model_name,
                )
                continue
            aliases_to_add[alias] = model_info  # same dict reference

    # Remove the ``aliases`` key from entries so it doesn't pollute model info
    for key in keys_with_aliases:
        model_cost[key].pop("aliases", None)

    model_cost.update(aliases_to_add)
    return model_cost


def _finalize_model_cost_map(model_cost: dict) -> dict:
    """Extract fallback generalizations out of the raw map, then expand aliases.

    The ``fallback_generalizations`` block is installed into the generalizations
    module and removed from the map so it is never treated as a model entry.
    """
    raw: Final = model_cost.pop(FALLBACK_GENERALIZATIONS_KEY, None)
    rules: Final = raw.get("rules") if isinstance(raw, dict) else None
    set_fallback_generalizations(rules)
    return _expand_model_aliases(model_cost)


def get_model_cost_map(url: str) -> dict:
    """
    Public entry point — returns the model cost map dict.

    1. If ``LITELLM_LOCAL_MODEL_COST_MAP`` is set, uses the local backup only.
    2. Otherwise fetches from ``url``, validates integrity, and falls back
       to the local backup on any failure.

    Only the backup model count is cached (a single int) for validation.
    The full backup dict is only parsed when it must be *returned* as a
    fallback — it is never held in memory long-term.
    """
    _cost_map_source_info.loaded_at = datetime.now(timezone.utc)
    # Note: can't use get_secret_bool here — this runs during litellm.__init__
    # before litellm._key_management_settings is set.
    if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        _cost_map_source_info.source = "local"
        _cost_map_source_info.url = None
        _cost_map_source_info.is_env_forced = True
        _cost_map_source_info.fallback_reason = None
        return _finalize_model_cost_map(GetModelCostMap.load_local_model_cost_map())

    _cost_map_source_info.url = url
    _cost_map_source_info.is_env_forced = False

    try:
        content: Final = GetModelCostMap.fetch_remote_model_cost_map(url)
    except Exception as e:
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. Falling back to local backup.",
            url,
            str(e),
        )
        _cost_map_source_info.source = "local"
        _cost_map_source_info.fallback_reason = f"Remote fetch failed: {e}"
        return _finalize_model_cost_map(GetModelCostMap.load_local_model_cost_map())

    # Validate using cached count (cheap int comparison, no file I/O)
    if not GetModelCostMap.validate_model_cost_map(
        fetched_map=content,
        backup_model_count=GetModelCostMap._get_backup_model_count(),
    ):
        verbose_logger.warning(
            "LiteLLM: Fetched model cost map failed integrity check. Using local backup instead. url=%s",
            url,
        )
        _cost_map_source_info.source = "local"
        _cost_map_source_info.fallback_reason = "Remote data failed integrity validation"
        return _finalize_model_cost_map(GetModelCostMap.load_local_model_cost_map())

    _cost_map_source_info.source = "remote"
    _cost_map_source_info.fallback_reason = None
    return _finalize_model_cost_map(content)
