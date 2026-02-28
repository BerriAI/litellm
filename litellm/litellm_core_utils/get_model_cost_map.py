"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import json
import os
import threading
from importlib.resources import files
from typing import Optional

import httpx

from litellm import verbose_logger
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_RATIO,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)


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
        content = json.loads(
            files("litellm")
            .joinpath("model_prices_and_context_window_backup.json")
            .read_text(encoding="utf-8")
        )
        return content

    @classmethod
    def _get_backup_model_count(cls) -> int:
        """Return the number of models in the local backup (cached int)."""
        if cls._backup_model_count < 0:
            backup = cls.load_local_model_cost_map()
            cls._backup_model_count = len(backup)
        return cls._backup_model_count

    @staticmethod
    def _check_is_valid_dict(fetched_map: dict) -> bool:
        """Check 1: fetched map is a non-empty dict."""
        if not isinstance(fetched_map, dict):
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map is not a dict (type=%s). "
                "Falling back to local backup.",
                type(fetched_map).__name__,
            )
            return False

        if len(fetched_map) == 0:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map is empty. "
                "Falling back to local backup.",
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
        fetched_count = len(fetched_map)

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
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()


class ModelCostMapSourceInfo:
    """Tracks the source of the currently loaded model cost map."""

    source: str = "local"  # "local" or "remote"
    url: Optional[str] = None
    is_env_forced: bool = False
    fallback_reason: Optional[str] = None


# Module-level singleton tracking the source of the current cost map
_cost_map_source_info = ModelCostMapSourceInfo()


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
    # Note: can't use get_secret_bool here — this runs during litellm.__init__
    # before litellm._key_management_settings is set.
    if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        _cost_map_source_info.source = "local"
        _cost_map_source_info.url = None
        _cost_map_source_info.is_env_forced = True
        _cost_map_source_info.fallback_reason = None
        return GetModelCostMap.load_local_model_cost_map()

    _cost_map_source_info.url = url
    _cost_map_source_info.is_env_forced = False

    try:
        content = GetModelCostMap.fetch_remote_model_cost_map(url)
    except Exception as e:
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. "
            "Falling back to local backup.",
            url,
            str(e),
        )
        _cost_map_source_info.source = "local"
        _cost_map_source_info.fallback_reason = f"Remote fetch failed: {str(e)}"
        return GetModelCostMap.load_local_model_cost_map()

    # Validate using cached count (cheap int comparison, no file I/O)
    if not GetModelCostMap.validate_model_cost_map(
        fetched_map=content,
        backup_model_count=GetModelCostMap._get_backup_model_count(),
    ):
        verbose_logger.warning(
            "LiteLLM: Fetched model cost map failed integrity check. "
            "Using local backup instead. url=%s",
            url,
        )
        _cost_map_source_info.source = "local"
        _cost_map_source_info.fallback_reason = "Remote data failed integrity validation"
        return GetModelCostMap.load_local_model_cost_map()

    _cost_map_source_info.source = "remote"
    _cost_map_source_info.fallback_reason = None
    return content


class LazyModelCostMap(dict):
    """A dict that loads local model cost data eagerly and defers the remote
    HTTP fetch until the data is first accessed after import.

    At import time the bundled local backup is loaded immediately (~12 ms,
    no network).  The first dict operation after ``litellm`` is fully imported
    triggers ``get_model_cost_map()`` which may fetch from the remote URL,
    then re-populates the known-model sets via ``litellm.add_known_models()``.

    This eliminates the blocking HTTP request during ``import litellm`` while
    preserving the existing remote-fetch-and-validate behaviour for callers
    that actually use the cost map.
    """

    def __init__(self, url: str):
        # Load local backup immediately — fast (~12 ms), no network I/O.
        local_data = GetModelCostMap.load_local_model_cost_map()
        dict.update(self, local_data)
        self._url = url
        self._remote_merged = (
            os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true"
        )
        self._lock = threading.Lock()
        # During import, iteration should use local data without triggering
        # the remote fetch.  Call seal_import_phase() after module init.
        self._import_phase = True

        # Track source info for local-only path
        if self._remote_merged:
            _cost_map_source_info.source = "local"
            _cost_map_source_info.url = None
            _cost_map_source_info.is_env_forced = True
            _cost_map_source_info.fallback_reason = None

    def seal_import_phase(self) -> None:
        """Mark the end of import-time initialisation.

        After this call, any dict access will trigger the deferred remote
        fetch (if not already done and not in local-only mode).
        """
        self._import_phase = False

    # -- lazy remote merge --------------------------------------------------

    def _ensure_remote_merged(self) -> None:
        """Fetch and merge remote cost map on first access (once only)."""
        if self._remote_merged or self._import_phase:
            return
        with self._lock:
            if self._remote_merged:  # double-check after acquiring lock
                return
            self._remote_merged = True

            _cost_map_source_info.url = self._url
            _cost_map_source_info.is_env_forced = False

            try:
                remote = GetModelCostMap.fetch_remote_model_cost_map(self._url)
            except Exception as e:
                verbose_logger.info(
                    "LiteLLM: Deferred remote model cost map fetch failed: %s. "
                    "Continuing with local backup.",
                    str(e),
                )
                _cost_map_source_info.source = "local"
                _cost_map_source_info.fallback_reason = (
                    f"Remote fetch failed: {str(e)}"
                )
                return

            if not GetModelCostMap.validate_model_cost_map(
                fetched_map=remote,
                backup_model_count=dict.__len__(self),
            ):
                verbose_logger.info(
                    "LiteLLM: Deferred remote model cost map failed integrity "
                    "check. Continuing with local backup. url=%s",
                    self._url,
                )
                _cost_map_source_info.source = "local"
                _cost_map_source_info.fallback_reason = (
                    "Remote data failed integrity validation"
                )
                return

            # Replace local data with validated remote data
            dict.clear(self)
            dict.update(self, remote)
            _cost_map_source_info.source = "remote"
            _cost_map_source_info.fallback_reason = None

            # Re-populate provider model sets with remote data
            try:
                import litellm

                litellm.add_known_models(self)
            except Exception:
                pass

    # -- dict method overrides ---------------------------------------------
    # These ensure the remote fetch happens transparently on first access.
    # We use dict.xxx(self, ...) instead of super() to avoid recursion.

    def __getitem__(self, key):
        self._ensure_remote_merged()
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        self._ensure_remote_merged()
        return dict.__contains__(self, key)

    def __iter__(self):
        self._ensure_remote_merged()
        return dict.__iter__(self)

    def __len__(self):
        self._ensure_remote_merged()
        return dict.__len__(self)

    def __bool__(self):
        self._ensure_remote_merged()
        return dict.__len__(self) > 0

    def get(self, key, default=None):
        self._ensure_remote_merged()
        return dict.get(self, key, default)

    def keys(self):
        self._ensure_remote_merged()
        return dict.keys(self)

    def values(self):
        self._ensure_remote_merged()
        return dict.values(self)

    def items(self):
        self._ensure_remote_merged()
        return dict.items(self)

    def __repr__(self):
        self._ensure_remote_merged()
        return dict.__repr__(self)

    def __eq__(self, other):
        self._ensure_remote_merged()
        return dict.__eq__(self, other)

    def copy(self):
        self._ensure_remote_merged()
        return dict.copy(self)

    def pop(self, key, *args):
        self._ensure_remote_merged()
        return dict.pop(self, key, *args)

    def setdefault(self, key, default=None):
        self._ensure_remote_merged()
        return dict.setdefault(self, key, default)

    def update(self, *args, **kwargs):
        self._ensure_remote_merged()
        return dict.update(self, *args, **kwargs)

    def __setitem__(self, key, value):
        self._ensure_remote_merged()
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._ensure_remote_merged()
        dict.__delitem__(self, key)
