"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import json
import os
from pathlib import Path

from importlib.resources import files
from typing import Dict, List, Optional

import httpx

from litellm import verbose_logger
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_RATIO,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)


class GetModelCostMap:
    """
    Handles fetching, validating, and loading the model cost map.

    Only the local model *count* is cached (a single int). The full
    local dict is never held in memory — it is only parsed when it
    needs to be *returned* as a fallback.
    """

    _local_model_count: int = -1  # -1 = not yet loaded

    @staticmethod
    def load_local_model_cost_map() -> dict:
        """Load the local model cost map.

        Tries to load from:
        1. Package resources (production, after pip install)
        2. Project root (development)
        """
        try:
            content = json.loads(
                files("litellm")
                .joinpath("model_prices_and_context_window.json")
                .read_text(encoding="utf-8")
            )
            return content
        except FileNotFoundError:
            pass
        except ModuleNotFoundError:
            verbose_logger.warning(
                "LiteLLM: Could not load model cost map from package resources. "
                "Falling back to project root."
            )

        current_dir = Path(__file__).parent.parent.parent
        model_cost_map_path = current_dir / "model_prices_and_context_window.json"
        with open(model_cost_map_path, "r") as f:
            return json.load(f)

    @classmethod
    def _get_local_model_count(cls) -> int:
        """Return the number of models in the local model cost map (cached int)."""
        if cls._local_model_count < 0:
            local = cls.load_local_model_cost_map()
            cls._local_model_count = len(local)
        return cls._local_model_count

    @staticmethod
    def _check_is_valid_dict(fetched_map: dict) -> bool:
        """Check 1: fetched map is a non-empty dict."""
        if not isinstance(fetched_map, dict):
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map is not a dict (type=%s). "
                "Falling back to local model cost map.",
                type(fetched_map).__name__,
            )
            return False

        if len(fetched_map) == 0:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map is empty. "
                "Falling back to local model cost map.",
            )
            return False

        return True

    @classmethod
    def _check_model_count_not_reduced(
        cls,
        fetched_map: dict,
        local_model_count: int,
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio: float = MODEL_COST_MAP_MAX_SHRINK_RATIO,
    ) -> bool:
        """Check 2: model count has not reduced significantly vs local."""
        fetched_count = len(fetched_map)

        if fetched_count < min_model_count:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map has only %d models (minimum=%d). "
                "This may indicate a corrupted upstream file. "
                "Falling back to local model cost map.",
                fetched_count,
                min_model_count,
            )
            return False

        if (
            local_model_count > 0
            and fetched_count < local_model_count * max_shrink_ratio
        ):
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map shrank significantly "
                "(fetched=%d, local=%d, threshold=%.0f%%). "
                "This may indicate a corrupted upstream file. "
                "Falling back to local model cost map.",
                fetched_count,
                local_model_count,
                max_shrink_ratio * 100,
            )
            return False

        return True

    @classmethod
    def validate_model_cost_map(
        cls,
        fetched_map: dict,
        local_model_count: int,
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_ratio: float = MODEL_COST_MAP_MAX_SHRINK_RATIO,
    ) -> bool:
        """
        Validate the integrity of a fetched model cost map.

        Runs each check in order and returns False on the first failure.

        Checks:
        1. ``_check_is_valid_dict`` -- fetched map is a non-empty dict.
        2. ``_check_model_count_not_reduced`` -- model count meets minimum
           and has not shrunk >``max_shrink_ratio`` vs local.

        Returns True if all checks pass, False otherwise.
        """
        if not cls._check_is_valid_dict(fetched_map):
            return False

        if not cls._check_model_count_not_reduced(
            fetched_map=fetched_map,
            local_model_count=local_model_count,
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


def _expand_model_aliases(model_cost: dict) -> dict:
    """
    Expand ``aliases`` lists in model cost entries into top-level entries.

    Each alias gets a reference to the **same** dict object as the canonical
    entry (zero memory overhead).  The ``aliases`` key is removed from the
    entry so downstream code never sees it.

    If an alias collides with an existing canonical entry the alias is
    skipped and a warning is logged.
    """
    aliases_to_add: Dict[str, dict] = {}
    keys_with_aliases: List[str] = []

    for model_name, model_info in model_cost.items():
        aliases: Optional[list] = model_info.get("aliases")
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


def get_model_cost_map(url: str) -> dict:
    """
    Public entry point — returns the model cost map dict.

    1. If ``LITELLM_LOCAL_MODEL_COST_MAP`` is set, uses the local model cost map only.
    2. Otherwise fetches from ``url``, validates integrity, and falls back
       to the local model cost map on any failure.

    Only the local model count is cached (a single int) for validation.
    The full local dict is only parsed when it must be *returned* as a
    fallback — it is never held in memory long-term.
    """
    # Note: can't use get_secret_bool here — this runs during litellm.__init__
    # before litellm._key_management_settings is set.
    if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        _cost_map_source_info.source = "local"
        _cost_map_source_info.url = None
        _cost_map_source_info.is_env_forced = True
        _cost_map_source_info.fallback_reason = None
        return _expand_model_aliases(GetModelCostMap.load_local_model_cost_map())

    _cost_map_source_info.url = url
    _cost_map_source_info.is_env_forced = False

    try:
        content = GetModelCostMap.fetch_remote_model_cost_map(url)
    except Exception as e:
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. "
            "Falling back to local model cost map.",
            url,
            str(e),
        )
        _cost_map_source_info.source = "local"
        _cost_map_source_info.fallback_reason = f"Remote fetch failed: {str(e)}"
        return _expand_model_aliases(GetModelCostMap.load_local_model_cost_map())

    # Validate using cached count (cheap int comparison, no file I/O)
    if not GetModelCostMap.validate_model_cost_map(
        fetched_map=content,
        local_model_count=GetModelCostMap._get_local_model_count(),
    ):
        verbose_logger.warning(
            "LiteLLM: Fetched model cost map failed integrity check. "
            "Using local model cost map instead. url=%s",
            url,
        )
        _cost_map_source_info.source = "local"
        _cost_map_source_info.fallback_reason = (
            "Remote data failed integrity validation"
        )
        return _expand_model_aliases(GetModelCostMap.load_local_model_cost_map())

    _cost_map_source_info.source = "remote"
    _cost_map_source_info.fallback_reason = None
    return _expand_model_aliases(content)
