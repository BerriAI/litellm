"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import json
import os
from importlib.resources import files
from typing import Dict, List

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


def _expand_model_aliases(model_cost: dict) -> dict:
    """
    Expand ``aliases`` lists in model cost entries into top-level entries.

    Each alias gets a reference to the **same** dict object as the canonical
    entry (zero memory overhead).  The ``aliases`` key is removed from the
    entry so downstream code never sees it.

    If an alias collides with an existing canonical entry the alias is
    silently skipped and a warning is logged.
    """
    aliases_to_add: Dict[str, dict] = {}
    keys_with_aliases: List[str] = []

    for model_name, model_info in model_cost.items():
        aliases: Optional[list] = model_info.get("aliases")
        if not aliases:
            continue
        keys_with_aliases.append(model_name)
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
        return _expand_model_aliases(GetModelCostMap.load_local_model_cost_map())

    try:
        content = GetModelCostMap.fetch_remote_model_cost_map(url)
    except Exception as e:
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. "
            "Falling back to local backup.",
            url,
            str(e),
        )
        return _expand_model_aliases(GetModelCostMap.load_local_model_cost_map())

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
        return _expand_model_aliases(GetModelCostMap.load_local_model_cost_map())

    return _expand_model_aliases(content)
