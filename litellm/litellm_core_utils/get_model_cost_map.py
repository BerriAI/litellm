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

import httpx

from litellm import verbose_logger
from litellm.constants import (
    MODEL_COST_MAP_MAX_SHRINK_PERCENT,
    MODEL_COST_MAP_MIN_MODEL_COUNT,
)


class GetModelCostMap:
    """
    Handles fetching, validating, and loading the model cost map.

    The backup model count is cached on first access so that validation
    never needs to parse the full backup JSON — only the count is needed
    for the shrinkage check.  The full backup is only loaded when it must
    be *returned* as a fallback.
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
        """Return the number of models in the local backup (cached)."""
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
        max_shrink_pct: float = MODEL_COST_MAP_MAX_SHRINK_PERCENT,
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

        if backup_model_count > 0 and fetched_count < backup_model_count * max_shrink_pct:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map shrank significantly "
                "(fetched=%d, backup=%d, threshold=%.0f%%). "
                "This may indicate a corrupted upstream file. "
                "Falling back to local backup.",
                fetched_count,
                backup_model_count,
                max_shrink_pct * 100,
            )
            return False

        return True

    @classmethod
    def validate_model_cost_map(
        cls,
        fetched_map: dict,
        backup_model_count: int,
        min_model_count: int = MODEL_COST_MAP_MIN_MODEL_COUNT,
        max_shrink_pct: float = MODEL_COST_MAP_MAX_SHRINK_PERCENT,
    ) -> bool:
        """
        Validate the integrity of a fetched model cost map.

        Runs each check in order and returns False on the first failure.

        Checks:
        1. ``_check_is_valid_dict`` -- fetched map is a non-empty dict.
        2. ``_check_model_count_not_reduced`` -- model count meets minimum
           and has not shrunk >``max_shrink_pct`` vs backup.

        Returns True if all checks pass, False otherwise.
        """
        if not cls._check_is_valid_dict(fetched_map):
            return False

        if not cls._check_model_count_not_reduced(
            fetched_map=fetched_map,
            backup_model_count=backup_model_count,
            min_model_count=min_model_count,
            max_shrink_pct=max_shrink_pct,
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


def get_model_cost_map(url: str) -> dict:
    """
    Public entry point — returns the model cost map dict.

    1. If ``LITELLM_LOCAL_MODEL_COST_MAP`` is set, uses the local backup only.
    2. Otherwise fetches from ``url``, validates integrity, and falls back
       to the local backup on any failure.

    The backup model count is cached in ``GetModelCostMap`` so validation
    only costs a cheap integer comparison — the full backup JSON is only
    parsed when it needs to be *returned* as a fallback.
    """
    # Note: can't use get_secret_bool here — this runs during litellm.__init__
    # before litellm._key_management_settings is set.
    if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        return GetModelCostMap.load_local_model_cost_map()

    try:
        content = GetModelCostMap.fetch_remote_model_cost_map(url)
    except Exception as e:
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. "
            "Falling back to local backup.",
            url,
            str(e),
        )
        return GetModelCostMap.load_local_model_cost_map()

    # Validate fetched JSON integrity — uses cached backup count, no file I/O
    if not GetModelCostMap.validate_model_cost_map(
        fetched_map=content,
        backup_model_count=GetModelCostMap._get_backup_model_count(),
    ):
        verbose_logger.warning(
            "LiteLLM: Fetched model cost map failed integrity check. "
            "Using local backup instead. url=%s",
            url,
        )
        return GetModelCostMap.load_local_model_cost_map()

    return content
