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


class GetModelCostMap:
    """
    Handles fetching, validating, and loading the model cost map.

    All methods are static — no instance state is needed. This class groups
    the helpers that support `get_model_cost_map()` into a single namespace.
    """

    @staticmethod
    def load_local_model_cost_map() -> dict:
        """Load the local backup model cost map bundled with the package."""
        content = json.loads(
            files("litellm")
            .joinpath("model_prices_and_context_window_backup.json")
            .read_text(encoding="utf-8")
        )
        return content

    @staticmethod
    def validate_model_cost_map(
        fetched_map: dict,
        backup_map: dict,
        min_model_count: int = 10,
        max_shrink_pct: float = 0.5,
    ) -> bool:
        """
        Validate the integrity of a fetched model cost map.

        Checks:
        1. The fetched map is a non-empty dict.
        2. It has at least ``min_model_count`` models.
        3. It did not shrink by more than ``max_shrink_pct`` (50%) compared to the backup.

        Returns True if the fetched map looks valid, False otherwise.
        """
        if not isinstance(fetched_map, dict):
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map is not a dict (type=%s). "
                "Falling back to local backup.",
                type(fetched_map).__name__,
            )
            return False

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

        backup_count = len(backup_map) if isinstance(backup_map, dict) else 0
        if backup_count > 0 and fetched_count < backup_count * max_shrink_pct:
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map shrank significantly "
                "(fetched=%d, backup=%d, threshold=%.0f%%). "
                "This may indicate a corrupted upstream file. "
                "Falling back to local backup.",
                fetched_count,
                backup_count,
                max_shrink_pct * 100,
            )
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
    """
    if (
        os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False)
        or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False) == "True"
    ):
        return GetModelCostMap.load_local_model_cost_map()

    backup_map = GetModelCostMap.load_local_model_cost_map()

    try:
        content = GetModelCostMap.fetch_remote_model_cost_map(url)

        # Validate fetched JSON integrity before using it
        if not GetModelCostMap.validate_model_cost_map(
            fetched_map=content, backup_map=backup_map
        ):
            verbose_logger.warning(
                "LiteLLM: Fetched model cost map failed integrity check. "
                "Using local backup instead. url=%s",
                url,
            )
            return backup_map

        return content
    except Exception as e:
        verbose_logger.warning(
            "LiteLLM: Failed to fetch remote model cost map from %s: %s. "
            "Falling back to local backup.",
            url,
            str(e),
        )
        return backup_map
