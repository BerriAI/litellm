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

from litellm.constants import MIN_MODEL_COST_MAP_ENTRIES


def _load_local_model_cost_map() -> dict:
    """Load the model cost map from the bundled backup file."""
    return json.loads(
        files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text(encoding="utf-8")
    )


def validate_model_cost_map(data: dict) -> bool:
    """
    Returns True if the model cost map looks structurally sound.

    Checks:
    - data is a dict
    - has more than MIN_MODEL_COST_MAP_ENTRIES entries
    """
    if not isinstance(data, dict):
        return False
    if len(data) < MIN_MODEL_COST_MAP_ENTRIES:
        return False
    return True


def get_model_cost_map(url: str) -> dict:
    if (
        os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False)
        or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False) == "True"
    ):
        return _load_local_model_cost_map()

    try:
        response = httpx.get(
            url, timeout=5
        )  # set a 5 second timeout for the get request
        response.raise_for_status()  # Raise an exception if the request is unsuccessful
        content = response.json()

        if not validate_model_cost_map(content):
            raise ValueError(
                "Remote model cost map failed validation: "
                f"got {type(content).__name__} with {len(content) if isinstance(content, dict) else 'N/A'} entries"
            )

        return content
    except Exception:
        return _load_local_model_cost_map()
