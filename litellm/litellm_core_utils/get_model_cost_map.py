"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import os
from pathlib import Path

import httpx


def _load_local_model_cost_map() -> dict:
    """Load model cost map from local filesystem.

    Tries to load from:
    1. Package resources (production, after pip install)
    2. Project root (development)
    """
    import json

    # Try loading from package resources (production)
    try:
        import importlib.resources
        with importlib.resources.open_text(
            "litellm", "model_prices_and_context_window.json"
        ) as f:
            return json.load(f)
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    # Try loading from project root (development)
    try:
        current_dir = Path(__file__).parent.parent.parent
        model_cost_map_path = current_dir / "model_prices_and_context_window.json"
        with open(model_cost_map_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find model_prices_and_context_window.json in package or project root"
        )


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
        return content
    except Exception:
        return _load_local_model_cost_map()
