"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import os

import httpx


def get_model_cost_map(url: str):
    if (
        os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False)
        or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False) == "True"
    ):
        import importlib.resources
        import json

        with importlib.resources.open_text(
            "litellm", "model_prices_and_context_window_backup.json"
        ) as f:
            content = json.load(f)
            return content

    try:
        response = httpx.get(
            url, timeout=5
        )  # set a 5 second timeout for the get request
        response.raise_for_status()  # Raise an exception if the request is unsuccessful
        content = response.json()
        return content
    except Exception:
        import importlib.resources
        import json

        with importlib.resources.open_text(
            "litellm", "model_prices_and_context_window_backup.json"
        ) as f:
            content = json.load(f)
            return content
