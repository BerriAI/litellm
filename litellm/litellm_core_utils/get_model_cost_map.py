"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import os

import httpx


def get_model_cost_map(url: str) -> dict:
    if (
        os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False)
        or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", False) == "True"
    ):
        from importlib.resources import files
        import json

        content = json.loads(
            files("litellm")
            .joinpath("model_prices_and_context_window_backup.json")
            .read_text(encoding="utf-8")
        )
        return content

    try:
        response = httpx.get(
            url, timeout=5
        )  # set a 5 second timeout for the get request
        response.raise_for_status()  # Raise an exception if the request is unsuccessful
        content = response.json()
        return content
    except Exception:
        from importlib.resources import files
        import json

        content = json.loads(
            files("litellm")
            .joinpath("model_prices_and_context_window_backup.json")
            .read_text(encoding="utf-8")
        )
        return content
