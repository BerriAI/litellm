"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.
LITELLM_PRICE_DIR must be relative to /app ( root dir of litellm)

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
export LITELLM_PRICE_DIR=config
```
"""
import os
import json
import httpx
import importlib.resources


def get_model_cost_map(url: str) -> dict:
    """
    Retrieves the model cost and context window data from a remote source.
    Falls back to a local JSON file if:
    - The environment variable LITELLM_LOCAL_MODEL_COST_MAP is set to "True", or
    - The remote fetch fails
    """


    def load_local_backup() -> dict:
        """Load model cost data from the local backup JSON file."""
        backup_path = os.path.join(os.getenv("LITELLM_PRICE_DIR", "litellm"), "model_prices_and_context_window_backup.json")
        with open(backup_path, "r") as f:
            return json.load(f)

    if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        return load_local_backup()

    try:
        response = httpx.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        print("[WARN] Remote fetch failed, falling back to local model cost map.")
        return load_local_backup()
