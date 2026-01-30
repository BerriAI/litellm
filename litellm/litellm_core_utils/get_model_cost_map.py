"""
Pulls the cost + context window + provider route for known models from https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json

This can be disabled by setting the LITELLM_LOCAL_MODEL_COST_MAP environment variable to True.

```
export LITELLM_LOCAL_MODEL_COST_MAP=True
```
"""

import os
import sys
from typing import Optional, Tuple

import httpx


# Minimum number of model entries expected in a valid model cost map
# This prevents accepting an empty or truncated response
MIN_MODEL_ENTRIES = 100

# Sample of required fields that should be present in model entries
# At minimum, every model entry should have a litellm_provider
REQUIRED_MODEL_FIELDS = {"litellm_provider"}

# Valid modes for model entries
VALID_MODES = {
    "audio_speech",
    "audio_transcription",
    "chat",
    "completion",
    "embedding",
    "image_edit",
    "image_generation",
    "moderation",
    "ocr",
    "rerank",
    "responses",
    "search",
    "vector_store",
    "video_generation",
}


def _get_logger():
    """Get the verbose logger if available, otherwise return a no-op logger."""
    try:
        from litellm._logging import verbose_logger
        return verbose_logger
    except ImportError:
        import logging
        return logging.getLogger(__name__)


def validate_model_cost_map(data: dict) -> Tuple[bool, Optional[str]]:
    """
    Validate that fetched data is a valid model cost map.
    
    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, "reason") if invalid
    """
    if not isinstance(data, dict):
        return False, f"Expected dict, got {type(data).__name__}"
    
    # Check minimum number of entries (should have hundreds of models)
    # Exclude sample_spec from count
    model_count = len([k for k in data.keys() if k != "sample_spec"])
    if model_count < MIN_MODEL_ENTRIES:
        return False, f"Too few model entries: {model_count} (expected at least {MIN_MODEL_ENTRIES})"
    
    # Validate structure of model entries (sample check for performance)
    errors = []
    entries_checked = 0
    max_entries_to_check = 50  # Check a sample for performance
    
    for key, value in data.items():
        if key == "sample_spec":
            continue
            
        entries_checked += 1
        if entries_checked > max_entries_to_check:
            break
        
        if not isinstance(value, dict):
            errors.append(f"Entry '{key}' is not a dict")
            continue
        
        # Check required fields
        if "litellm_provider" not in value:
            errors.append(f"Entry '{key}' missing required field 'litellm_provider'")
        
        # Validate mode if present
        if "mode" in value and value["mode"] not in VALID_MODES:
            errors.append(f"Entry '{key}' has invalid mode: '{value['mode']}'")
        
        # Validate cost fields are numeric if present
        for cost_field in ["input_cost_per_token", "output_cost_per_token"]:
            if cost_field in value and not isinstance(value[cost_field], (int, float)):
                errors.append(f"Entry '{key}' has non-numeric {cost_field}: {type(value[cost_field]).__name__}")
    
    if errors:
        # Return first few errors to avoid huge error messages
        error_sample = errors[:5]
        if len(errors) > 5:
            error_sample.append(f"... and {len(errors) - 5} more errors")
        return False, "; ".join(error_sample)
    
    return True, None


def _load_local_model_cost_map() -> dict:
    """Load the local backup model cost map."""
    from importlib.resources import files
    import json

    content = json.loads(
        files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text(encoding="utf-8")
    )
    return content


def get_model_cost_map(url: str) -> dict:
    """
    Get the model cost map, either from remote URL or local backup.
    
    Priority:
    1. If LITELLM_LOCAL_MODEL_COST_MAP=True, use local backup
    2. Try to fetch from remote URL
    3. Validate the remote response before accepting
    4. Fall back to local backup if remote fails or is invalid
    
    Args:
        url: The URL to fetch the model cost map from
        
    Returns:
        dict: The model cost map
    """
    logger = _get_logger()
    
    # Check if local-only mode is enabled
    local_mode = os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() in ("true", "1", "yes")
    
    if local_mode:
        logger.debug("LITELLM_LOCAL_MODEL_COST_MAP is set, using local model cost map")
        return _load_local_model_cost_map()

    # Try to fetch from remote
    try:
        response = httpx.get(
            url, timeout=5
        )  # set a 5 second timeout for the get request
        response.raise_for_status()  # Raise an exception if the request is unsuccessful
        content = response.json()
        
        # Validate the response before accepting
        is_valid, error_message = validate_model_cost_map(content)
        
        if not is_valid:
            logger.warning(
                f"Remote model cost map failed validation: {error_message}. "
                f"Falling back to local backup. "
                f"Set LITELLM_LOCAL_MODEL_COST_MAP=True to always use local backup."
            )
            return _load_local_model_cost_map()
        
        logger.debug(f"Successfully loaded remote model cost map with {len(content)} entries")
        return content
        
    except httpx.TimeoutException:
        logger.warning(
            "Timeout fetching remote model cost map, falling back to local backup. "
            "Set LITELLM_LOCAL_MODEL_COST_MAP=True to always use local backup."
        )
        return _load_local_model_cost_map()
    except httpx.HTTPStatusError as e:
        logger.warning(
            f"HTTP error fetching remote model cost map (status {e.response.status_code}), "
            f"falling back to local backup. "
            f"Set LITELLM_LOCAL_MODEL_COST_MAP=True to always use local backup."
        )
        return _load_local_model_cost_map()
    except Exception as e:
        logger.warning(
            f"Error fetching remote model cost map: {type(e).__name__}: {e}. "
            f"Falling back to local backup. "
            f"Set LITELLM_LOCAL_MODEL_COST_MAP=True to always use local backup."
        )
        return _load_local_model_cost_map()
