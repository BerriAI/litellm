"""
ASI Common Utilities Module

This module provides common utilities for the ASI provider integration.
"""

from typing import Optional, Dict, Any, List
import re

def is_asi_model(model: str) -> bool:
    """
    Check if a model is an ASI model.

    Args:
        model: The model name to check

    Returns:
        True if the model is an ASI model, False otherwise
    """
    # Check if the model starts with "asi" or "asi/"
    if model.startswith("asi/") or model.startswith("asi-") or model == "asi":
        return True
    
    # Check for specific ASI model names
    asi_models = ["asi1-mini"]
    
    # Remove any provider prefix (e.g., "asi/")
    clean_model = model.split("/")[-1] if "/" in model else model
    
    return clean_model in asi_models

def get_asi_model_name(model: str) -> str:
    """
    Get the ASI model name without any provider prefix.

    Args:
        model: The model name with potential provider prefix

    Returns:
        The ASI model name without provider prefix
    """
    # Remove any provider prefix (e.g., "asi/")
    if model.startswith("asi/"):
        return model[4:]
    
    return model

def validate_environment(api_key: Optional[str] = None) -> None:
    """
    Validate that the necessary environment variables are set for ASI.

    Args:
        api_key: Optional API key to check

    Raises:
        ValueError: If the API key is not provided and not set in environment variables
    """
    if api_key is None:
        from litellm.utils import get_secret
        
        api_key_value = get_secret("ASI_API_KEY")
        
        # Ensure api_key is either a string or None
        if isinstance(api_key_value, str):
            api_key = api_key_value
        elif api_key_value is not None and not isinstance(api_key_value, bool):
            # Try to convert to string if possible
            try:
                api_key = str(api_key_value)
            except:
                api_key = None
        
        if api_key is None:
            raise ValueError(
                "ASI API key not provided. Please provide an API key via the api_key parameter or set the ASI_API_KEY environment variable."
            )
