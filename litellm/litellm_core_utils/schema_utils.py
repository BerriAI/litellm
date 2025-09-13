"""
Utility functions for handling JSON schemas and $defs in LiteLLM.
"""

from typing import Dict, Any
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders


def should_preserve_defs(custom_llm_provider: str, model: str) -> bool:
    """
    Determine whether to preserve $defs in JSON schemas based on the provider.
    
    Args:
        custom_llm_provider: The LLM provider (e.g., "anthropic", "bedrock", "vertex_ai")
        model: The model name
        
    Returns:
        bool: True if $defs should be preserved, False if they should be removed/flattened
    """
    if not custom_llm_provider:
        return False
        
    try:
        # Get the provider config to check $defs support
        provider_enum = LlmProviders(custom_llm_provider)
        provider_config = ProviderConfigManager.get_provider_chat_config(
            model=model, provider=provider_enum
        )
        
        if provider_config and hasattr(provider_config, 'supports_defs'):
            return provider_config.supports_defs
            
    except (ValueError, AttributeError):
        # If we can't determine the provider or get the config, default to False
        pass
    
    # Default behavior: preserve $defs for known providers that support them
    if custom_llm_provider in ["anthropic", "openai", "azure"]:
        return True
    
    # Remove $defs for providers that don't support them
    if custom_llm_provider in ["bedrock", "vertex_ai", "vertex_ai_beta"]:
        return False
    
    # Default to False for unknown providers
    return False


def process_schema_defs(
    parameters: Dict[str, Any], 
    custom_llm_provider: str, 
    model: str,
    unpack_defs_func=None
) -> Dict[str, Any]:
    """
    Process $defs in a schema based on provider support.
    
    Args:
        parameters: The schema parameters dictionary
        custom_llm_provider: The LLM provider
        model: The model name
        unpack_defs_func: Function to unpack $defs (if provided)
        
    Returns:
        Dict[str, Any]: The processed parameters
    """
    if should_preserve_defs(custom_llm_provider, model):
        # Preserve $defs for providers that support them
        return parameters
    
    # Remove and flatten $defs for providers that don't support them
    if "$defs" in parameters and unpack_defs_func:
        defs = parameters.pop("$defs", {})
        # Flatten the defs
        for name, value in defs.items():
            unpack_defs_func(value, defs)
        unpack_defs_func(parameters, defs)
    
    return parameters
