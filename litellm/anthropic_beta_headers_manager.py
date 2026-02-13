"""
Centralized manager for Anthropic beta headers across different providers.

This module provides utilities to:
1. Load beta header configuration from JSON (mapping of supported headers per provider)
2. Filter and map beta headers based on provider support
3. Handle provider-specific header name mappings (e.g., advanced-tool-use -> tool-search-tool)

Design:
- JSON config contains mapping of beta headers for each provider
- Keys are input header names, values are provider-specific header names (or null if unsupported)
- Only headers present in mapping keys with non-null values can be forwarded
- This enforces stricter validation than the previous unsupported list approach
"""

import json
import os
from typing import Dict, List, Optional, Set

from litellm.litellm_core_utils.litellm_logging import verbose_logger

# Cache for the loaded configuration
_BETA_HEADERS_CONFIG: Optional[Dict] = None


def _load_beta_headers_config() -> Dict:
    """
    Load the beta headers configuration from JSON file.
    Uses caching to avoid repeated file reads.
    
    Returns:
        Dict containing the beta headers configuration
    """
    global _BETA_HEADERS_CONFIG
    
    if _BETA_HEADERS_CONFIG is not None:
        return _BETA_HEADERS_CONFIG
    
    config_path = os.path.join(
        os.path.dirname(__file__),
        "anthropic_beta_headers_config.json"
    )
    
    try:
        with open(config_path, "r") as f:
            _BETA_HEADERS_CONFIG = json.load(f)
            verbose_logger.debug(f"Loaded beta headers config from {config_path}")
            return _BETA_HEADERS_CONFIG
    except Exception as e:
        verbose_logger.error(f"Failed to load beta headers config: {e}")
        # Return empty config as fallback (empty mappings)
        return {
            "anthropic": {},
            "azure_ai": {},
            "bedrock": {},
            "bedrock_converse": {},
            "vertex_ai": {}
        }


def get_provider_name(provider: str) -> str:
    """
    Resolve provider aliases to canonical provider names.
    
    Args:
        provider: Provider name (may be an alias)
        
    Returns:
        Canonical provider name
    """
    config = _load_beta_headers_config()
    aliases = config.get("provider_aliases", {})
    return aliases.get(provider, provider)


def filter_and_transform_beta_headers(
    beta_headers: List[str],
    provider: str,
) -> List[str]:
    """
    Filter and transform beta headers based on provider's mapping configuration.
    
    This function:
    1. Only allows headers that are present in the provider's mapping keys
    2. Filters out headers with null values (unsupported)
    3. Maps headers to provider-specific names (e.g., advanced-tool-use -> tool-search-tool)
    
    Args:
        beta_headers: List of Anthropic beta header values
        provider: Provider name (e.g., "anthropic", "bedrock", "vertex_ai")
        
    Returns:
        List of filtered and transformed beta headers for the provider
    """
    if not beta_headers:
        return []
    
    config = _load_beta_headers_config()
    provider = get_provider_name(provider)
    
    # Get the header mapping for this provider
    provider_mapping = config.get(provider, {})
    
    filtered_headers: Set[str] = set()
    
    for header in beta_headers:
        header = header.strip()
        
        # Check if header is in the mapping
        if header not in provider_mapping:
            verbose_logger.debug(
                f"Dropping unknown beta header '{header}' for provider '{provider}' (not in mapping)"
            )
            continue
        
        # Get the mapped header value
        mapped_header = provider_mapping[header]
        
        # Skip if header is unsupported (null value)
        if mapped_header is None:
            verbose_logger.debug(
                f"Dropping unsupported beta header '{header}' for provider '{provider}'"
            )
            continue
        
        # Add the mapped header
        filtered_headers.add(mapped_header)
    
    return sorted(list(filtered_headers))


def is_beta_header_supported(
    beta_header: str,
    provider: str,
) -> bool:
    """
    Check if a specific beta header is supported by a provider.
    
    Args:
        beta_header: The Anthropic beta header value
        provider: Provider name
        
    Returns:
        True if the header is in the mapping with a non-null value, False otherwise
    """
    config = _load_beta_headers_config()
    provider = get_provider_name(provider)
    provider_mapping = config.get(provider, {})
    
    # Header is supported if it's in the mapping and has a non-null value
    return beta_header in provider_mapping and provider_mapping[beta_header] is not None


def get_provider_beta_header(
    anthropic_beta_header: str,
    provider: str,
) -> Optional[str]:
    """
    Get the provider-specific beta header name for a given Anthropic beta header.
    
    This function handles header transformations/mappings (e.g., advanced-tool-use -> tool-search-tool).
    
    Args:
        anthropic_beta_header: The Anthropic beta header value
        provider: Provider name
        
    Returns:
        The provider-specific header name if supported, or None if unsupported/unknown
    """
    config = _load_beta_headers_config()
    provider = get_provider_name(provider)
    
    # Get the header mapping for this provider
    provider_mapping = config.get(provider, {})
    
    # Check if header is in the mapping
    if anthropic_beta_header not in provider_mapping:
        return None
    
    # Return the mapped value (could be None if unsupported)
    return provider_mapping[anthropic_beta_header]


def update_headers_with_filtered_beta(
    headers: dict,
    provider: str,
) -> dict:
    """
    Update headers dict by filtering and transforming anthropic-beta header values.
    Modifies the headers dict in place and returns it.
    
    Args:
        headers: Request headers dict (will be modified in place)
        provider: Provider name
        
    Returns:
        Updated headers dict
    """
    existing_beta = headers.get("anthropic-beta")
    if not existing_beta:
        return headers
    
    # Parse existing beta headers
    beta_values = [b.strip() for b in existing_beta.split(",") if b.strip()]
    
    # Filter and transform based on provider
    filtered_beta_values = filter_and_transform_beta_headers(
        beta_headers=beta_values,
        provider=provider,
    )
    
    # Update or remove the header
    if filtered_beta_values:
        headers["anthropic-beta"] = ",".join(filtered_beta_values)
    else:
        # Remove the header if no values remain
        headers.pop("anthropic-beta", None)
    
    return headers


def get_unsupported_headers(provider: str) -> List[str]:
    """
    Get all beta headers that are unsupported by a provider (have null values in mapping).
    
    Args:
        provider: Provider name
        
    Returns:
        List of unsupported Anthropic beta header names
    """
    config = _load_beta_headers_config()
    provider = get_provider_name(provider)
    provider_mapping = config.get(provider, {})
    
    # Return headers with null values
    return [header for header, value in provider_mapping.items() if value is None]
