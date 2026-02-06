"""
Centralized manager for Anthropic beta headers across different providers.

This module provides utilities to:
1. Load beta header configuration from JSON (lists unsupported headers per provider)
2. Filter out unsupported beta headers
3. Handle provider-specific header name mappings (e.g., advanced-tool-use -> tool-search-tool)

Design:
- JSON config lists UNSUPPORTED headers for each provider
- Headers not in the unsupported list are passed through
- Header mappings allow renaming headers for specific providers
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
        # Return empty config as fallback
        return {
            "anthropic": [],
            "azure_ai": [],
            "bedrock": [],
            "bedrock_converse": [],
            "vertex_ai": []
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
    Filter beta headers based on provider's unsupported list.
    
    This function:
    1. Removes headers that are in the provider's unsupported list
    2. Passes through all other headers as-is
    
    Note: Header transformations/mappings (e.g., advanced-tool-use -> tool-search-tool)
    are handled in each provider's transformation code, not here.
    
    Args:
        beta_headers: List of Anthropic beta header values
        provider: Provider name (e.g., "anthropic", "bedrock", "vertex_ai")
        
    Returns:
        List of filtered beta headers for the provider
    """
    if not beta_headers:
        return []
    
    config = _load_beta_headers_config()
    provider = get_provider_name(provider)
    
    # Get unsupported headers for this provider
    unsupported_headers = set(config.get(provider, []))
    
    filtered_headers: Set[str] = set()
    
    for header in beta_headers:
        header = header.strip()
        
        # Skip if header is unsupported
        if header in unsupported_headers:
            verbose_logger.debug(
                f"Dropping unsupported beta header '{header}' for provider '{provider}'"
            )
            continue
        
        # Pass through as-is
        filtered_headers.add(header)
    
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
        True if the header is supported (not in unsupported list), False otherwise
    """
    config = _load_beta_headers_config()
    provider = get_provider_name(provider)
    unsupported_headers = set(config.get(provider, []))
    return beta_header not in unsupported_headers


def get_provider_beta_header(
    anthropic_beta_header: str,
    provider: str,
) -> Optional[str]:
    """
    Check if a beta header is supported by a provider.
    
    Note: This does NOT handle header transformations/mappings.
    Those are handled in each provider's transformation code.
    
    Args:
        anthropic_beta_header: The Anthropic beta header value
        provider: Provider name
        
    Returns:
        The original header if supported, or None if unsupported
    """
    config = _load_beta_headers_config()
    provider = get_provider_name(provider)
    
    # Check if unsupported
    unsupported_headers = set(config.get(provider, []))
    if anthropic_beta_header in unsupported_headers:
        return None
    
    return anthropic_beta_header


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
    Get all beta headers that are unsupported by a provider.
    
    Args:
        provider: Provider name
        
    Returns:
        List of unsupported Anthropic beta header names
    """
    config = _load_beta_headers_config()
    provider = get_provider_name(provider)
    return config.get(provider, [])
