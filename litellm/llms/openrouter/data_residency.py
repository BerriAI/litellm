"""
OpenRouter In-Region Routing and Data Residency Helpers.

OpenRouter provides dedicated regional endpoints for end-to-end data residency:
- EU: https://eu.openrouter.ai/api/v1
- US: https://us.openrouter.ai/api/v1
- Global default: https://openrouter.ai/api/v1

Docs: https://openrouter.ai/docs/guides/features/in-region-routing
"""

from typing import Final
from urllib.parse import urlparse

import litellm
from litellm.secret_managers.main import get_secret_str

# Normalized region names mapped to their dedicated OpenRouter base URLs
_OPENROUTER_REGIONAL_URLS: Final[dict[str, str]] = {
    "eu": "https://eu.openrouter.ai/api/v1",
    "europe": "https://eu.openrouter.ai/api/v1",
    "us": "https://us.openrouter.ai/api/v1",
    "usa": "https://us.openrouter.ai/api/v1",
    "united states": "https://us.openrouter.ai/api/v1",
}

# OpenRouter regional hostnames mapped to canonical data residency codes ("eu", "us")
_OPENROUTER_REGIONAL_HOSTNAMES: Final[dict[str, str]] = {
    "eu.openrouter.ai": "eu",
    "us.openrouter.ai": "us",
}

_DEFAULT_OPENROUTER_API_BASE: Final[str] = "https://openrouter.ai/api/v1"


def resolve_openrouter_api_base(
    api_base: str | None = None,
    region: str | None = None,
    litellm_params: dict | None = None,
    optional_params: dict | None = None,
) -> str:
    """
    Resolve the OpenRouter API base URL based on custom api_base, explicit region, or environment.

    Precedence:
    1. If `api_base` is passed and is a custom URL (not the default global openrouter.ai URL), use it.
    2. If `region` is provided (via param, litellm_params, optional_params, or environment), map to the regional endpoint.
    3. Fallback to `api_base or litellm.api_base or OPENROUTER_API_BASE or default global URL`.
    """
    # Check if a custom api_base was provided that is not the default global endpoint
    is_custom_api_base = False
    if api_base:
        cleaned_base = api_base.rstrip("/").lower()
        if cleaned_base not in ("https://openrouter.ai/api/v1", "https://openrouter.ai", "http://openrouter.ai"):
            is_custom_api_base = True

    if is_custom_api_base and api_base:
        return api_base

    # Check for region in litellm_params, optional_params, explicit region argument, config, or environment
    resolved_region = None
    if isinstance(litellm_params, dict):
        resolved_region = (
            litellm_params.get("region") or litellm_params.get("region_name") or litellm_params.get("data_residency")
        )
    if not resolved_region and isinstance(optional_params, dict):
        resolved_region = (
            optional_params.pop("region", None)
            or optional_params.pop("region_name", None)
            or optional_params.pop("data_residency", None)
        )
    if not resolved_region:
        resolved_region = (
            region or getattr(litellm, "openrouter_region", None) or getattr(litellm, "openrouter_data_residency", None)
        )

    if resolved_region and isinstance(resolved_region, str):
        normalized_region = resolved_region.strip().lower()
        if normalized_region in ("global", "default"):
            return _DEFAULT_OPENROUTER_API_BASE
        if normalized_region in _OPENROUTER_REGIONAL_URLS:
            return _OPENROUTER_REGIONAL_URLS[normalized_region]
        if normalized_region.startswith(("eu-", "eu_", "europe-", "europe_")):
            return "https://eu.openrouter.ai/api/v1"
        if normalized_region.startswith(("us-", "us_", "usa-", "usa_")):
            return "https://us.openrouter.ai/api/v1"
        raise ValueError(
            f"Unsupported OpenRouter region: '{resolved_region}'. Supported regions are 'eu', 'us', 'global'."
        )

    return api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or _DEFAULT_OPENROUTER_API_BASE


def infer_openrouter_data_residency(
    custom_llm_provider: str | None,
    api_base: str | None,
) -> str | None:
    """
    Derive the OpenRouter data-residency region from an api_base URL.

    Returns "eu" for eu.openrouter.ai, "us" for us.openrouter.ai, and None for
    the default global host or non-OpenRouter URLs.
    """
    if custom_llm_provider != "openrouter" or not api_base:
        return None

    try:
        host: Final = urlparse(api_base).hostname
    except (TypeError, ValueError):
        return None

    if not host:
        return None

    return _OPENROUTER_REGIONAL_HOSTNAMES.get(host.lower())
