"""
OpenRouter In-Region Routing and Data Residency Helpers.

OpenRouter provides dedicated regional endpoints for end-to-end data residency:
- EU: https://eu.openrouter.ai/api/v1
- US: https://us.openrouter.ai/api/v1
- Global default: https://openrouter.ai/api/v1

Docs: https://openrouter.ai/docs/guides/features/in-region-routing
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

import litellm
from litellm.secret_managers.main import get_secret_str

# Normalized region names mapped to their dedicated OpenRouter base URLs
_OPENROUTER_REGIONAL_URLS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "eu": "https://eu.openrouter.ai/api/v1",
        "europe": "https://eu.openrouter.ai/api/v1",
        "us": "https://us.openrouter.ai/api/v1",
        "usa": "https://us.openrouter.ai/api/v1",
        "united states": "https://us.openrouter.ai/api/v1",
    }
)

# OpenRouter regional hostnames mapped to canonical data residency codes ("eu", "us")
_OPENROUTER_REGIONAL_HOSTNAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "eu.openrouter.ai": "eu",
        "us.openrouter.ai": "us",
    }
)

_DEFAULT_OPENROUTER_API_BASE: Final[str] = "https://openrouter.ai/api/v1"


def _extract_region_value(params: Mapping[str, object] | None) -> str | None:
    if params is None:
        return None
    for key in ("region", "region_name", "data_residency"):
        val: object = params.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def resolve_openrouter_api_base(
    api_base: str | None = None,
    region: str | None = None,
    litellm_params: Mapping[str, object] | None = None,
    optional_params: Mapping[str, object] | None = None,
) -> str:
    """
    Resolve the OpenRouter API base URL based on custom api_base, explicit region, or environment.

    Precedence:
    1. If `api_base` is passed and is a custom URL (not the default global openrouter.ai URL), use it.
    2. If `region` is provided (via param, litellm_params, optional_params, or environment), map to the regional endpoint.
    3. Fallback to `api_base or litellm.api_base or OPENROUTER_API_BASE or default global URL`.
    """
    # Check if a custom api_base was provided that is not the default global endpoint
    cleaned_base: Final[str | None] = api_base.rstrip("/").lower() if api_base else None
    is_custom_api_base: Final[bool] = bool(
        cleaned_base
        and cleaned_base not in ("https://openrouter.ai/api/v1", "https://openrouter.ai", "http://openrouter.ai")
    )

    if is_custom_api_base and api_base:
        return api_base

    # Check for region in litellm_params, optional_params, explicit region argument, config, or environment
    region_from_params: Final[str | None] = _extract_region_value(litellm_params)
    region_from_optional: Final[str | None] = _extract_region_value(optional_params)

    litellm_region: Final[object] = getattr(litellm, "openrouter_region", None)
    litellm_data_residency: Final[object] = getattr(litellm, "openrouter_data_residency", None)
    attr_region: Final[str | None] = (
        litellm_region
        if isinstance(litellm_region, str)
        else (litellm_data_residency if isinstance(litellm_data_residency, str) else None)
    )

    explicit_region: Final[str | None] = region.strip() if isinstance(region, str) and region.strip() else None
    resolved_region: Final[str | None] = region_from_params or region_from_optional or explicit_region or attr_region

    if resolved_region is not None:
        normalized_region: Final[str] = resolved_region.strip().lower()
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

    fallback_api_base: Final[str | None] = api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE")
    return fallback_api_base or _DEFAULT_OPENROUTER_API_BASE


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
        parsed_url: Final = urlparse(api_base)
        host: Final[str | None] = parsed_url.hostname
    except (TypeError, ValueError):
        return None

    if host is None:
        return None

    return _OPENROUTER_REGIONAL_HOSTNAMES.get(host.lower())
