"""
Helpers for resolving OpenAI data-residency (regional processing) from an
api_base URL.

OpenAI enforces hostname-per-region for projects with geography restrictions
enabled and rejects requests sent to the wrong host, so the api_base hostname
is the authoritative signal of which region a request was processed in.
"""

import os
from typing import Final
from urllib.parse import urlparse

import litellm

# Mapping of OpenAI regional hostnames to the corresponding data-residency
# value used by the cost calculator. See
# https://developers.openai.com/api/docs/pricing for the regional-processing
# uplift these hostnames trigger.
_OPENAI_REGIONAL_HOSTS: Final[dict[str, str]] = {
    "eu.api.openai.com": "eu",
    "us.api.openai.com": "us",
    "gov.api.openai.com": "fedramp",
}


def infer_openai_data_residency(custom_llm_provider: str | None, api_base: str | None) -> str | None:
    """
    Derive the OpenAI data-residency region from an api_base URL.

    Returns ``"eu"`` for the EU regional host, ``"us"`` for the US regional
    host, ``"fedramp"`` for ``gov.api.openai.com``, and ``None`` for the
    default global host, any non-OpenAI provider, or any non-OpenAI URL.

    When ``api_base`` is unset, falls back to the same defaults the OpenAI
    handler uses (``litellm.api_base``, ``OPENAI_BASE_URL``, ``OPENAI_API_BASE``)
    so env-configured regional endpoints are billed with their uplift.
    """
    if custom_llm_provider != "openai":
        return None
    resolved_api_base: Final = (
        api_base or litellm.api_base or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    )
    if not resolved_api_base:
        return None
    try:
        host: Final = urlparse(resolved_api_base).hostname
    except (TypeError, ValueError):
        return None
    if not host:
        return None
    return _OPENAI_REGIONAL_HOSTS.get(host.lower())
