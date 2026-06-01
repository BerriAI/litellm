"""
Helpers for resolving OpenAI data-residency (regional processing) from an
api_base URL.

OpenAI enforces hostname-per-region for projects with geography restrictions
enabled and rejects requests sent to the wrong host, so the api_base hostname
is the authoritative signal of which region a request was processed in.
"""

from typing import Dict, Optional
from urllib.parse import urlparse

# Mapping of OpenAI regional hostnames to the corresponding data-residency
# value used by the cost calculator. See
# https://developers.openai.com/api/docs/pricing for the regional-processing
# uplift these hostnames trigger.
_OPENAI_REGIONAL_HOSTS: Dict[str, str] = {
    "eu.api.openai.com": "eu",
    "us.api.openai.com": "us",
}


def infer_openai_data_residency(
    custom_llm_provider: Optional[str], api_base: Optional[str]
) -> Optional[str]:
    """
    Derive the OpenAI data-residency region from an api_base URL.

    Returns ``"eu"`` for the EU regional host, ``"us"`` for the US regional
    host, and ``None`` for the default global host, any non-OpenAI provider,
    or any non-OpenAI URL.
    """
    if custom_llm_provider != "openai" or not api_base:
        return None
    try:
        host = urlparse(api_base).hostname
    except (TypeError, ValueError):
        return None
    if not host:
        return None
    return _OPENAI_REGIONAL_HOSTS.get(host.lower())
