"""
Helpers for resolving OpenAI data residency from an api_base URL.

OpenAI uses dedicated hosts for regional and FedRAMP requests, so the api_base
hostname determines which data-residency uplift applies.
"""

from typing import Final
from urllib.parse import urlparse

# Mapping of OpenAI data-residency hostnames to the cost-calculator values. See
# https://developers.openai.com/api/docs/pricing for the regional-processing
# uplift these hostnames trigger.
_OPENAI_DATA_RESIDENCY_HOSTS: Final[dict[str, str]] = {
    "eu.api.openai.com": "eu",
    "us.api.openai.com": "us",
    "gov.api.openai.com": "fedramp",
}


def infer_openai_data_residency(custom_llm_provider: str | None, api_base: str | None) -> str | None:
    """
    Derive the OpenAI data-residency mode from an api_base URL.

    Returns ``"eu"`` for the EU regional host, ``"us"`` for the US regional
    host, ``"fedramp"`` for ``gov.api.openai.com``, and ``None`` for the
    default global host, any non-OpenAI provider, or any non-OpenAI URL.
    """
    if custom_llm_provider != "openai" or not api_base:
        return None
    try:
        host: Final = urlparse(api_base).hostname
    except (TypeError, ValueError):
        return None
    if not host:
        return None
    return _OPENAI_DATA_RESIDENCY_HOSTS.get(host.lower())
