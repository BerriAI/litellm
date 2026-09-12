"""
TTL-cached gateway catalog metadata.

Providers that serve a model catalog with pricing/capability metadata
implement ``BaseLLMModelInfo.get_models_with_info``. This module fetches
that catalog once per (provider, api_base, api_key) per TTL window and
registers the entries into ``litellm.model_cost`` so wildcard-expanded
models carry real costs, context windows, and modalities.
"""

import hashlib
import time
from typing import Final

from litellm._logging import verbose_logger

CATALOG_TTL_SECONDS: Final = 300

_catalog_cache: dict[tuple[str, str | None, str], tuple[float, dict[str, dict]]] = {}


def _cache_key(provider: str, api_key: str | None, api_base: str | None) -> tuple[str, str | None, str]:
    key_hash: Final = hashlib.sha256((api_key or "").encode()).hexdigest()[:12]
    return (provider, api_base, key_hash)


def optional_float(value: object) -> float | None:
    """Parse a catalog price field (string or number) into a float."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]  # catalog fields arrive as str | float
    except (TypeError, ValueError):
        return None


def get_catalog(provider: str, api_key: str | None, api_base: str | None) -> dict[str, dict] | None:
    """
    Returns {model_id: model_info_dict} for the provider's live catalog,
    or None when the provider does not serve catalog metadata or the
    fetch fails.
    """
    key: Final = _cache_key(provider, api_key, api_base)
    cached: Final = _catalog_cache.get(key)
    if cached is not None and time.time() - cached[0] < CATALOG_TTL_SECONDS:
        return cached[1]

    import litellm
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    try:
        provider_enum: Final = LlmProviders(provider)
    except ValueError:
        return None

    config: Final = ProviderConfigManager.get_provider_model_info(model=None, provider=provider_enum)
    if config is None:
        return None

    try:
        entries: Final = config.get_models_with_info(api_key=api_key, api_base=api_base)
    except Exception as e:
        verbose_logger.warning("gateway catalog fetch failed for %s: %s", provider, e)
        return None

    if entries is None:
        return None

    catalog: Final = {entry["key"].split("/", 1)[1]: entry for entry in entries if entry.get("key")}
    _catalog_cache[key] = (time.time(), catalog)
    return catalog


def register_catalog_into_model_cost(prefix: str, catalog: dict[str, dict]) -> None:
    """
    Register catalog entries into litellm.model_cost under
    ``{prefix}/{model_id}`` so expanded wildcard models resolve real
    pricing and limits. In-memory only.
    """
    import litellm

    for model_id, model_info in catalog.items():
        registered_key: Final = f"{prefix}/{model_id}"
        model_info["key"] = registered_key
        litellm.model_cost[registered_key] = model_info
