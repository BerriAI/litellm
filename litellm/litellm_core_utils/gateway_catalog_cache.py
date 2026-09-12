"""
TTL-cached gateway catalog metadata.

Providers that serve a model catalog with pricing/capability metadata implement
``BaseLLMModelInfo.get_models_with_info``. This module fetches that catalog once
per (provider, api_base, api_key) per TTL window and registers the entries into
``litellm.model_cost`` so wildcard-expanded models carry real costs, context
windows, and modalities.

Catalog payloads are third-party JSON, so every field is read through the
defensive accessors below rather than indexed directly.
"""

import hashlib
import time
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

from litellm._logging import verbose_logger
from litellm.types.utils import ModelInfoBase

CATALOG_TTL_SECONDS: Final = 300
CatalogEntries: TypeAlias = Mapping[str, ModelInfoBase]

EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})
_EMPTY_SEQUENCE: Final[Sequence[object]] = ()

_CATALOG_CACHE: Final[dict[str, tuple[float, CatalogEntries]]] = {}  # mutable-ok: process-wide TTL cache


def _cache_key(provider: str, api_key: str | None, api_base: str | None) -> str:
    key_hash: Final = hashlib.sha256((api_key or "").encode()).hexdigest()[:12]
    return f"{provider}|{api_base}|{key_hash}"


def optional_float(value: object) -> float | None:
    """Parse a catalog price field (string or number) into a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else EMPTY_MAPPING


def as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return _EMPTY_SEQUENCE
    return value


def float_field(entry: Mapping[str, object], field: str) -> float | None:
    return optional_float(entry.get(field))


def int_field(entry: Mapping[str, object], field: str) -> int | None:
    return optional_int(entry.get(field))


def bool_field(entry: Mapping[str, object], field: str) -> bool | None:
    return optional_bool(entry.get(field))


def per_token(per_million: float | None) -> float | None:
    """Convert a per-million-token price to a per-token price."""
    return None if per_million is None else per_million / 1_000_000


def freeze_catalog(entries: Iterable[tuple[str, ModelInfoBase]]) -> CatalogEntries:
    """Build the immutable ``{bare model id: info}`` catalog the cache holds."""
    return MappingProxyType(dict(entries))


def prefix_model_ids(namespace: str, models: Iterable[str]) -> list[str]:  # mutable-ok: get_models contract
    """Namespace catalog ids with the litellm provider, e.g. ``openrouter/anthropic/claude-4``.

    Gateway catalogs return upstream ids whose first segment is another org or
    provider (``anthropic/claude-opus-4-6``). Wildcard expansion reads such a
    leading segment as the model's own provider prefix and replaces it with the
    public one, which would drop the org and yield an uncallable model. Naming
    each id after the litellm provider up front keeps the upstream id whole.
    """
    prefix: Final = f"{namespace}/"
    return [model if model.startswith(prefix) else prefix + model for model in models]  # mutable-ok: get_models contract


def get_catalog(provider: str, api_key: str | None, api_base: str | None) -> CatalogEntries | None:
    """
    The provider's live catalog keyed by bare model id, or None when the
    provider serves no catalog metadata or the fetch fails.
    """
    key: Final = _cache_key(provider, api_key, api_base)
    cached: Final = _CATALOG_CACHE.get(key)
    if cached is not None and time.time() - cached[0] < CATALOG_TTL_SECONDS:
        return cached[1]

    catalog: Final = _fetch_catalog(provider=provider, api_key=api_key, api_base=api_base)
    if catalog is not None:
        _CATALOG_CACHE[key] = (time.time(), catalog)
    return catalog


def _fetch_catalog(provider: str, api_key: str | None, api_base: str | None) -> CatalogEntries | None:
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
        verbose_logger.warning("gateway catalog fetch failed for provider=%s: %s", provider, e)
        return None
    return entries


def register_catalog_into_model_cost(prefix: str, catalog: CatalogEntries) -> None:
    """
    Register catalog entries into ``litellm.model_cost`` under
    ``{prefix}/{model_id}`` so expanded wildcard models resolve real pricing
    and limits. In-memory only, like the router's deployment registration.
    """
    import litellm

    litellm.model_cost.update(
        MappingProxyType({f"{prefix}/{model_id}": entry for model_id, entry in catalog.items()})
    )
