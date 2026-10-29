"""
Storage pricing for Gemini / Vertex AI explicit context caches.

Google bills a cachedContents object twice: once for the cached tokens when it is created, and then for storage,
per cached token per hour, until it expires. LiteLLM creates the cache on the caller's behalf from a client
``cache_control`` marker, takes the TTL from the client, and never deletes the cache, so the storage Google bills is
``cached tokens x lifetime hours x storage rate`` where the lifetime is the cache's ``expireTime - createTime`` (the
requested TTL, or Google's 1 hour default).

* ``cache_storage_cost_per_token_per_hour`` in the model cost map is the storage rate.
* ``gemini_context_cache_max_ttl_seconds`` (litellm setting, default 3600) caps the TTL a client may request; a
  larger value is lowered to the cap before the cache is created.
* A model without a storage rate gets no explicit cache (the request is sent uncached), so storage is never bought
  without a price.
"""

from datetime import datetime
from typing import Final

import litellm

CACHE_STORAGE_PRICE_KEY: Final = "cache_storage_cost_per_token_per_hour"
GOOGLE_DEFAULT_CACHE_TTL_SECONDS: Final = 3600.0
DEFAULT_MAX_CACHE_TTL_SECONDS: Final = 3600.0


def max_cache_ttl_seconds() -> float:
    configured: Final[object] = getattr(litellm, "gemini_context_cache_max_ttl_seconds", None)
    if isinstance(configured, (int, float)) and not isinstance(configured, bool) and configured > 0:
        return float(configured)
    return DEFAULT_MAX_CACHE_TTL_SECONDS


def cap_cache_ttl(ttl: str | None) -> str | None:
    """Lower a normalized ``"<seconds>s"`` TTL to the configured maximum. ``None`` keeps Google's 1 hour default
    unless the cap is shorter than that."""
    cap: Final = max_cache_ttl_seconds()
    seconds: Final = float(ttl[:-1]) if ttl else GOOGLE_DEFAULT_CACHE_TTL_SECONDS
    if seconds <= cap:
        return ttl
    return f"{cap:.9f}".rstrip("0").rstrip(".") + "s"


def cache_storage_price(model: str, custom_llm_provider: str) -> float | None:
    """Storage rate (USD per cached token per hour) from the model cost map, or None when the model has none."""
    provider_prefixed: Final = f"{custom_llm_provider}/{model}"
    candidates: Final = (provider_prefixed, model) if custom_llm_provider == "gemini" else (model, provider_prefixed)
    for key in candidates:
        entry = litellm.model_cost.get(key)
        if isinstance(entry, dict):
            price = entry.get(CACHE_STORAGE_PRICE_KEY)
            if isinstance(price, (int, float)) and not isinstance(price, bool) and price >= 0:
                return float(price)
    return None


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def cache_lifetime_hours(created: dict, requested_ttl: str | None) -> float:
    """Lifetime Google will bill storage for: the created cache's expireTime - createTime, else the requested TTL,
    else Google's 1 hour default."""
    create_time: Final = _parse_time(created.get("createTime"))
    expire_time: Final = _parse_time(created.get("expireTime"))
    if create_time is not None and expire_time is not None and expire_time > create_time:
        return (expire_time - create_time).total_seconds() / 3600.0
    if requested_ttl:
        try:
            return float(requested_ttl[:-1]) / 3600.0
        except ValueError:
            pass
    return GOOGLE_DEFAULT_CACHE_TTL_SECONDS / 3600.0


def record_cache_storage(
    logging_obj: object, created: dict, requested_ttl: str | None, model: str, custom_llm_provider: str
) -> None:
    """Store the storage cost of a cache this request created on the logging object; the cost calculator adds it."""
    tokens: Final = (created.get("usageMetadata") or {}).get("totalTokenCount") or 0
    price: Final = cache_storage_price(model, custom_llm_provider) or 0.0
    hours: Final = cache_lifetime_hours(created, requested_ttl)
    cost: Final = max(tokens, 0) * hours * price
    logging_obj.vertex_ai_cache_storage_token_hours = max(tokens, 0) * hours  # type: ignore[attr-defined]  # rebind-ok
    logging_obj.vertex_ai_cache_storage_cost = cost  # type: ignore[attr-defined]  # rebind-ok


def recorded_cache_storage_cost(logging_obj: object | None) -> float:
    recorded: Final[object] = getattr(logging_obj, "vertex_ai_cache_storage_cost", 0.0) if logging_obj else 0.0
    return float(recorded) if isinstance(recorded, (int, float)) and recorded > 0 else 0.0
