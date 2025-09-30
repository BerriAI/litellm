"""Helpers for initialising LiteLLM's cache subsystem."""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import litellm
from dotenv import load_dotenv

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore

from litellm.caching.caching import Cache as LiteLLMCache, LiteLLMCacheType

from .log_utils import truncate_large_value

__all__ = [
    "initialize_litellm_cache",
    "test_litellm_cache",
]

logger = logging.getLogger(__name__)
load_dotenv()


def initialize_litellm_cache() -> None:
    """Configure LiteLLM's cache using Redis if available; fallback to memory."""

    if os.getenv("LITELLM_DISABLE_CACHE", "").lower() in {"1", "true", "yes", "y"}:
        try:
            if hasattr(litellm, "disable_cache"):
                litellm.disable_cache()  # type: ignore[attr-defined]
            setattr(litellm, "cache", None)
        except Exception:  # pragma: no cover - best effort
            pass
        logger.info("LiteLLM caching disabled via LITELLM_DISABLE_CACHE")
        return

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    password = os.getenv("REDIS_PASSWORD")

    if redis is None:
        logger.warning("redis package not installed; using in-memory cache")
        litellm.cache = LiteLLMCache(type=LiteLLMCacheType.LOCAL)
        litellm.enable_cache()
        return

    try:
        client = redis.Redis(
            host=host,
            port=port,
            password=password,
            socket_timeout=2,
            decode_responses=True,
        )
        if not client.ping():
            raise ConnectionError("Redis ping failed")

        keys = client.keys("*")
        if keys:
            logger.debug("Existing Redis keys: %s", truncate_large_value(keys))

        litellm.cache = LiteLLMCache(
            type=LiteLLMCacheType.REDIS,
            host=host,
            port=str(port),
            password=password,
            supported_call_types=["acompletion", "completion"],
            ttl=60 * 60 * 24 * 2,
        )
        litellm.enable_cache()

        try:
            test_key = "litellm_cache_test"
            client.set(test_key, "test_value", ex=60)
            assert client.get(test_key) == "test_value"
            client.delete(test_key)
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning("Redis test write/read failed: %s", exc)
    except Exception as exc:
        logger.warning("Redis cache unavailable (%s); falling back to memory", exc)
        litellm.cache = LiteLLMCache(type=LiteLLMCacheType.LOCAL)
        litellm.enable_cache()


def test_litellm_cache() -> Tuple[bool, Dict[str, Optional[bool]]]:
    """Perform a simple cache warm-up to verify the configuration."""

    initialize_litellm_cache()
    from litellm import completion

    _ = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "ping"}])
    cache_hit = getattr(litellm, "cache", None) is not None
    return cache_hit, {"cache_configured": cache_hit}
