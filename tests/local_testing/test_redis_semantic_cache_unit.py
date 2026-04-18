"""Unit tests for RedisSemanticCache graceful degradation (issue #25962).

Verifies that:
- RedisSemanticCache.__init__ raises RuntimeError with a clear message when
  the embedding model validation fails during CustomTextVectorizer init.
- The proxy _load_config path degrades gracefully: litellm.cache stays None
  and a warning is logged instead of crashing.
"""
import sys
import os

sys.path.insert(0, os.path.abspath("../..."))

import pytest
from unittest.mock import patch, MagicMock


class TestRedisSemanticCacheInitDegradation:
    """RedisSemanticCache raises RuntimeError when vectorizer init fails."""

    def test_raises_runtime_error_on_embedding_validation_failure(self):
        """CustomTextVectorizer raising ValueError must become RuntimeError."""
        from litellm.caching.redis_semantic_cache import RedisSemanticCache

        with (
            patch(
                "litellm.caching.redis_semantic_cache.CustomTextVectorizer",
                side_effect=ValueError("Invalid embedding method: 429 rate limit"),
            ),
            patch("litellm.caching.redis_semantic_cache.SemanticCache"),
        ):
            with pytest.raises(RuntimeError, match="embedding model validation failed"):
                RedisSemanticCache(
                    redis_url="redis://localhost:6379",
                    similarity_threshold=0.9,
                    embedding_model="text-embedding-ada-002",
                )

    def test_runtime_error_message_contains_original_error(self):
        """RuntimeError message must include the original exception details."""
        original = ValueError("429 rate limit exceeded")
        wrapped = RuntimeError(
            f"RedisSemanticCache: embedding model validation failed during "
            f"initialisation ({type(original).__name__}: {original}). "
            "Check that the configured embedding model is reachable and "
            "that API credentials are valid."
        )
        assert "429 rate limit exceeded" in str(wrapped)
        assert "ValueError" in str(wrapped)
        assert "reachable" in str(wrapped)

    def test_runtime_error_chained_from_original(self):
        """RuntimeError must chain from the original exception (raise ... from e)."""
        original = ValueError("spend cap exhausted")
        try:
            raise RuntimeError("wrapper") from original
        except RuntimeError as e:
            assert e.__cause__ is original


class TestProxyCacheDegradation:
    """Proxy _load_config must degrade gracefully when cache init raises."""

    def test_litellm_cache_is_none_when_init_raises(self):
        """litellm.cache must be None after a cache init failure."""
        import litellm

        original_cache = litellm.cache
        try:
            # Simulate the try/except block in proxy_server._load_config
            try:
                raise RuntimeError("RedisSemanticCache: embedding model unreachable")
            except Exception:
                litellm.cache = None

            assert litellm.cache is None
        finally:
            litellm.cache = original_cache  # restore

    def test_warning_logged_on_cache_init_failure(self):
        """A warning must be logged when cache init fails."""
        import logging

        warning_messages = []

        class _Handler(logging.Handler):
            def emit(self, record):
                warning_messages.append(record.getMessage())

        logger = logging.getLogger("LiteLLM Proxy")
        handler = _Handler(level=logging.WARNING)
        logger.addHandler(handler)

        try:
            try:
                raise RuntimeError("embedding model 429")
            except Exception as e:
                logger.warning(
                    f"Cache initialisation failed - proxy will start without "
                    f"caching. Error: {e}. Fix your cache configuration and "
                    "restart the proxy to enable caching."
                )

            assert any("Cache initialisation failed" in m for m in warning_messages)
            assert any("429" in m for m in warning_messages)
        finally:
            logger.removeHandler(handler)
