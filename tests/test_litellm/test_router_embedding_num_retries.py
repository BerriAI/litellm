"""
Tests that Router.embedding() and Router.aembedding() set num_retries
in kwargs before calling function_with_fallbacks, matching the pattern
used by every other router method (acompletion, acreate_file, etc.).

Without num_retries, the retry loop gets num_retries=0 and fails
immediately with no retries or failover to other deployments.

See: https://github.com/BerriAI/litellm/issues/27363
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm import Router


def _make_router(num_retries: int = 3) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "test-embedding",
                "litellm_params": {
                    "model": "openai/text-embedding-ada-002",
                    "api_key": "fake-key",
                },
            },
        ],
        num_retries=num_retries,
    )


class TestEmbeddingNumRetries:
    @pytest.mark.asyncio
    async def test_aembedding_sets_num_retries(self):
        """aembedding must set num_retries before calling
        async_function_with_fallbacks."""
        router = _make_router(num_retries=3)
        captured_kwargs = {}

        async def capture_kwargs(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch.object(
            router, "async_function_with_fallbacks", side_effect=capture_kwargs
        ):
            await router.aembedding(model="test-embedding", input="hello")

        assert "num_retries" in captured_kwargs
        assert captured_kwargs["num_retries"] == 3

    @pytest.mark.asyncio
    async def test_aembedding_preserves_caller_num_retries(self):
        """If the caller passes num_retries explicitly, aembedding must
        not overwrite it with the router default."""
        router = _make_router(num_retries=3)
        captured_kwargs = {}

        async def capture_kwargs(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch.object(
            router, "async_function_with_fallbacks", side_effect=capture_kwargs
        ):
            await router.aembedding(
                model="test-embedding", input="hello", num_retries=5
            )

        assert captured_kwargs["num_retries"] == 5

    def test_embedding_sets_num_retries(self):
        """Sync embedding must also set num_retries before calling
        function_with_fallbacks."""
        router = _make_router(num_retries=3)
        captured_kwargs = {}

        def capture_kwargs(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch.object(
            router, "function_with_fallbacks", side_effect=capture_kwargs
        ):
            router.embedding(model="test-embedding", input="hello")

        assert "num_retries" in captured_kwargs
        assert captured_kwargs["num_retries"] == 3

    def test_embedding_preserves_caller_num_retries(self):
        """If the caller passes num_retries explicitly, sync embedding must
        not overwrite it."""
        router = _make_router(num_retries=3)
        captured_kwargs = {}

        def capture_kwargs(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with patch.object(
            router, "function_with_fallbacks", side_effect=capture_kwargs
        ):
            router.embedding(model="test-embedding", input="hello", num_retries=7)

        assert captured_kwargs["num_retries"] == 7
