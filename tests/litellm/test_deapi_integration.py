"""
Live-call integration tests for the deAPI provider.

These tests make real network calls to https://oai.deapi.ai and are
intentionally placed outside `tests/test_litellm/llms/openai_like/`
(which is mock-only). They are skipped by default and only run when
DEAPI_API_KEY is set in the environment.
"""

import os
import sys

try:
    import pytest
except ImportError:
    pytest = None

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestDeAPIIntegration:
    """Live-call integration tests (skipped unless DEAPI_API_KEY is set)."""

    def test_deapi_embedding_basic(self):
        """End-to-end embedding call against deAPI (requires DEAPI_API_KEY)."""
        if not os.environ.get("DEAPI_API_KEY"):
            if pytest:
                pytest.skip("DEAPI_API_KEY not set")
            return

        response = litellm.embedding(
            model="deapi/Bge_M3_FP16",
            input="hello world",
        )

        assert response is not None
        assert hasattr(response, "data")
        assert len(response.data) > 0
        assert hasattr(response.data[0], "embedding")
        assert isinstance(response.data[0].embedding, list)
        assert len(response.data[0].embedding) > 0

    def test_deapi_image_generation_basic(self):
        """End-to-end image generation call (requires DEAPI_API_KEY)."""
        if not os.environ.get("DEAPI_API_KEY"):
            if pytest:
                pytest.skip("DEAPI_API_KEY not set")
            return

        response = litellm.image_generation(
            model="deapi/Flux1schnell",
            prompt="a small red square",
        )

        assert response is not None
        assert hasattr(response, "data")
        assert len(response.data) > 0
