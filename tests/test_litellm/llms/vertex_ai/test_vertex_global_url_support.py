"""
Comprehensive tests for Vertex AI global URL support across all endpoints.

This test suite ensures that all Vertex AI endpoints properly handle the 'global' location,
which uses a different URL format than regional endpoints.

Regional: https://{region}-aiplatform.googleapis.com/...
Global: https://aiplatform.googleapis.com/...
"""

from unittest.mock import patch

import pytest

from litellm.llms.vertex_ai.common_utils import (
    _get_embedding_url,
    _get_vertex_url,
    get_vertex_base_url,
)


class TestVertexBaseURL:
    """Test the centralized get_vertex_base_url helper function."""

    @pytest.mark.parametrize(
        "vertex_location, expected_base_url",
        [
            ("us-central1", "https://us-central1-aiplatform.googleapis.com"),
            ("us-east1", "https://us-east1-aiplatform.googleapis.com"),
            ("europe-west1", "https://europe-west1-aiplatform.googleapis.com"),
            ("asia-northeast1", "https://asia-northeast1-aiplatform.googleapis.com"),
            ("global", "https://aiplatform.googleapis.com"),
        ],
    )
    def test_get_vertex_base_url(self, vertex_location, expected_base_url):
        """Test that get_vertex_base_url returns correct URL for all location types."""
        result = get_vertex_base_url(vertex_location)
        assert result == expected_base_url
        assert not result.endswith("/")  # No trailing slash


class TestChatCompletionURLs:
    """Test chat/completion endpoint URL construction with global location."""

    @pytest.mark.parametrize(
        "vertex_location, stream, expected_url_pattern",
        [
            # Regional, non-streaming
            (
                "us-central1",
                False,
                "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent",
            ),
            # Regional, streaming
            (
                "us-central1",
                True,
                "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:streamGenerateContent?alt=sse",
            ),
            # Global, non-streaming
            (
                "global",
                False,
                "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/gemini-1.5-pro:generateContent",
            ),
            # Global, streaming
            (
                "global",
                True,
                "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/gemini-1.5-pro:streamGenerateContent?alt=sse",
            ),
        ],
    )
    def test_chat_url_construction(
        self, vertex_location, stream, expected_url_pattern
    ):
        """Test that chat URLs are correctly constructed for regional and global locations."""
        with patch(
            "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
            side_effect=lambda model: model,
        ):
            url, endpoint = _get_vertex_url(
                mode="chat",
                model="gemini-1.5-pro",
                stream=stream,
                vertex_project="test-project",
                vertex_location=vertex_location,
                vertex_api_version="v1",
            )

        assert url == expected_url_pattern
        if stream:
            assert endpoint == "streamGenerateContent"
            assert "?alt=sse" in url
        else:
            assert endpoint == "generateContent"
            assert "?alt=sse" not in url

    @pytest.mark.parametrize(
        "vertex_location, stream",
        [
            ("us-central1", False),
            ("us-central1", True),
            ("global", False),
            ("global", True),
        ],
    )
    def test_finetuned_model_url_construction(self, vertex_location, stream):
        """Test that fine-tuned models (numeric IDs) use endpoints/ path correctly."""
        with patch(
            "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
            side_effect=lambda model: model,
        ):
            url, endpoint = _get_vertex_url(
                mode="chat",
                model="1234567890",  # Numeric model ID
                stream=stream,
                vertex_project="test-project",
                vertex_location=vertex_location,
                vertex_api_version="v1",
            )

        # Should use endpoints/ path instead of publishers/google/models/
        assert "/endpoints/1234567890:" in url
        assert "/publishers/google/models/" not in url

        # Check base URL is correct
        if vertex_location == "global":
            assert url.startswith("https://aiplatform.googleapis.com")
        else:
            assert url.startswith(f"https://{vertex_location}-aiplatform.googleapis.com")


class TestEmbeddingURLs:
    """Test embedding endpoint URL construction with global location."""

    @pytest.mark.parametrize(
        "vertex_location, model, expected_url_pattern",
        [
            # Regional, regular model
            (
                "us-central1",
                "text-embedding-004",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/text-embedding-004:predict",
            ),
            # Global, regular model
            (
                "global",
                "text-embedding-004",
                "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/text-embedding-004:predict",
            ),
            # Regional, numeric endpoint
            (
                "us-central1",
                "1234567890",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/endpoints/1234567890:predict",
            ),
            # Global, numeric endpoint
            (
                "global",
                "1234567890",
                "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/endpoints/1234567890:predict",
            ),
        ],
    )
    def test_embedding_url_construction(
        self, vertex_location, model, expected_url_pattern
    ):
        """Test that embedding URLs are correctly constructed for regional and global locations."""
        url, endpoint = _get_embedding_url(
            model=model,
            vertex_project="test-project",
            vertex_location=vertex_location,
            vertex_api_version="v1",
        )

        assert url == expected_url_pattern
        assert endpoint == "predict"

        # Verify base URL format
        if vertex_location == "global":
            assert url.startswith("https://aiplatform.googleapis.com")
            assert "-aiplatform.googleapis.com" not in url
        else:
            assert url.startswith(f"https://{vertex_location}-aiplatform.googleapis.com")

    @pytest.mark.parametrize(
        "vertex_location",
        ["us-central1", "europe-west1", "global"],
    )
    def test_embedding_url_with_routing_prefix(self, vertex_location):
        """Test that routing prefixes (bge/, gemma/, etc.) are stripped from URLs."""
        url, endpoint = _get_embedding_url(
            model="bge/1234567890",  # Model with routing prefix
            vertex_project="test-project",
            vertex_location=vertex_location,
            vertex_api_version="v1",
        )

        # Routing prefix should be stripped
        assert "bge/" not in url
        assert "/endpoints/1234567890:" in url


class TestCountTokensURLs:
    """Test count_tokens endpoint URL construction with global location."""

    @pytest.mark.parametrize(
        "vertex_location, expected_url_pattern",
        [
            (
                "us-central1",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:countTokens",
            ),
            (
                "global",
                "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/gemini-1.5-pro:countTokens",
            ),
        ],
    )
    def test_count_tokens_url_construction(self, vertex_location, expected_url_pattern):
        """Test that count_tokens URLs are correctly constructed for regional and global locations."""
        with patch(
            "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
            side_effect=lambda model: model,
        ):
            url, endpoint = _get_vertex_url(
                mode="count_tokens",
                model="gemini-1.5-pro",
                stream=None,
                vertex_project="test-project",
                vertex_location=vertex_location,
                vertex_api_version="v1",
            )

        assert url == expected_url_pattern
        assert endpoint == "countTokens"


class TestImageGenerationURLs:
    """Test image_generation endpoint URL construction with global location."""

    @pytest.mark.parametrize(
        "vertex_location, model, expected_url_pattern",
        [
            # Regional, regular model
            (
                "us-central1",
                "imagen-3.0-generate-001",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/imagen-3.0-generate-001:predict",
            ),
            # Global, regular model
            (
                "global",
                "imagen-3.0-generate-001",
                "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/imagen-3.0-generate-001:predict",
            ),
            # Regional, numeric endpoint
            (
                "us-central1",
                "9876543210",
                "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/endpoints/9876543210:predict",
            ),
            # Global, numeric endpoint
            (
                "global",
                "9876543210",
                "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/endpoints/9876543210:predict",
            ),
        ],
    )
    def test_image_generation_url_construction(
        self, vertex_location, model, expected_url_pattern
    ):
        """Test that image_generation URLs are correctly constructed for regional and global locations."""
        with patch(
            "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
            side_effect=lambda model: model,
        ):
            url, endpoint = _get_vertex_url(
                mode="image_generation",
                model=model,
                stream=None,
                vertex_project="test-project",
                vertex_location=vertex_location,
                vertex_api_version="v1",
            )

        assert url == expected_url_pattern
        assert endpoint == "predict"


class TestAPIVersions:
    """Test that both v1 and v1beta1 API versions work with global location."""

    @pytest.mark.parametrize(
        "api_version, vertex_location",
        [
            ("v1", "us-central1"),
            ("v1", "global"),
            ("v1beta1", "us-central1"),
            ("v1beta1", "global"),
        ],
    )
    def test_api_versions_in_urls(self, api_version, vertex_location):
        """Test that API version is correctly included in URLs for all locations."""
        with patch(
            "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
            side_effect=lambda model: model,
        ):
            url, _ = _get_vertex_url(
                mode="chat",
                model="gemini-1.5-pro",
                stream=False,
                vertex_project="test-project",
                vertex_location=vertex_location,
                vertex_api_version=api_version,
            )

        # API version should be in the URL
        assert f"/{api_version}/" in url


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_global_location_no_region_prefix(self):
        """Ensure global URLs never have a region prefix."""
        base_url = get_vertex_base_url("global")
        assert base_url == "https://aiplatform.googleapis.com"
        assert "global-aiplatform" not in base_url
        assert "-aiplatform.googleapis.com" not in base_url

    @pytest.mark.parametrize(
        "mode",
        ["chat", "embedding", "count_tokens", "image_generation"],
    )
    def test_all_modes_support_global(self, mode):
        """Test that all URL modes support global location."""
        with patch(
            "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
            side_effect=lambda model: model,
        ):
            if mode == "embedding":
                url, _ = _get_embedding_url(
                    model="text-embedding-004",
                    vertex_project="test-project",
                    vertex_location="global",
                    vertex_api_version="v1",
                )
            else:
                url, _ = _get_vertex_url(
                    mode=mode,
                    model="gemini-1.5-pro",
                    stream=False,
                    vertex_project="test-project",
                    vertex_location="global",
                    vertex_api_version="v1",
                )

        # All URLs should use global format
        assert url.startswith("https://aiplatform.googleapis.com")
        assert "/locations/global/" in url

    def test_location_in_path_matches_parameter(self):
        """Ensure the location in the URL path matches the vertex_location parameter."""
        test_locations = ["us-central1", "europe-west1", "global"]

        for location in test_locations:
            with patch(
                "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
                side_effect=lambda model: model,
            ):
                url, _ = _get_vertex_url(
                    mode="chat",
                    model="gemini-1.5-pro",
                    stream=False,
                    vertex_project="test-project",
                    vertex_location=location,
                    vertex_api_version="v1",
                )

            # Location should appear in the path
            assert f"/locations/{location}/" in url


class TestBackwardCompatibility:
    """Ensure changes don't break existing functionality."""

    def test_regional_urls_unchanged(self):
        """Test that regional URL construction hasn't changed."""
        with patch(
            "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
            side_effect=lambda model: model,
        ):
            url, _ = _get_vertex_url(
                mode="chat",
                model="gemini-1.5-pro",
                stream=False,
                vertex_project="my-project",
                vertex_location="us-central1",
                vertex_api_version="v1",
            )

        # Should match the traditional regional format
        assert (
            url
            == "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent"
        )

    def test_streaming_urls_unchanged(self):
        """Test that streaming URL construction hasn't changed."""
        with patch(
            "litellm.VertexGeminiConfig.get_model_for_vertex_ai_url",
            side_effect=lambda model: model,
        ):
            url, _ = _get_vertex_url(
                mode="chat",
                model="gemini-1.5-pro",
                stream=True,
                vertex_project="my-project",
                vertex_location="us-central1",
                vertex_api_version="v1",
            )

        # Should include streaming endpoint and alt=sse
        assert ":streamGenerateContent?alt=sse" in url

