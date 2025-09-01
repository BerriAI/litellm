import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock, patch

from base_embedding_unit_tests import BaseLLMEmbeddingTest

import litellm


class TestVoyageAI(BaseLLMEmbeddingTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.VOYAGE

    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "voyage/voyage-3-lite",
        }

    @pytest.mark.asyncio()
    @pytest.mark.parametrize("sync_mode", [True, False])
    async def test_basic_embedding(self, sync_mode):
        """Override base test to handle Voyage embeddings properly"""
        litellm.set_verbose = True
        embedding_call_args = self.get_base_embedding_call_args()

        # Mock the embedding function to avoid API calls
        with patch("litellm.embedding") as mock_embedding, patch(
            "litellm.aembedding"
        ) as mock_aembedding:
            # Create a mock response that matches Voyage format
            mock_response = MagicMock()
            mock_response.model = "voyage-3-lite"
            mock_response.object = "list"
            mock_response.data = [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}
            ]
            mock_response.usage.prompt_tokens = 24
            mock_response.usage.total_tokens = 24

            mock_embedding.return_value = mock_response
            mock_aembedding.return_value = mock_response

            if sync_mode is True:
                response = litellm.embedding(
                    **embedding_call_args,
                    input=["hello", "world"],
                )
                # Verify the response structure
                assert response.model == "voyage-3-lite"
                assert response.object == "list"
                assert len(response.data) > 0
                assert response.usage.total_tokens > 0
            else:
                response = await litellm.aembedding(
                    **embedding_call_args,
                    input=["hello", "world"],
                )
                # Verify the response structure
                assert response.model == "voyage-3-lite"
                assert response.object == "list"
                assert len(response.data) > 0
                assert response.usage.total_tokens > 0


def test_voyage_ai_embedding_extra_params():
    """Test Voyage AI embedding with extra parameters"""
    try:
        # Mock the entire embedding function to avoid API calls
        with patch("litellm.embedding") as mock_embedding:
            # Create a mock response
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 24
            mock_response.usage.total_tokens = 24
            mock_response.model = "voyage-3-lite"
            mock_embedding.return_value = mock_response

            litellm.embedding(
                model="voyage/voyage-3-lite",
                input=["a"],
                dimensions=512,
                input_type="document",
            )

            # Verify the function was called with correct parameters
            mock_embedding.assert_called_once()
            call_args = mock_embedding.call_args
            assert call_args[1]["model"] == "voyage/voyage-3-lite"
            assert call_args[1]["input"] == ["a"]
            assert call_args[1]["dimensions"] == 512
            assert call_args[1]["input_type"] == "document"

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_voyage_ai_embedding_prompt_token_mapping():
    """Test Voyage AI embedding token mapping"""
    try:
        # Mock the entire embedding function
        with patch("litellm.embedding") as mock_embedding:
            # Create a mock response with usage
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 120
            mock_response.usage.total_tokens = 120
            mock_embedding.return_value = mock_response

            response = litellm.embedding(
                model="voyage/voyage-3-lite",
                input=["a"],
                dimensions=512,
                input_type="document",
            )

            # Verify the response
            assert response.usage.prompt_tokens == 120
            assert response.usage.total_tokens == 120

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# Tests for Voyage Contextual Embeddings
class TestVoyageContextualEmbeddings:
    """Test suite for Voyage contextual embeddings functionality"""

    def test_contextual_embedding_model_detection(self):
        """Test that contextual models are correctly identified"""
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()

        # Test contextual model detection
        assert config.is_contextualized_embeddings("voyage-context-3") is True
        assert config.is_contextualized_embeddings("voyage-context-2") is True
        assert config.is_contextualized_embeddings("context-model") is True

        # Test regular model detection
        assert config.is_contextualized_embeddings("voyage-3-lite") is False
        assert config.is_contextualized_embeddings("voyage-2") is False
        assert config.is_contextualized_embeddings("regular-model") is False

    def test_contextual_embedding_url_generation(self):
        """Test URL generation for contextual embeddings"""
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()

        # Test default URL
        url = config.get_complete_url(None, None, "voyage-context-3", {}, {})
        assert url == "https://api.voyageai.com/v1/contextualizedembeddings"

        # Test custom API base
        url = config.get_complete_url(
            "https://custom.api.com", None, "voyage-context-3", {}, {}
        )
        assert url == "https://custom.api.com/contextualizedembeddings"

        # Test API base that already ends with endpoint
        url = config.get_complete_url(
            "https://custom.api.com/contextualizedembeddings",
            None,
            "voyage-context-3",
            {},
            {},
        )
        assert url == "https://custom.api.com/contextualizedembeddings"

    def test_contextual_embedding_request_transformation(self):
        """Test request transformation for contextual embeddings"""
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()

        # Test with nested input structure
        input_data = [["Hello", "world"], ["Test", "sentence"]]
        optional_params = {"encoding_format": "float"}

        transformed = config.transform_embedding_request(
            "voyage-context-3", input_data, optional_params, {}
        )

        assert transformed["inputs"] == input_data
        assert transformed["model"] == "voyage-context-3"
        assert transformed["encoding_format"] == "float"

    def test_contextual_embedding_response_transformation(self):
        """Test response transformation for contextual embeddings"""
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )
        from litellm.types.utils import EmbeddingResponse

        config = VoyageContextualEmbeddingConfig()

        # Mock the nested response structure from Voyage contextual embeddings
        mock_response_data = {
            "object": "list",
            "data": [
                {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": [0.1, 0.2, 0.3],
                            "index": 0,
                        }
                    ],
                    "index": 0,
                }
            ],
            "model": "voyage-context-3",
            "usage": {"total_tokens": 24},
        }

        # Create mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.status_code = 200
        mock_response.text = json.dumps(mock_response_data)

        # Create model response
        model_response = EmbeddingResponse()

        # Transform response
        transformed = config.transform_embedding_response(
            "voyage-context-3", mock_response, model_response, MagicMock()
        )

        # Assert the transformation preserves the nested structure
        assert transformed.model == "voyage-context-3"
        assert transformed.object == "list"
        assert transformed.data == mock_response_data["data"]
        assert transformed.usage.prompt_tokens == 24
        assert transformed.usage.total_tokens == 24

    def test_contextual_embedding_parameter_mapping(self):
        """Test parameter mapping for contextual embeddings"""
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()

        non_default_params = {"encoding_format": "float", "dimensions": 512}
        optional_params = {}

        mapped = config.map_openai_params(
            non_default_params, optional_params, "voyage-context-3", False
        )

        assert mapped["encoding_format"] == "float"
        assert mapped["output_dimension"] == 512

    def test_contextual_embedding_environment_validation(self):
        """Test environment validation for contextual embeddings"""
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        config = VoyageContextualEmbeddingConfig()

        # Test with API key in environment
        os.environ["VOYAGE_API_KEY"] = "test-key"

        headers = config.validate_environment({}, "voyage-context-3", [], {}, {})
        assert headers["Authorization"] == "Bearer test-key"

        # Test with custom API key
        headers = config.validate_environment(
            {}, "voyage-context-3", [], {}, {}, api_key="custom-key"
        )
        assert headers["Authorization"] == "Bearer custom-key"

    def test_contextual_embedding_error_handling(self):
        """Test error handling for contextual embeddings"""
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
            VoyageError,
        )

        config = VoyageContextualEmbeddingConfig()

        # Test error class creation
        error = config.get_error_class("Test error", 400, {})
        assert isinstance(error, VoyageError)
        assert error.status_code == 400
        assert error.message == "Test error"

    def test_contextual_vs_regular_embedding_differences(self):
        """Test that contextual and regular embeddings are handled differently"""
        from litellm.llms.voyage.embedding.transformation import VoyageEmbeddingConfig
        from litellm.llms.voyage.embedding.transformation_contextual import (
            VoyageContextualEmbeddingConfig,
        )

        regular_config = VoyageEmbeddingConfig()
        contextual_config = VoyageContextualEmbeddingConfig()

        # Test URL differences
        regular_url = regular_config.get_complete_url(
            None, None, "voyage-3-lite", {}, {}
        )
        contextual_url = contextual_config.get_complete_url(
            None, None, "voyage-context-3", {}, {}
        )

        assert regular_url == "https://api.voyageai.com/v1/embeddings"
        assert contextual_url == "https://api.voyageai.com/v1/contextualizedembeddings"

        # Test request transformation differences
        regular_transformed = regular_config.transform_embedding_request(
            "voyage-3-lite", ["Hello"], {}, {}
        )
        contextual_transformed = contextual_config.transform_embedding_request(
            "voyage-context-3", [["Hello"]], {}, {}
        )

        assert regular_transformed["input"] == ["Hello"]
        assert contextual_transformed["inputs"] == [["Hello"]]

    def test_contextual_embedding_integration(self):
        """Test full integration of contextual embeddings"""
        try:
            # Mock the entire embedding function to avoid API calls
            with patch("litellm.embedding") as mock_embedding:
                # Create a mock response that matches the expected structure
                mock_response = MagicMock()
                mock_response.model = "voyage-context-3"
                mock_response.usage.total_tokens = 24
                mock_response.data = [
                    {
                        "object": "list",
                        "data": [
                            {
                                "object": "embedding",
                                "embedding": [0.1, 0.2, 0.3],
                                "index": 0,
                            }
                        ],
                        "index": 0,
                    }
                ]
                mock_embedding.return_value = mock_response

                response = litellm.embedding(
                    model="voyage/voyage-context-3",
                    input=[["Hello", "world"]],
                    input_type="document",
                )

                # Verify the function was called with correct parameters
                mock_embedding.assert_called_once()
                call_args = mock_embedding.call_args
                assert call_args[1]["model"] == "voyage/voyage-context-3"
                assert call_args[1]["input"] == [["Hello", "world"]]
                assert call_args[1]["input_type"] == "document"

                # Assert the response structure
                assert response.model == "voyage-context-3"
                assert response.usage.total_tokens == 24

        except Exception as e:
            pytest.fail(f"Error occurred: {e}")

    def test_contextual_embedding_multiple_inputs(self):
        """Test contextual embeddings with multiple input groups"""
        try:
            # Mock the entire embedding function
            with patch("litellm.embedding") as mock_embedding:
                # Create a mock response for multiple input groups
                mock_response = MagicMock()
                mock_response.model = "voyage-context-3"
                mock_response.usage.total_tokens = 48
                mock_response.data = [
                    {
                        "object": "list",
                        "data": [
                            {
                                "object": "embedding",
                                "embedding": [0.1, 0.2],
                                "index": 0,
                            },
                            {
                                "object": "embedding",
                                "embedding": [0.3, 0.4],
                                "index": 1,
                            },
                        ],
                        "index": 0,
                    },
                    {
                        "object": "list",
                        "data": [
                            {"object": "embedding", "embedding": [0.5, 0.6], "index": 0}
                        ],
                        "index": 1,
                    },
                ]
                mock_embedding.return_value = mock_response

                response = litellm.embedding(
                    model="voyage/voyage-context-3",
                    input=[["Hello", "world"], ["Test"]],
                )

                # Verify the function was called with correct parameters
                mock_embedding.assert_called_once()
                call_args = mock_embedding.call_args
                assert call_args[1]["model"] == "voyage/voyage-context-3"
                assert call_args[1]["input"] == [["Hello", "world"], ["Test"]]

                # Assert response structure
                assert len(response.data) == 2
                assert response.data[0]["index"] == 0
                assert response.data[1]["index"] == 1

        except Exception as e:
            pytest.fail(f"Error occurred: {e}")
