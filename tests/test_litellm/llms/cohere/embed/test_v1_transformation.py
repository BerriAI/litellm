import os
import sys
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.cohere.embed.v1_transformation import CohereEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


class TestCohereEmbeddingV1Transform:
    def setup_method(self):
        self.config = CohereEmbeddingConfig()
        self.model = "embed-english-v3.0"
        self.logging_obj = MagicMock()
        self.encoding = MagicMock()
        # Mock the encoding to return a fixed token count
        self.encoding.encode = MagicMock(return_value=[1, 2, 3, 4, 5])

    def test_transform_response_regular_embeddings(self):
        """Test that regular embeddings are correctly transformed"""
        # Mock httpx.Response
        mock_response = MagicMock()
        response_json = {
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ],
            "meta": {
                "billed_units": {
                    "input_tokens": 10
                }
            }
        }
        mock_response.json = MagicMock(return_value=response_json)

        input_data = ["test text 1", "test text 2"]
        data = {"texts": input_data, "input_type": "search_query"}
        model_response = EmbeddingResponse()

        result = self.config._transform_response(
            response=mock_response,
            api_key="test-api-key",
            logging_obj=self.logging_obj,
            data=data,
            model_response=model_response,
            model=self.model,
            encoding=self.encoding,
            input=input_data,
        )

        # Verify the response structure
        assert result.object == "list"
        assert result.model == self.model
        assert len(result.data) == 2
        
        # Verify each embedding object
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert "type" not in result.data[0]
        
        assert result.data[1]["object"] == "embedding"
        assert result.data[1]["index"] == 1
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
        assert "type" not in result.data[1]

        # Verify usage
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 10
        assert result.usage.completion_tokens == 0

    def test_transform_response_embeddings_by_type(self):
        """Test that embeddings_by_type are correctly transformed"""
        # Mock httpx.Response
        mock_response = MagicMock()
        response_json = {
            "response_type": "embeddings_by_type",
            "embeddings": {
                "float": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                "int8": [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
            },
            "meta": {
                "billed_units": {
                    "input_tokens": 10
                }
            }
        }
        mock_response.json = MagicMock(return_value=response_json)

        input_data = ["test text 1", "test text 2"]
        data = {"texts": input_data, "input_type": "search_query", "embedding_types": ["float", "int8"]}
        model_response = EmbeddingResponse()

        result = self.config._transform_response(
            response=mock_response,
            api_key="test-api-key",
            logging_obj=self.logging_obj,
            data=data,
            model_response=model_response,
            model=self.model,
            encoding=self.encoding,
            input=input_data,
        )

        # Verify the response structure
        assert result.object == "list"
        assert result.model == self.model
        assert len(result.data) == 4  # 2 texts * 2 embedding types
        
        # Verify float embeddings
        assert result.data[0]["object"] == "embedding"
        assert result.data[0]["index"] == 0
        assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert result.data[0]["type"] == "float"
        
        assert result.data[1]["object"] == "embedding"
        assert result.data[1]["index"] == 1
        assert result.data[1]["embedding"] == [0.4, 0.5, 0.6]
        assert result.data[1]["type"] == "float"

        # Verify int8 embeddings
        assert result.data[2]["object"] == "embedding"
        assert result.data[2]["index"] == 0
        assert result.data[2]["embedding"] == [1, 2, 3]
        assert result.data[2]["type"] == "int8"
        
        assert result.data[3]["object"] == "embedding"
        assert result.data[3]["index"] == 1
        assert result.data[3]["embedding"] == [4, 5, 6]
        assert result.data[3]["type"] == "int8"

        # Verify usage
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 10
        assert result.usage.completion_tokens == 0

    def test_transform_response_with_image_tokens(self):
        """Test that image token billing is correctly handled"""
        # Mock httpx.Response
        mock_response = MagicMock()
        response_json = {
            "embeddings": [
                [0.1, 0.2, 0.3],
            ],
            "meta": {
                "billed_units": {
                    "input_tokens": 5,
                    "images": 100
                }
            }
        }
        mock_response.json = MagicMock(return_value=response_json)

        input_data = ["test image"]
        data = {"images": input_data, "input_type": "image"}
        model_response = EmbeddingResponse()

        result = self.config._transform_response(
            response=mock_response,
            api_key="test-api-key",
            logging_obj=self.logging_obj,
            data=data,
            model_response=model_response,
            model=self.model,
            encoding=self.encoding,
            input=input_data,
        )

        # Verify usage includes both text and image tokens
        assert result.usage is not None
        assert result.usage.prompt_tokens == 105  # 5 text + 100 image
        assert result.usage.total_tokens == 105
        assert result.usage.completion_tokens == 0
        assert result.usage.prompt_tokens_details is not None
        assert result.usage.prompt_tokens_details.text_tokens == 5
        assert result.usage.prompt_tokens_details.image_tokens == 100

    def test_transform_response_fallback_token_counting(self):
        """Test that token counting falls back to encoding when billed_units not present"""
        # Mock httpx.Response
        mock_response = MagicMock()
        response_json = {
            "embeddings": [
                [0.1, 0.2, 0.3],
            ],
            "meta": {}  # No billed_units
        }
        mock_response.json = MagicMock(return_value=response_json)

        input_data = ["test text"]
        data = {"texts": input_data, "input_type": "search_query"}
        model_response = EmbeddingResponse()

        result = self.config._transform_response(
            response=mock_response,
            api_key="test-api-key",
            logging_obj=self.logging_obj,
            data=data,
            model_response=model_response,
            model=self.model,
            encoding=self.encoding,
            input=input_data,
        )

        # Verify usage uses encoding (mocked to return 5 tokens)
        assert result.usage is not None
        assert result.usage.prompt_tokens == 5
        assert result.usage.total_tokens == 5
        assert result.usage.completion_tokens == 0
        assert result.usage.prompt_tokens_details is None
        
        # Verify encoding was called
        self.encoding.encode.assert_called()

