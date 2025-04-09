import os
import sys
from unittest.mock import MagicMock


sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.cohere.embed.transformation import CohereEmbeddingConfig, EmbeddingResponse
from litellm.utils import decode_base64_floats

class TestCohereTransform:
    def setup_method(self):
        self.config = CohereEmbeddingConfig()
        self.model = "embed-english-v3.0"
        self.logging_obj = MagicMock()

    def test_map_cohere_params(self):
        """Test that parameters are correctly mapped"""

        # Respects float when specified
        result = self.config.map_openai_params(
            non_default_params={"encoding_format": "float"},
            optional_params={},
        )
        assert result == {"embedding_types": ["float"]}

        # Overwrites base64 with float
        result = self.config.map_openai_params(
            non_default_params={"encoding_format": "base64"},
            optional_params={},
        )
        assert result == {"embedding_types": ["float"]}

    def test__transform_response(self):
        """Test that the response is transformed correctly"""

        # Test encoding_format="float"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "9746d887-7ced-43d4-81b9-cc5f7deaa950",
            "texts": ["What is this"],
            "embeddings": {
                "float": [[-0.013511658, 0.002840042, -0.046295166]],
            },
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"input_tokens": 3},
            },
            "response_type": "embeddings_by_type",
        }
        encoding = MagicMock()
        encoding.encode = lambda x: x
        result = self.config._transform_response(
            response=mock_response,
            api_key=None,
            logging_obj=self.logging_obj,
            data={},
            model_response=EmbeddingResponse(),
            model=self.model,
            encoding=encoding,
            input=["What is this"],
            encoding_format="float",
        )
        assert result.data == [
            {"object": "embedding", "index": 0, "embedding": [-0.013511658, 0.002840042, -0.046295166]}
        ]

        # Test encoding_format="base64"
        # (this uses the same mock response as above since we route it through float)
        result = self.config._transform_response(
            response=mock_response,
            api_key=None,
            logging_obj=self.logging_obj,
            data={},
            model_response=EmbeddingResponse(),
            model=self.model,
            encoding=encoding,
            input=["What is this"],
            encoding_format="base64",
        )
        assert result.data == [
            {"object": "embedding", "index": 0, "embedding": "AGBdvAAgOjsAoD29"}
        ]
        assert decode_base64_floats(result.data[0]["embedding"]) == [-0.01351165771484375, 0.0028400421142578125, -0.046295166015625]

        # Test encoding_format=None
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "9746d887-7ced-43d4-81b9-cc5f7deaa950",
            "texts": ["What is this"],
            "embeddings": [[-0.013511658, 0.002840042, -0.046295166]],
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"input_tokens": 3},
            },
            "response_type": "embeddings_by_type",
        }
        result = self.config._transform_response(
            response=mock_response,
            api_key=None,
            logging_obj=self.logging_obj,
            data={},
            model_response=EmbeddingResponse(),
            model=self.model,
            encoding=encoding,
            input=["What is this"],
            encoding_format=None,
        )
        assert result.data == [
            {"object": "embedding", "index": 0, "embedding": [-0.013511658, 0.002840042, -0.046295166]}
        ]
