import os
import sys
from unittest.mock import MagicMock

import httpx

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.jina_ai.embedding.transformation import JinaAIEmbeddingConfig
from litellm.types.utils import EmbeddingResponse


class TestJinaAIEmbeddingTransform:
    def setup_method(self):
        self.config = JinaAIEmbeddingConfig()
        self.model = "jina-embeddings-v2-base-en"
        self.logging_obj = MagicMock()

    def test_map_openai_params(self):
        """Test that 'dimensions' parameter is correctly mapped"""
        test_params = {"dimensions": 1024}
        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        assert result == {"dimensions": 1024}

    def test_transform_embedding_request_text_input(self):
        """Test transformation of a standard text embedding request"""
        input_data = ["hello world", "hello world again"]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        expected_result = {
            "model": self.model,
            "input": input_data,
        }
        assert result == expected_result

    def test_transform_embedding_request_image_input(self):
        """Test transformation of an image embedding request"""
        # a fake base64 string for testing purposes
        input_data = [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        ]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        expected_input = [
            {
                "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            }
        ]
        expected_result = {
            "model": self.model,
            "input": expected_input,
        }
        assert result == expected_result

    def test_transform_embedding_request_mixed_input(self):
        """Test transformation of a mixed text and image embedding request"""
        # a fake base64 string for testing purposes
        base64_str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        input_data = ["hello world", base64_str]
        result = self.config.transform_embedding_request(
            model=self.model,
            input=input_data,
            optional_params={},
            headers={},
        )
        expected_input = [
            {"text": "hello world"},
            {
                "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
            },
        ]
        expected_result = {
            "model": self.model,
            "input": expected_input,
        }
        assert result == expected_result

    def test_transform_embedding_response_logs_request_input(self):
        request_data = {"model": self.model, "input": ["hello world"]}
        raw_response = httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3],
                        "index": 0,
                    }
                ],
                "model": self.model,
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            request=httpx.Request("POST", "https://api.jina.ai/v1/embeddings"),
        )

        response = self.config.transform_embedding_response(
            model=self.model,
            raw_response=raw_response,
            model_response=EmbeddingResponse(),
            logging_obj=self.logging_obj,
            api_key="test-api-key",
            request_data=request_data,
            optional_params={},
            litellm_params={},
        )

        assert response.model == self.model
        assert self.logging_obj.post_call.call_args.kwargs["input"] == ["hello world"]
