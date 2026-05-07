import os
import sys
from unittest.mock import MagicMock

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

import litellm.llms.jina_ai.embedding.transformation as jina_embedding_transformation
from litellm.llms.jina_ai.embedding.transformation import JinaAIEmbeddingConfig


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

    def test_provider_info_uses_jina_api_key_alias(self, monkeypatch):
        """Test provider info resolves the shared Jina credential alias"""

        def fake_get_secret_str(secret_name):
            if secret_name == "JINA_API_KEY":
                return "test-jina-key"
            return None

        monkeypatch.setattr(
            jina_embedding_transformation, "get_secret_str", fake_get_secret_str
        )

        custom_llm_provider, api_base, dynamic_api_key = (
            self.config._get_openai_compatible_provider_info(
                api_base=None,
                api_key=None,
            )
        )

        assert custom_llm_provider == "jina_ai"
        assert api_base == "https://api.jina.ai/v1"
        assert dynamic_api_key == "test-jina-key"

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
