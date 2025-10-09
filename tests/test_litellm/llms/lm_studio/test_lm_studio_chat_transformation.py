import os
import sys
from unittest.mock import patch

from pydantic import BaseModel

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.llms.lm_studio.chat.transformation import LMStudioChatConfig
from litellm.utils import get_optional_params


class Book(BaseModel):
    title: str
    author: str
    year: int


class TestLMStudioChatConfigResponseFormat:
    def test_get_optional_params_with_pydantic_model(self):
        optional_params = get_optional_params(
            model="lm_studio/test-model",
            response_format=Book,
            custom_llm_provider="lm_studio",
        )

        assert "response_format" in optional_params
        transformed = optional_params["response_format"]
        assert transformed.get("type") == "json_schema"
        schema = transformed.get("json_schema", {}).get("schema")
        assert schema["properties"] == Book.model_json_schema()["properties"]

    def test_map_openai_params_with_dict_json_schema(self):
        config = LMStudioChatConfig()
        schema = Book.model_json_schema()
        response_format_dict = {
            "type": "json_schema",
            "json_schema": {"schema": schema},
        }

        non_default_params = {"response_format": response_format_dict}
        optional_params = get_optional_params(
            model="lm_studio/test-model",
            response_format=response_format_dict,
            custom_llm_provider="lm_studio",
        )

        mapped = config.map_openai_params(
            non_default_params, {}, "lm_studio/test-model", False
        )
        mapped_schema = mapped["response_format"]["json_schema"]["schema"]
        assert mapped_schema["properties"] == schema["properties"]
        opt_schema = optional_params["response_format"]["json_schema"]["schema"]
        assert opt_schema["properties"] == schema["properties"]


def test_lm_studio_get_openai_compatible_provider_info():
    """Test provider info retrieval"""
    config = LMStudioChatConfig()
    
    # Test default behavior (no API key provided)
    _, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_key == "fake-api-key"
    
    # Test explicit API key
    _, api_key = config._get_openai_compatible_provider_info(None, "test-key")
    assert api_key == "test-key"


def test_lm_studio_get_openai_compatible_provider_info_with_env():
    """Test provider info retrieval with environment variables."""
    config = LMStudioChatConfig()
    
    with patch.dict(
        "os.environ",
        {
            "LM_STUDIO_API_BASE": "http://localhost:1234/v1",
            "LM_STUDIO_API_KEY": "env_api_key",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "http://localhost:1234/v1"
        assert api_key == "env_api_key"
