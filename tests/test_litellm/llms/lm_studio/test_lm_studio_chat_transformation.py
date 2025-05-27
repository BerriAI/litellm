import os
import sys

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
