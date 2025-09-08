import inspect
import os
import sys

import pytest
from pydantic import BaseModel

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.llms.ollama.chat.transformation import OllamaChatConfig
from litellm.utils import get_optional_params


class TestEvent(BaseModel):
    name: str
    value: int


class TestOllamaChatConfigResponseFormat:
    def test_get_optional_params_with_pydantic_model(self):
        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format=TestEvent,
            custom_llm_provider="ollama_chat",
        )
        print(f"optional_params: {optional_params}")

        assert "format" in optional_params
        transformed_format = optional_params["format"]

        expected_schema_structure = TestEvent.model_json_schema()
        transformed_format.pop("additionalProperties")

        assert (
            transformed_format == expected_schema_structure
        ), f"Transformed schema does not match expected. Got: {transformed_format}, Expected: {expected_schema_structure}"

    def test_map_openai_params_with_dict_json_schema(self):
        config = OllamaChatConfig()

        direct_schema = TestEvent.model_json_schema()
        response_format_dict = {
            "type": "json_schema",
            "json_schema": {"schema": direct_schema},
        }

        non_default_params = {"response_format": response_format_dict}

        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format=response_format_dict,
            custom_llm_provider="ollama_chat",
        )

        assert "format" in optional_params
        assert (
            optional_params["format"] == direct_schema
        ), f"Schema from dict did not pass through correctly. Got: {optional_params['format']}, Expected: {direct_schema}"

    def test_map_openai_params_with_json_object(self):
        optional_params = get_optional_params(
            model="ollama_chat/test-model",
            response_format={"type": "json_object"},
            custom_llm_provider="ollama_chat",
        )

        assert "format" in optional_params
        assert (
            optional_params["format"] == "json"
        ), f"Expected 'json' for type 'json_object', got: {optional_params['format']}"

    def test_transform_request_loads_config_parameters(self):
        """Test that transform_request loads config parameters without overriding existing optional_params"""
        # Set config parameters on the class
        import litellm

        litellm.OllamaChatConfig(num_ctx=8000, temperature=0.0)

        try:
            config = OllamaChatConfig()

            # Initial optional_params with existing temperature (should not be overridden)
            optional_params = {"temperature": 0.3}

            # Transform request
            result = config.transform_request(
                model="llama2",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params=optional_params,
                litellm_params={},
                headers={},
            )

            # Verify config values were loaded but existing optional_params were preserved
            assert result["options"]["temperature"] == 0.3  # Should keep existing value
            assert result["options"]["num_ctx"] == 8000  # Should load from config

        finally:
            # Clean up class attributes
            delattr(litellm.OllamaChatConfig, "num_ctx")
            delattr(litellm.OllamaChatConfig, "temperature")
