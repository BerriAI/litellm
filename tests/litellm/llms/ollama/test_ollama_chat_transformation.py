import os
import sys
import pytest
from pydantic import BaseModel
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

from litellm.llms.ollama_chat import OllamaChatConfig

class TestEvent(BaseModel):
    name: str
    value: int

class TestOllamaChatConfigResponseFormat:
    def test_map_openai_params_with_pydantic_model(self):
        config = OllamaChatConfig()
        
        non_default_params = {
            "response_format": TestEvent
        }
        optional_params = {}
        
        expected_schema_structure = TestEvent.model_json_schema()

        config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="ollama_chat/test-model",
            drop_params=False
        )
        
        assert "format" in optional_params, "Transformed 'format' key not found in optional_params"
        
        transformed_format = optional_params["format"]
        
        assert transformed_format == expected_schema_structure, \
               f"Transformed schema does not match expected. Got: {transformed_format}, Expected: {expected_schema_structure}"

    def test_map_openai_params_with_dict_json_schema(self):
        config = OllamaChatConfig()
        
        direct_schema = TestEvent.model_json_schema()
        response_format_dict = {
            "type": "json_schema",
            "json_schema": {"schema": direct_schema}
        }
        
        non_default_params = {
            "response_format": response_format_dict
        }
        optional_params = {}
        
        config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="ollama_chat/test-model",
            drop_params=False
        )
        
        assert "format" in optional_params
        assert optional_params["format"] == direct_schema, \
               f"Schema from dict did not pass through correctly. Got: {optional_params['format']}, Expected: {direct_schema}"

    def test_map_openai_params_with_json_object(self):
        config = OllamaChatConfig()
        
        non_default_params = {
            "response_format": {"type": "json_object"}
        }
        optional_params = {}
        
        config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="ollama_chat/test-model",
            drop_params=False
        )
        
        assert "format" in optional_params
        assert optional_params["format"] == "json", \
               f"Expected 'json' for type 'json_object', got: {optional_params['format']}" 