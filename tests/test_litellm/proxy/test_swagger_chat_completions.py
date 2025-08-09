"""
Unit test to validate that /chat/completions has the expected schema in Swagger after add_llm_api_request_schema_body runs.

This test ensures that the ProxyChatCompletionRequest Pydantic model is properly added to the OpenAPI schema
for the /chat/completions endpoint, showing all expected fields in the Swagger documentation.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.common_utils.custom_openapi_spec import CustomOpenAPISpec
from litellm.proxy.proxy_server import app


class TestSwaggerChatCompletions:
    """Test suite for validating /chat/completions schema in Swagger documentation."""

    @pytest.fixture
    def client(self):
        """FastAPI test client for the proxy server."""
        return TestClient(app)

    def test_openapi_schema_includes_chat_completions_request_body(self, client):
        """
        Test that the OpenAPI schema includes ProxyChatCompletionRequest schema 
        for /chat/completions endpoints after add_llm_api_request_schema_body runs.
        """
        # Clear any cached schema to ensure we get the latest version
        from litellm.proxy.proxy_server import app
        app.openapi_schema = None
        
        # Get the OpenAPI schema from the running app
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_schema = response.json()
        
        # Verify the schema has the expected structure
        assert "openapi" in openapi_schema
        assert "paths" in openapi_schema
        assert "components" in openapi_schema
        assert "schemas" in openapi_schema["components"]
        
        # Check that ProxyChatCompletionRequest schema is in components
        assert "ProxyChatCompletionRequest" in openapi_schema["components"]["schemas"]
        
        # Get the ProxyChatCompletionRequest schema
        chat_completion_schema = openapi_schema["components"]["schemas"]["ProxyChatCompletionRequest"]
        
        # Verify it has the expected properties structure
        assert "properties" in chat_completion_schema
        properties = chat_completion_schema["properties"]
        
        # Check for core OpenAI chat completion fields
        expected_core_fields = [
            "model",
            "messages", 
            "temperature",
            "top_p",
            "max_tokens",
            "stream",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "logprobs",
            "top_logprobs"
        ]
        
        for field in expected_core_fields:
            assert field in properties, f"Expected field '{field}' not found in ProxyChatCompletionRequest schema"
        
        # Check for LiteLLM-specific fields added by ProxyChatCompletionRequest
        expected_litellm_fields = [
            "guardrails",
            "caching", 
            "num_retries",
            "context_window_fallback_dict",
            "fallbacks"
        ]
        
        for field in expected_litellm_fields:
            assert field in properties, f"Expected LiteLLM field '{field}' not found in ProxyChatCompletionRequest schema"
        
        # Verify model and messages are required fields
        if "required" in chat_completion_schema:
            required_fields = chat_completion_schema["required"]
            assert "model" in required_fields, "Field 'model' should be required"
            assert "messages" in required_fields, "Field 'messages' should be required"

    def test_chat_completions_endpoints_have_request_body_schema(self, client):
        """
        Test that all /chat/completions endpoints reference the ProxyChatCompletionRequest schema
        in their request body definitions.
        """
        # Get the OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_schema = response.json()
        paths = openapi_schema["paths"]
        
        # Check all chat completion paths defined in CustomOpenAPISpec
        chat_completion_paths = [
            "/v1/chat/completions",
            "/chat/completions",
            "/engines/{model}/chat/completions", 
            "/openai/deployments/{model}/chat/completions"
        ]
        
        for path in chat_completion_paths:
            # Some paths might use parameter syntax differences, check if path exists
            path_found = False
            for api_path in paths.keys():
                # Handle parameter differences between {model} and {model:path}
                if path.replace("{model}", "{model:path}") == api_path or path == api_path:
                    path_found = True
                    path_to_check = api_path
                    break
            
            if not path_found:
                continue  # Skip if this specific path variant isn't in the current schema
                
            assert "post" in paths[path_to_check], f"POST method not found for path {path_to_check}"
            post_spec = paths[path_to_check]["post"]
            
            # Verify request body is defined
            if "requestBody" in post_spec:
                request_body = post_spec["requestBody"]
                assert "content" in request_body
                assert "application/json" in request_body["content"]
                
                json_content = request_body["content"]["application/json"]
                assert "schema" in json_content
                
                # Check that it references ProxyChatCompletionRequest
                schema_ref = json_content["schema"]
                expected_ref = "#/components/schemas/ProxyChatCompletionRequest"
                assert "$ref" in schema_ref and schema_ref["$ref"] == expected_ref, \
                    f"Path {path_to_check} should reference ProxyChatCompletionRequest schema"

    @patch('litellm.proxy.common_utils.custom_openapi_spec.CustomOpenAPISpec.add_chat_completion_request_schema')
    def test_add_llm_api_request_schema_body_calls_chat_completion_method(self, mock_add_chat):
        """
        Test that add_llm_api_request_schema_body calls add_chat_completion_request_schema.
        """
        # Create a mock schema
        mock_schema = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {}
        }
        
        # Configure the mock to return the schema
        mock_add_chat.return_value = mock_schema
        
        # Call the main method
        result = CustomOpenAPISpec.add_llm_api_request_schema_body(mock_schema)
        
        # Verify the chat completion method was called
        mock_add_chat.assert_called_once_with(mock_schema)
        assert result == mock_schema

    def test_custom_openapi_spec_chat_completion_paths_constant(self):
        """
        Test that the CHAT_COMPLETION_PATHS constant includes all expected endpoints.
        """
        expected_paths = [
            "/v1/chat/completions",
            "/chat/completions", 
            "/engines/{model}/chat/completions",
            "/openai/deployments/{model}/chat/completions"
        ]
        
        assert hasattr(CustomOpenAPISpec, 'CHAT_COMPLETION_PATHS')
        actual_paths = CustomOpenAPISpec.CHAT_COMPLETION_PATHS
        
        for expected_path in expected_paths:
            assert expected_path in actual_paths, f"Expected path '{expected_path}' not found in CHAT_COMPLETION_PATHS"

    def test_proxy_chat_completion_request_pydantic_model_works(self):
        """
        Test that ProxyChatCompletionRequest properly generates schemas
        and includes the expected LiteLLM-specific fields.
        """
        from litellm.proxy._types import ProxyChatCompletionRequest

        # Check that we can get the schema
        try:
            # Try Pydantic v2 method first
            schema = ProxyChatCompletionRequest.model_json_schema()
        except AttributeError:
            try:
                # Fallback to Pydantic v1 method
                schema = ProxyChatCompletionRequest.schema()
            except AttributeError:
                pytest.fail("Could not get schema from ProxyChatCompletionRequest using either Pydantic v1 or v2 methods")
        
        # Verify schema has properties
        assert "properties" in schema
        properties = schema["properties"]
        
        # Check for core required fields
        assert "model" in properties, "Field 'model' should be in schema"
        assert "messages" in properties, "Field 'messages' should be in schema"
        
        # Check for LiteLLM-specific fields
        litellm_fields = ["guardrails", "caching", "num_retries", "context_window_fallback_dict", "fallbacks"]
        for field in litellm_fields:
            assert field in properties, f"LiteLLM field '{field}' should be in ProxyChatCompletionRequest schema"