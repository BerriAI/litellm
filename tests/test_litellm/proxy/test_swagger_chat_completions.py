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

    def test_chat_completions_endpoints_have_expanded_request_body(self, client):
        """
        Test that /chat/completions endpoint has an expanded request body schema 
        with all individual fields visible (not just a $ref).
        """
        # Clear any cached schema to ensure we get the latest version
        from litellm.proxy.proxy_server import app
        app.openapi_schema = None
        
        # Get the OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_schema = response.json()
        paths = openapi_schema["paths"]
        
        # Check main chat completion path
        path_to_check = "/chat/completions"
        assert path_to_check in paths, f"Path {path_to_check} not found in OpenAPI schema"
        assert "post" in paths[path_to_check], f"POST method not found for path {path_to_check}"
        
        post_spec = paths[path_to_check]["post"]
        
        # Should have request body with expanded schema (not just $ref)
        assert "requestBody" in post_spec, f"Path {path_to_check} should have requestBody"
        request_body = post_spec["requestBody"]
        
        # Check request body structure
        assert "content" in request_body
        assert "application/json" in request_body["content"]
        json_content = request_body["content"]["application/json"]
        assert "schema" in json_content
        
        schema_def = json_content["schema"]
        
        # Should be an expanded object schema, not a $ref
        assert schema_def.get("type") == "object", "Schema should be an expanded object type"
        assert "properties" in schema_def, "Schema should have expanded properties"
        assert "$ref" not in schema_def, "Schema should not be a reference (should be expanded inline)"
        
        # Should have all Pydantic fields as individual properties
        properties = schema_def["properties"]
        assert len(properties) >= 25, f"Expected at least 25 properties, got {len(properties)}"
        
        # Should have core OpenAI fields
        core_fields = ["model", "messages", "temperature", "max_tokens", "stream"]
        for field in core_fields:
            assert field in properties, f"Core field '{field}' should be in expanded properties"
        
        # Should have LiteLLM-specific fields  
        litellm_fields = ["guardrails", "caching", "fallbacks", "num_retries"]
        for field in litellm_fields:
            assert field in properties, f"LiteLLM field '{field}' should be in expanded properties"
        
        # Check required fields
        required_fields = schema_def.get("required", [])
        assert "model" in required_fields, "Model should be marked as required"
        assert "messages" in required_fields, "Messages should be marked as required"
        
        # Should have minimal parameters (only path parameters)
        parameters = post_spec.get("parameters", [])
        # All parameters should be path parameters, no query parameters
        for param in parameters:
            assert param.get("in") == "path", f"Only path parameters expected, found {param.get('in')} parameter: {param.get('name')}"

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

    def test_messages_field_has_example(self, client):
        """
        Test that the messages field in the expanded request body includes a helpful example.
        """
        # Clear any cached schema to ensure we get the latest version
        from litellm.proxy.proxy_server import app
        app.openapi_schema = None
        
        # Get the OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_schema = response.json()
        
        # Navigate to the chat completions request body schema
        chat_completions_post = openapi_schema["paths"]["/chat/completions"]["post"]
        request_body = chat_completions_post["requestBody"]
        schema_def = request_body["content"]["application/json"]["schema"]
        
        # Check that messages field has an example
        messages_field = schema_def["properties"]["messages"]
        assert "example" in messages_field, "Messages field should have an example"
        
        # Verify the example structure
        example = messages_field["example"]
        assert isinstance(example, list), "Messages example should be a list"
        assert len(example) >= 1, "Messages example should have at least 1 message"
        
        # Check that example messages have proper structure
        for message in example:
            assert "role" in message, "Each example message should have a role"
            assert "content" in message, "Each example message should have content"
            assert message["role"] in ["user", "assistant", "system"], f"Invalid role: {message['role']}"
            assert isinstance(message["content"], str), "Message content should be a string"

    def test_request_body_accepts_actual_chat_request(self, client):
        """
        Test that the expanded request body schema accepts a real chat completion request.
        This ensures our schema modifications don't break actual API functionality.
        """
        # Test data that should be valid according to our expanded schema
        test_request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "guardrails": ["no-harmful-content"],
            "caching": True
        }
        
        # This should validate against our schema without errors
        # Note: We're not actually calling the endpoint (which would require API keys)
        # but testing that the request structure is accepted by the schema
        
        # Get the OpenAPI schema to verify our test data matches
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_schema = response.json()
        chat_completions_post = openapi_schema["paths"]["/chat/completions"]["post"]
        
        # Should have expanded request body (not just $ref)
        assert "requestBody" in chat_completions_post
        request_body = chat_completions_post["requestBody"]
        schema_def = request_body["content"]["application/json"]["schema"]
        
        # Verify our test request has fields that exist in the schema
        properties = schema_def["properties"]
        for field_name in test_request.keys():
            assert field_name in properties, f"Field '{field_name}' should be in expanded schema properties"
        
        # Verify required fields are present in test request
        required_fields = schema_def.get("required", [])
        for required_field in required_fields:
            assert required_field in test_request, f"Required field '{required_field}' should be in test request"

    def test_openapi_schema_servers_url_with_root_path(self):
        """
        Test that OpenAPI schema includes correct servers URL when server_root_path is set.
        This ensures Swagger UI works correctly with reverse proxies and subpath deployments.
        """
        from unittest.mock import patch
        from litellm.proxy.proxy_server import get_openapi_schema, custom_openapi, app

        # Test cases: (server_root_path, expected_servers_url)
        # Note: empty string is falsy in Python, so servers won't be set
        test_cases = [
            ("/litellm", "/litellm"),
            ("/litellm/", "/litellm"),  # trailing slash should be removed
            ("litellm", "/litellm"),  # missing leading slash should be added
            ("/api/v1", "/api/v1"),
        ]

        for root_path, expected_url in test_cases:
            # Clear cached schema
            app.openapi_schema = None

            with patch("litellm.proxy.proxy_server.server_root_path", root_path):
                # Test get_openapi_schema
                schema = get_openapi_schema()

                # Should have servers field with correct URL
                assert "servers" in schema, f"servers field should exist when server_root_path={root_path}"
                assert schema["servers"][0]["url"] == expected_url, \
                    f"Expected servers URL '{expected_url}', got '{schema['servers'][0]['url']}' for root_path '{root_path}'"

            # Test custom_openapi as well
            app.openapi_schema = None
            with patch("litellm.proxy.proxy_server.server_root_path", root_path):
                schema = custom_openapi()

                assert "servers" in schema, f"servers field should exist in custom_openapi when server_root_path={root_path}"
                assert schema["servers"][0]["url"] == expected_url, \
                    f"Expected servers URL '{expected_url}' in custom_openapi, got '{schema['servers'][0]['url']}'"        