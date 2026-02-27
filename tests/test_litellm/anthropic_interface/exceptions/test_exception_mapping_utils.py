"""
Tests for AnthropicExceptionMapping class in litellm/anthropic_interface/exceptions/exception_mapping_utils.py
"""

import json

from litellm.anthropic_interface.exceptions import AnthropicExceptionMapping


class TestCreateErrorResponse:
    """Tests for AnthropicExceptionMapping.create_error_response()"""

    def test_400_invalid_request_error(self):
        """Test 400 maps to invalid_request_error."""
        response = AnthropicExceptionMapping.create_error_response(400, "Invalid request")
        assert response["type"] == "error"
        assert response["error"]["type"] == "invalid_request_error"
        assert response["error"]["message"] == "Invalid request"
        assert "request_id" not in response

    def test_401_authentication_error(self):
        """Test 401 maps to authentication_error."""
        response = AnthropicExceptionMapping.create_error_response(401, "Unauthorized")
        assert response["error"]["type"] == "authentication_error"

    def test_403_permission_error(self):
        """Test 403 maps to permission_error."""
        response = AnthropicExceptionMapping.create_error_response(403, "Forbidden")
        assert response["error"]["type"] == "permission_error"

    def test_404_not_found_error(self):
        """Test 404 maps to not_found_error."""
        response = AnthropicExceptionMapping.create_error_response(404, "Not found")
        assert response["error"]["type"] == "not_found_error"

    def test_429_rate_limit_error(self):
        """Test 429 maps to rate_limit_error."""
        response = AnthropicExceptionMapping.create_error_response(429, "Rate limit exceeded")
        assert response["error"]["type"] == "rate_limit_error"

    def test_500_api_error(self):
        """Test 500 maps to api_error."""
        response = AnthropicExceptionMapping.create_error_response(500, "Internal error")
        assert response["error"]["type"] == "api_error"

    def test_with_request_id(self):
        """Test request_id is included when provided."""
        response = AnthropicExceptionMapping.create_error_response(400, "Error", request_id="req_123")
        assert response["request_id"] == "req_123"

    def test_unknown_status_defaults_to_api_error(self):
        """Test unknown status code defaults to api_error."""
        response = AnthropicExceptionMapping.create_error_response(418, "I'm a teapot")
        assert response["error"]["type"] == "api_error"


class TestExtractErrorMessage:
    """Tests for AnthropicExceptionMapping.extract_error_message()"""

    def test_bedrock_format(self):
        """Test extraction from Bedrock format: {"detail": {"message": "..."}}"""
        bedrock_msg = '{"detail":{"message":"Input is too long for requested model."}}'
        assert AnthropicExceptionMapping.extract_error_message(bedrock_msg) == "Input is too long for requested model."

    def test_aws_message_format(self):
        """Test extraction from AWS format: {"Message": "..."}"""
        msg = '{"Message":"Bearer Token has expired"}'
        assert AnthropicExceptionMapping.extract_error_message(msg) == "Bearer Token has expired"

    def test_generic_message_format(self):
        """Test extraction from generic format: {"message": "..."}"""
        msg = '{"message":"Some error occurred"}'
        assert AnthropicExceptionMapping.extract_error_message(msg) == "Some error occurred"

    def test_plain_string(self):
        """Test plain string is returned as-is."""
        assert AnthropicExceptionMapping.extract_error_message("Plain error message") == "Plain error message"

    def test_invalid_json(self):
        """Test invalid JSON is returned as-is."""
        assert AnthropicExceptionMapping.extract_error_message("Not JSON {invalid}") == "Not JSON {invalid}"

    def test_empty_dict(self):
        """Test empty dict returns original string."""
        assert AnthropicExceptionMapping.extract_error_message("{}") == "{}"


class TestTransformToAnthropicError:
    """Tests for AnthropicExceptionMapping.transform_to_anthropic_error()"""

    def test_passthrough_anthropic_error(self):
        """Test that Anthropic errors pass through unchanged."""
        anthropic_error = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Rate limited"}
        }
        raw = json.dumps(anthropic_error)
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=429,
            raw_message=raw,
        )
        assert result["type"] == "error"
        assert result["error"]["type"] == "rate_limit_error"
        assert result["error"]["message"] == "Rate limited"

    def test_passthrough_preserves_existing_request_id(self):
        """Test that existing request_id in Anthropic error is preserved."""
        anthropic_error = {
            "type": "error",
            "error": {"type": "api_error", "message": "Server error"},
            "request_id": "req_existing"
        }
        raw = json.dumps(anthropic_error)
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=500,
            raw_message=raw,
            request_id="req_new",  # Should not override existing
        )
        assert result["request_id"] == "req_existing"

    def test_passthrough_adds_request_id_if_missing(self):
        """Test that request_id is added to Anthropic error if missing."""
        anthropic_error = {
            "type": "error",
            "error": {"type": "api_error", "message": "Server error"}
        }
        raw = json.dumps(anthropic_error)
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=500,
            raw_message=raw,
            request_id="req_123",
        )
        assert result["request_id"] == "req_123"

    def test_translates_bedrock_error(self):
        """Test that Bedrock errors are translated to Anthropic format."""
        bedrock_error = json.dumps({"detail": {"message": "Access denied"}})
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=403,
            raw_message=bedrock_error,
        )
        assert result["type"] == "error"
        assert result["error"]["type"] == "permission_error"
        assert result["error"]["message"] == "Access denied"

    def test_translates_aws_error(self):
        """Test that AWS errors are translated to Anthropic format."""
        aws_error = json.dumps({"Message": "Resource not found"})
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=404,
            raw_message=aws_error,
        )
        assert result["type"] == "error"
        assert result["error"]["type"] == "not_found_error"
        assert result["error"]["message"] == "Resource not found"

    def test_handles_plain_string(self):
        """Test that plain string errors are wrapped in Anthropic format."""
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=400,
            raw_message="Invalid request parameters",
        )
        assert result["type"] == "error"
        assert result["error"]["type"] == "invalid_request_error"
        assert result["error"]["message"] == "Invalid request parameters"

    def test_handles_generic_message_json(self):
        """Test that generic {"message": "..."} JSON is translated."""
        generic_error = json.dumps({"message": "Something went wrong"})
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=500,
            raw_message=generic_error,
        )
        assert result["type"] == "error"
        assert result["error"]["type"] == "api_error"
        assert result["error"]["message"] == "Something went wrong"

    def test_handles_non_dict_json(self):
        """Test that non-dict JSON (e.g., array) is treated as plain string."""
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=400,
            raw_message='["error1", "error2"]',
        )
        assert result["type"] == "error"
        assert result["error"]["message"] == '["error1", "error2"]'
