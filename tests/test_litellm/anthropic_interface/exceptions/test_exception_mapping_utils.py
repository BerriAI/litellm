"""
Tests for AnthropicExceptionMapping class in litellm/anthropic_interface/exceptions/exception_mapping_utils.py
"""

import json

from litellm.anthropic_interface.exceptions import AnthropicExceptionMapping


class TestCreateErrorResponse:
    """Tests for AnthropicExceptionMapping.create_error_response()"""

    def test_400_invalid_request_error(self):
        """Test 400 maps to invalid_request_error."""
        response = AnthropicExceptionMapping.create_error_response(
            400, "Invalid request"
        )
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
        response = AnthropicExceptionMapping.create_error_response(
            429, "Rate limit exceeded"
        )
        assert response["error"]["type"] == "rate_limit_error"

    def test_500_api_error(self):
        """Test 500 maps to api_error."""
        response = AnthropicExceptionMapping.create_error_response(
            500, "Internal error"
        )
        assert response["error"]["type"] == "api_error"

    def test_with_request_id(self):
        """Test request_id is included when provided."""
        response = AnthropicExceptionMapping.create_error_response(
            400, "Error", request_id="req_123"
        )
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
        assert (
            AnthropicExceptionMapping.extract_error_message(bedrock_msg)
            == "Input is too long for requested model."
        )

    def test_aws_message_format(self):
        """Test extraction from AWS format: {"Message": "..."}"""
        msg = '{"Message":"Bearer Token has expired"}'
        assert (
            AnthropicExceptionMapping.extract_error_message(msg)
            == "Bearer Token has expired"
        )

    def test_generic_message_format(self):
        """Test extraction from generic format: {"message": "..."}"""
        msg = '{"message":"Some error occurred"}'
        assert (
            AnthropicExceptionMapping.extract_error_message(msg)
            == "Some error occurred"
        )

    def test_plain_string(self):
        """Test plain string is returned as-is."""
        assert (
            AnthropicExceptionMapping.extract_error_message("Plain error message")
            == "Plain error message"
        )

    def test_invalid_json(self):
        """Test invalid JSON is returned as-is."""
        assert (
            AnthropicExceptionMapping.extract_error_message("Not JSON {invalid}")
            == "Not JSON {invalid}"
        )

    def test_empty_dict(self):
        """Test empty dict returns original string."""
        assert AnthropicExceptionMapping.extract_error_message("{}") == "{}"


class TestTransformToAnthropicError:
    """Tests for AnthropicExceptionMapping.transform_to_anthropic_error()"""

    def test_passthrough_anthropic_error(self):
        """Test that Anthropic errors pass through unchanged."""
        anthropic_error = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "Rate limited"},
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
            "request_id": "req_existing",
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
            "error": {"type": "api_error", "message": "Server error"},
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

    def test_passthrough_through_litellm_provider_prefixes(self):
        """
        Upstream Anthropic JSON wrapped in `litellm.X: ProviderException - {...}`
        (the real shape of `exception.message`) should be unwrapped and passed
        through with the upstream error.type preserved.
        """
        anthropic_error = {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Number of request tokens has exceeded your rate limit",
            },
        }
        raw = "litellm.RateLimitError: AnthropicException - " + json.dumps(
            anthropic_error
        )
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=429,
            raw_message=raw,
        )
        assert result["type"] == "error"
        # Upstream enum preserved, not derived from status code.
        assert result["error"]["type"] == "rate_limit_error"
        assert (
            result["error"]["message"]
            == "Number of request tokens has exceeded your rate limit"
        )

    def test_wrap_strips_class_prefix_from_router_error(self):
        """
        A plain Router-side error string (no embedded JSON) still gets the
        `litellm.X:` class prefix stripped before being wrapped.
        """
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=429,
            raw_message="litellm.RateLimitError: No deployments available",
        )
        assert result["type"] == "error"
        assert result["error"]["type"] == "rate_limit_error"
        assert result["error"]["message"] == "No deployments available"

    def test_extracts_nested_openai_compat_error_message(self):
        """
        Upstream errors shaped `{"error":{"code","message","type"}}` (OpenAI,
        new-api, OpenAI-compat gateways) need their nested `error.message`
        extracted — falling back to the raw stringified JSON drags a
        provider-specific envelope into the Anthropic envelope.
        """
        upstream = json.dumps(
            {
                "error": {
                    "code": "model_not_found",
                    "message": "model 'foo' not available",
                    "type": "new_api_error",
                }
            }
        )
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=503,
            raw_message=upstream,
        )
        assert result["type"] == "error"
        # 503 → api_error per status map; nested message lifted out cleanly.
        assert result["error"]["type"] == "api_error"
        assert result["error"]["message"] == "model 'foo' not available"

    def test_passthrough_recovers_anthropic_json_with_trailing_garbage(self):
        """
        When the Router appends debug suffixes after the upstream Anthropic
        JSON body (`{"type":"error",...}. Received Model Group=...`), we
        should still detect and passthrough the leading Anthropic envelope
        instead of falling back to wrap-with-status-derived-type.
        """
        anthropic_body = (
            '{"type":"error","error":{"type":"invalid_request_error",'
            '"message":"field messages is required"}}'
        )
        raw = anthropic_body + (
            ". Received Model Group=claude-sonnet-cache"
            "\nAvailable Model Group Fallbacks=None"
        )
        result = AnthropicExceptionMapping.transform_to_anthropic_error(
            status_code=500,  # wrong status (LiteLLM lost upstream 400)
            raw_message=raw,
        )
        # Upstream type preserved even though status_code says 500.
        assert result["error"]["type"] == "invalid_request_error"
        assert result["error"]["message"] == "field messages is required"


class TestStripLitellmWrapperPrefixes:
    """Tests for AnthropicExceptionMapping._strip_litellm_wrapper_prefixes()"""

    def test_plain_text_unchanged(self):
        assert (
            AnthropicExceptionMapping._strip_litellm_wrapper_prefixes("just a message")
            == "just a message"
        )

    def test_empty_string(self):
        assert AnthropicExceptionMapping._strip_litellm_wrapper_prefixes("") == ""

    def test_single_litellm_prefix(self):
        assert (
            AnthropicExceptionMapping._strip_litellm_wrapper_prefixes(
                "litellm.RateLimitError: slow down"
            )
            == "slow down"
        )

    def test_stacked_litellm_prefixes(self):
        assert (
            AnthropicExceptionMapping._strip_litellm_wrapper_prefixes(
                "litellm.ContextWindowExceededError: litellm.BadRequestError: too long"
            )
            == "too long"
        )

    def test_provider_exception_prefix(self):
        assert (
            AnthropicExceptionMapping._strip_litellm_wrapper_prefixes(
                'AnthropicException - {"type":"error"}'
            )
            == '{"type":"error"}'
        )

    def test_combined_litellm_and_provider_prefix(self):
        assert (
            AnthropicExceptionMapping._strip_litellm_wrapper_prefixes(
                'litellm.RateLimitError: AnthropicException - {"type":"error"}'
            )
            == '{"type":"error"}'
        )

    def test_exception_suffix_variant(self):
        """`litellm.XxxException:` (not Error) is also stripped."""
        assert (
            AnthropicExceptionMapping._strip_litellm_wrapper_prefixes(
                "litellm.APIException: boom"
            )
            == "boom"
        )

    def test_idempotent(self):
        once = AnthropicExceptionMapping._strip_litellm_wrapper_prefixes(
            "litellm.RateLimitError: AnthropicException - inner"
        )
        twice = AnthropicExceptionMapping._strip_litellm_wrapper_prefixes(once)
        assert once == twice == "inner"
