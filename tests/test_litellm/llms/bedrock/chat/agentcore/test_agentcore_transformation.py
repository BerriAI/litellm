"""
Unit tests for Bedrock AgentCore transformation — Accept header fix.

Verifies that AmazonAgentCoreConfig.sign_request() sets the
Accept: application/json, text/event-stream header required by
MCP servers on Bedrock AgentCore.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from unittest.mock import MagicMock, patch

import litellm
from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig


class TestAgentCoreAcceptHeader:
    """Tests for Accept header in AgentCore requests."""

    @pytest.fixture
    def config(self):
        return AmazonAgentCoreConfig()

    def test_sign_request_sets_accept_header_jwt_path(self, config):
        """Test that sign_request sets Accept header when using JWT/Bearer auth."""
        headers = {}
        result_headers, body = config.sign_request(
            headers=headers,
            optional_params={},
            request_data={"prompt": "test"},
            api_base="https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/test/invocations",
            api_key="test-jwt-token",
        )
        assert "Accept" in result_headers
        assert result_headers["Accept"] == "application/json, text/event-stream"

    def test_sign_request_sets_accept_header_sigv4_path(self, config):
        """Test that sign_request sets Accept header when using SigV4 auth."""
        headers = {}
        # SigV4 path requires AWS credentials — mock _sign_request to avoid needing them
        with patch.object(config, "_sign_request") as mock_sign:
            mock_sign.return_value = ({"Authorization": "AWS4-HMAC-SHA256 ..."}, b'{"prompt":"test"}')
            result_headers, body = config.sign_request(
                headers=headers,
                optional_params={},
                request_data={"prompt": "test"},
                api_base="https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/test/invocations",
            )
            # Verify _sign_request was called with Accept header already set
            call_args = mock_sign.call_args
            passed_headers = call_args.kwargs.get("headers") or call_args[1].get("headers", {})
            assert "Accept" in passed_headers
            assert passed_headers["Accept"] == "application/json, text/event-stream"

    def test_accept_header_in_completion_request_jwt(self):
        """
        End-to-end test: verify Accept header appears in the final HTTP request
        when using JWT auth through litellm.completion().
        """
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()
        with patch.object(client, "post", return_value=MagicMock()) as mock_post:
            try:
                litellm.completion(
                    model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/test_runtime",
                    messages=[{"role": "user", "content": "test"}],
                    api_key="test-jwt-token",
                    client=client,
                )
            except Exception:
                pass

            mock_post.assert_called_once()
            headers = mock_post.call_args.kwargs["headers"]
            assert "Accept" in headers
            assert headers["Accept"] == "application/json, text/event-stream"
