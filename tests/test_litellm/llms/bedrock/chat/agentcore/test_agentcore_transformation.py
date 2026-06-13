"""
Unit tests for Bedrock AgentCore transformation.

Tests:
- Accept header fix (sign_request sets Accept: application/json, text/event-stream)
- JSON response parsing fallback chain (_parse_json_response supports multiple schemas)
- Streaming Content-Type fallback (JSON responses converted to single-chunk streams)
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from unittest.mock import MagicMock, Mock, patch

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
            mock_sign.return_value = (
                {"Authorization": "AWS4-HMAC-SHA256 ..."},
                b'{"prompt":"test"}',
            )
            result_headers, body = config.sign_request(
                headers=headers,
                optional_params={},
                request_data={"prompt": "test"},
                api_base="https://bedrock-agentcore.us-east-1.amazonaws.com/runtimes/test/invocations",
            )
            # Verify _sign_request was called with Accept header already set
            call_args = mock_sign.call_args
            passed_headers = call_args.kwargs.get("headers") or call_args[1].get(
                "headers", {}
            )
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


class TestAgentCoreJsonResponseParsing:
    """Tests for _parse_json_response fallback chain."""

    @pytest.fixture
    def config(self):
        return AmazonAgentCoreConfig()

    def test_parse_json_standard_agentcore_format(self, config):
        """Strategy 1: standard {"result": {"content": [{"text": "..."}]}} format."""
        response_json = {
            "result": {
                "role": "assistant",
                "content": [{"text": "Hello from standard format"}],
            }
        }
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == "Hello from standard format"
        assert parsed["usage"] is None
        assert parsed["final_message"] == response_json["result"]

    def test_parse_json_strands_format(self, config):
        """Strategy 2: Strands {"response": [{"text": "..."}]} format."""
        response_json = {
            "response": [
                {"text": "Based on my research, "},
                {"text": "iOS 18.2 was released."},
            ]
        }
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == "Based on my research, iOS 18.2 was released."
        assert parsed["usage"] is None
        assert parsed["final_message"] is None

    def test_parse_json_string_result(self, config):
        """Strategy 3: plain string {"result": "text"} format."""
        response_json = {"result": "Simple text response"}
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == "Simple text response"
        assert parsed["usage"] is None

    def test_parse_json_string_response(self, config):
        """Strategy 3: plain string {"response": "text"} format."""
        response_json = {"response": "Another text response"}
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == "Another text response"
        assert parsed["usage"] is None

    def test_parse_json_unknown_format_fallback(self, config):
        """Strategy 4: unknown keys fall back to raw JSON."""
        response_json = {"custom_key": "custom_value", "data": [1, 2, 3]}
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == json.dumps(response_json)
        assert parsed["usage"] is None
        assert parsed["final_message"] is None

    def test_parse_json_non_dict_response(self, config):
        """Guard: non-dict JSON (e.g. array) falls back to raw JSON string."""
        response_json = [{"text": "array response"}]
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == json.dumps(response_json)
        assert parsed["usage"] is None
        assert parsed["final_message"] is None

    def test_parse_json_empty_content_in_result(self, config):
        """Standard format with empty content list - preserves existing behavior."""
        response_json = {
            "result": {
                "role": "assistant",
                "content": [],
            }
        }
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == ""
        assert parsed["final_message"] == response_json["result"]

    def test_parse_json_a2a_jsonrpc_nested_message(self, config):
        """Strategy 0: A2A JSON-RPC with result.message.parts[] format."""
        response_json = {
            "jsonrpc": "2.0",
            "id": "test_id",
            "result": {
                "message": {
                    "role": "agent",
                    "parts": [{"kind": "text", "text": "1 + 1 = 2"}],
                    "messageId": "123",
                }
            },
        }
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == "1 + 1 = 2"
        assert parsed["usage"] is None

    def test_parse_json_a2a_jsonrpc_direct_parts(self, config):
        """Strategy 0: A2A JSON-RPC with result.parts[] format (direct message)."""
        response_json = {
            "jsonrpc": "2.0",
            "id": "test_id",
            "result": {
                "kind": "message",
                "parts": [{"kind": "text", "text": "Direct response"}],
            },
        }
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == "Direct response"
        assert parsed["usage"] is None

    def test_parse_json_a2a_jsonrpc_multi_parts(self, config):
        """Strategy 0: A2A JSON-RPC with multiple text parts concatenated."""
        response_json = {
            "jsonrpc": "2.0",
            "id": "test_id",
            "result": {
                "message": {
                    "role": "agent",
                    "parts": [
                        {"kind": "text", "text": "First part"},
                        {"kind": "text", "text": "Second part"},
                    ],
                }
            },
        }
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == "First part Second part"
        assert parsed["usage"] is None

    def test_parse_json_a2a_jsonrpc_empty_falls_through(self, config):
        """Strategy 0: A2A JSON-RPC with empty result falls through to Strategy 3."""
        response_json = {
            "jsonrpc": "2.0",
            "id": "test_id",
            "result": "plain text fallback",
        }
        parsed = config._parse_json_response(response_json)
        assert parsed["content"] == "plain text fallback"
        assert parsed["usage"] is None


class TestAgentCoreNonStreamingJsonFormats:
    """Tests for _get_parsed_response with different JSON formats (non-streaming path)."""

    @pytest.fixture
    def config(self):
        return AmazonAgentCoreConfig()

    def test_get_parsed_response_strands_json(self, config):
        """
        Non-streaming path: _get_parsed_response routes application/json
        to _parse_json_response which handles the Strands format.
        """
        mock_response = Mock(spec=httpx.Response)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "response": [{"text": "Strands agent response via non-streaming"}]
        }
        parsed = config._get_parsed_response(mock_response)
        assert parsed["content"] == "Strands agent response via non-streaming"
        assert parsed["usage"] is None

    def test_get_parsed_response_raw_json_fallback(self, config):
        """
        Non-streaming path: unknown JSON schema falls back to raw JSON string.
        """
        response_json = {"output": "some value"}
        mock_response = Mock(spec=httpx.Response)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = response_json
        parsed = config._get_parsed_response(mock_response)
        assert parsed["content"] == json.dumps(response_json)


class TestAgentCoreStreamingJsonFallback:
    """Tests for streaming Content-Type check (JSON -> single-chunk stream)."""

    def test_sync_streaming_with_json_response(self):
        """
        When stream=True but the agent returns Content-Type: application/json,
        content is extracted and returned instead of silently returning empty.
        Exercises the full path through litellm.completion().
        """
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()
        json_body = {"response": [{"text": "Strands sync response"}]}

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.read.return_value = json.dumps(json_body).encode()

        with patch.object(client, "post", return_value=mock_response):
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/test_agent",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
                client=client,
                api_key="test-jwt-token",
            )

            # Collect content across all chunks
            # CustomStreamWrapper yields content chunk(s) + a synthetic stop chunk
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content

        assert content == "Strands sync response"

    async def test_async_streaming_with_json_response(self):
        """
        Async streaming: same Content-Type: application/json fallback via
        litellm.acompletion(stream=True).
        """
        from unittest.mock import AsyncMock

        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        client = AsyncHTTPHandler()
        json_body = {"response": [{"text": "Strands async response"}]}

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=json.dumps(json_body).encode())

        with patch.object(
            client, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            response = await litellm.acompletion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/test_agent",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
                client=client,
                api_key="test-jwt-token",
            )

            # Collect content across all chunks
            content = ""
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content

        assert content == "Strands async response"

    def test_sync_streaming_malformed_json_raises_error(self):
        """
        When stream=True and Content-Type is application/json but the body
        is malformed JSON, an error is raised with a descriptive message
        (not a raw JSONDecodeError).
        """
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.read.return_value = b"not valid json {{"

        with patch.object(client, "post", return_value=mock_response):
            with pytest.raises(
                Exception, match="Failed to read/parse JSON response body"
            ):
                litellm.completion(
                    model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/test_agent",
                    messages=[{"role": "user", "content": "test"}],
                    stream=True,
                    client=client,
                    api_key="test-jwt-token",
                )

    async def test_async_streaming_malformed_json_raises_error(self):
        """
        Async mirror: malformed JSON body raises a structured error, not a
        raw JSONDecodeError.
        """
        from unittest.mock import AsyncMock

        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        client = AsyncHTTPHandler()

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b"not valid json {{")

        with patch.object(
            client, "post", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(
                Exception, match="Failed to read/parse JSON response body"
            ):
                await litellm.acompletion(
                    model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/test_agent",
                    messages=[{"role": "user", "content": "test"}],
                    stream=True,
                    client=client,
                    api_key="test-jwt-token",
                )
