"""
Unit tests for Bedrock AgentCore transformation.

Tests:
- Accept header fix (sign_request sets Accept: application/json, text/event-stream)
- JSON response parsing fallback chain (_parse_json_response supports multiple schemas)
- Streaming Content-Type fallback (JSON responses converted to single-chunk streams)
- Multimodal content preservation (transform_request forwards OpenAI content blocks)
"""

import json

import httpx
import pytest


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

        No exception swallowing: if completion() raises (for example because the
        injected client was silently ignored and a real network call was made),
        the test must fail with that error, not a misleading mock assertion.
        """
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "result": {"role": "assistant", "content": [{"text": "agent reply"}]}
        }

        with patch.object(client, "post", return_value=mock_response) as mock_post:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/test_runtime",
                messages=[{"role": "user", "content": "test"}],
                api_key="test-jwt-token",
                client=client,
            )

        mock_post.assert_called_once()
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Accept"] == "application/json, text/event-stream"
        assert response.choices[0].message.content == "agent reply"


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


class TestAgentCoreMultimodalContent:
    """Tests for transform_request forwarding OpenAI multimodal content blocks.

    AgentCore Runtime is schemaless on the agent side — the agent author's
    @app.entrypoint handler parses whatever JSON arrives. transform_request
    only emits {"prompt": "<text>"} by default and drops image_url, file, and
    other non-text blocks.

    When the ``forward_multimodal_content`` litellm param is set, the OpenAI
    content list is forwarded verbatim under a "content" field whenever the last
    message contains a non-text block. This is opt-in: an agent must be written
    to read payload["content"]. Without the flag, the payload is byte-identical
    to the legacy {"prompt": "..."} shape.
    """

    @pytest.fixture
    def config(self):
        return AmazonAgentCoreConfig()

    @pytest.fixture
    def transform_kwargs(self):
        """Default kwargs — forwarding is OFF (no opt-in flag)."""
        return {
            "model": "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:111111111111:runtime/test_agent",
            "optional_params": {},
            "litellm_params": {},
            "headers": {},
        }

    @pytest.fixture
    def opted_in_kwargs(self, transform_kwargs):
        """Kwargs with the opt-in flag set in optional_params."""
        return {
            **transform_kwargs,
            "optional_params": {"forward_multimodal_content": True},
        }

    def test_string_content_payload_byte_identical_to_legacy(
        self, config, transform_kwargs
    ):
        """String content → exactly {"prompt": "<text>"}, no extra fields."""
        messages = [{"role": "user", "content": "hello agent"}]
        payload = config.transform_request(messages=messages, **transform_kwargs)
        assert payload == {"prompt": "hello agent"}

    def test_file_block_not_forwarded_by_default(self, config, transform_kwargs):
        """Default (no opt-in flag): file blocks are NOT forwarded — backward compat."""
        content = [
            {"type": "text", "text": "summarize this report"},
            {
                "type": "file",
                "file": {
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                },
            },
        ]
        messages = [{"role": "user", "content": content}]
        payload = config.transform_request(messages=messages, **transform_kwargs)
        assert payload == {"prompt": "summarize this report"}
        assert "content" not in payload

    def test_text_only_list_content_no_content_field(self, config, opted_in_kwargs):
        """All-text content list → no "content" field even when opted in."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello agent"}],
            }
        ]
        payload = config.transform_request(messages=messages, **opted_in_kwargs)
        assert payload == {"prompt": "hello agent"}
        assert "content" not in payload

    def test_file_data_block_passthrough(self, config, opted_in_kwargs):
        """Opted in: a file block → "content" carries the original list verbatim."""
        content = [
            {"type": "text", "text": "summarize this report"},
            {
                "type": "file",
                "file": {
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                },
            },
        ]
        messages = [{"role": "user", "content": content}]
        payload = config.transform_request(messages=messages, **opted_in_kwargs)
        assert payload["prompt"] == "summarize this report"
        # Contents forwarded verbatim, but as a distinct list (no aliasing).
        assert payload["content"] == content
        assert payload["content"] is not content

    def test_image_url_block_passthrough(self, config, opted_in_kwargs):
        """Opted in: an image_url block → "content" carries it verbatim."""
        content = [
            {"type": "text", "text": "what is in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ]
        messages = [{"role": "user", "content": content}]
        payload = config.transform_request(messages=messages, **opted_in_kwargs)
        assert payload["prompt"] == "what is in this image?"
        assert payload["content"] == content
        assert payload["content"] is not content

    def test_mixed_text_and_files_payload_shape(self, config, opted_in_kwargs):
        """Opted in: text + file + image → both "prompt" (text-only) and "content"."""
        content = [
            {"type": "text", "text": "first sentence."},
            {
                "type": "file",
                "file": {
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
                },
            },
            {"type": "text", "text": "second sentence."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ]
        messages = [{"role": "user", "content": content}]
        payload = config.transform_request(messages=messages, **opted_in_kwargs)
        # prompt is the text-only flatten produced by convert_content_list_to_str.
        assert "first sentence." in payload["prompt"]
        assert "second sentence." in payload["prompt"]
        assert "JVBERi0xLjQK" not in payload["prompt"]
        assert "iVBORw0KGgo=" not in payload["prompt"]
        # content carries every block in original order.
        assert payload["content"] == content

    def test_forwarded_content_does_not_alias_message(self, config, opted_in_kwargs):
        """Regression: the forwarded list is a shallow copy, so mutating the
        returned payload before serialization must not leak back into the caller's
        messages[-1]["content"]."""
        content = [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ]
        messages = [{"role": "user", "content": content}]
        payload = config.transform_request(messages=messages, **opted_in_kwargs)

        payload["content"].append({"type": "text", "text": "injected"})

        assert len(messages[-1]["content"]) == 2
        assert {"type": "text", "text": "injected"} not in messages[-1]["content"]

    def test_only_last_message_content_preserved(self, config, opted_in_kwargs):
        """Opted in: file blocks in earlier messages don't trigger "content" — last only."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "context"},
                    {
                        "type": "file",
                        "file": {
                            "filename": "old.pdf",
                            "file_data": "data:application/pdf;base64,Zm9v",
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "follow-up question with no files"},
        ]
        payload = config.transform_request(messages=messages, **opted_in_kwargs)
        assert payload == {"prompt": "follow-up question with no files"}
        assert "content" not in payload

    def test_unknown_non_text_block_type_passthrough(self, config, opted_in_kwargs):
        """Opted in: unknown block types (e.g. input_audio) flow through."""
        content = [
            {"type": "text", "text": "transcribe this"},
            {
                "type": "input_audio",
                "input_audio": {"data": "U29tZUF1ZGlvQnl0ZXM=", "format": "wav"},
            },
        ]
        messages = [{"role": "user", "content": content}]
        payload = config.transform_request(messages=messages, **opted_in_kwargs)
        assert payload["prompt"] == "transcribe this"
        assert payload["content"] == content
        assert payload["content"] is not content

    def test_forward_flag_as_string_true(self, config, transform_kwargs):
        """The opt-in flag accepts config/env string values like "true"."""
        content = [
            {"type": "text", "text": "hi"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ]
        messages = [{"role": "user", "content": content}]
        kwargs = {
            **transform_kwargs,
            "optional_params": {"forward_multimodal_content": "true"},
        }
        payload = config.transform_request(messages=messages, **kwargs)
        assert payload["content"] == content
        assert payload["content"] is not content

    def test_forward_flag_false_explicit(self, config, transform_kwargs):
        """Explicit falsy flag → no content field."""
        content = [
            {"type": "text", "text": "hi"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ]
        messages = [{"role": "user", "content": content}]
        kwargs = {
            **transform_kwargs,
            "optional_params": {"forward_multimodal_content": False},
        }
        payload = config.transform_request(messages=messages, **kwargs)
        assert "content" not in payload

    def test_forward_flag_via_litellm_params(self, config, transform_kwargs):
        """The opt-in flag is also honored when set in litellm_params."""
        content = [
            {"type": "text", "text": "hi"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
            },
        ]
        messages = [{"role": "user", "content": content}]
        kwargs = {
            **transform_kwargs,
            "litellm_params": {"forward_multimodal_content": True},
        }
        payload = config.transform_request(messages=messages, **kwargs)
        assert payload["content"] == content
        assert payload["content"] is not content

@pytest.mark.usefixtures("fake_provider_credentials")
def test_bedrock_agentcore_without_api_key_uses_sigv4():
    """
    Test that AgentCore uses AWS SigV4 signing when api_key is not provided
    """
    import json

    litellm.turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
                messages=[
                    {
                        "role": "user",
                        "content": "Test SigV4",
                    }
                ],
                # No api_key provided - should use SigV4
                runtimeSessionId="sigv4-test-session",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify headers - should have AWS SigV4 headers, not Bearer token
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")

        # Should NOT have Bearer Authorization when using SigV4
        if "Authorization" in headers:
            assert not headers["Authorization"].startswith("Bearer ")
            # Should have AWS4-HMAC-SHA256 signature
            assert "AWS4-HMAC-SHA256" in headers["Authorization"]

        # Session ID should still be present
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert (
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"]
            == "sigv4-test-session"
        )


def test_agentcore_transform_response_sse():
    """
    Integration test for transform_response with SSE response
    Verifies end-to-end transformation of SSE responses to ModelResponse
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig
    from litellm.types.utils import ModelResponse

    config = AmazonAgentCoreConfig()

    sse_data = """data: {"event":{"contentBlockDelta":{"delta":{"text":"SSE "}}}}

data: {"event":{"contentBlockDelta":{"delta":{"text":"response"}}}}

data: {"event":{"metadata":{"usage":{"inputTokens":20,"outputTokens":10,"totalTokens":30}}}}

data: {"message":{"role":"assistant","content":[{"text":"SSE response"}]}}
"""

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.text = sse_data
    mock_response.status_code = 200

    model_response = ModelResponse()

    mock_logging = MagicMock()

    result = config.transform_response(
        model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=mock_logging,
        request_data={},
        messages=[{"role": "user", "content": "test"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert len(result.choices) == 1
    assert result.choices[0].message.content == "SSE response"
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"

    assert hasattr(result, "usage")
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 30


@pytest.mark.usefixtures("fake_provider_credentials")
def test_bedrock_agentcore_with_runtime_user_id():
    """
    Test AgentCore with runtimeUserId parameter
    """
    import json

    litellm.turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
                messages=[
                    {
                        "role": "user",
                        "content": "Hello",
                    }
                ],
                runtimeUserId="test-user-123",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify headers - user ID should be in header
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")
        assert "X-Amzn-Bedrock-AgentCore-Runtime-User-Id" in headers
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-User-Id"] == "test-user-123"


def test_agentcore_parse_sse_response_without_final_message():
    """
    Unit test for SSE response parsing when only deltas are present (no final message)
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    sse_data = """data: {"event":{"contentBlockDelta":{"delta":{"text":"First "}}}}

data: {"event":{"contentBlockDelta":{"delta":{"text":"second "}}}}

data: {"event":{"contentBlockDelta":{"delta":{"text":"third"}}}}
"""

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.text = sse_data

    parsed = config._get_parsed_response(mock_response)

    assert parsed["content"] == "First second third"
    assert parsed["final_message"] is None


@pytest.mark.usefixtures("fake_provider_credentials")
def test_agentcore_synchronous_non_streaming_response():
    """
    Test that synchronous (non-streaming) AgentCore calls still work correctly
    after streaming simplification changes.

    This test verifies:
    1. Synchronous completion calls work (stream=False or no stream param)
    2. Response is properly parsed and returned as ModelResponse
    3. Content is extracted correctly
    4. Usage data is calculated when not provided by API

    This is a regression test for the streaming simplification changes
    to ensure we didn't break the non-streaming code path.
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm.turn_on_debug()
    client = HTTPHandler()

    # Mock a JSON response (typical for synchronous AgentCore calls)
    mock_json_response = {
        "result": {
            "role": "assistant",
            "content": [{"text": "This is a synchronous response from AgentCore."}],
        }
    }

    # Create a mock response object
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = mock_json_response

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        # Make a synchronous (non-streaming) completion call
        response = litellm.completion(
            model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
            messages=[
                {
                    "role": "user",
                    "content": "Test synchronous response",
                }
            ],
            stream=False,  # Explicitly disable streaming
            client=client,
        )

        # Verify the response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0

        # Verify content
        message = response.choices[0].message
        assert message is not None
        assert message.content == "This is a synchronous response from AgentCore."
        assert message.role == "assistant"

        # Verify completion metadata
        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].index == 0

        # Verify usage data exists (either from API or calculated)
        assert hasattr(response, "usage")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

        print(f"Synchronous response: {response}")
        print(f"Content: {message.content}")
        print(
            f"Usage: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}, total={response.usage.total_tokens}"
        )


def test_agentcore_parse_sse_response():
    """
    Unit test for SSE response parsing (streaming response consumed as text)
    Verifies that text/event-stream responses are parsed correctly
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    sse_data = """data: {"event":{"contentBlockDelta":{"delta":{"text":"Hello "}}}}

data: {"event":{"contentBlockDelta":{"delta":{"text":"from SSE"}}}}

data: {"event":{"metadata":{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}}}

data: {"message":{"role":"assistant","content":[{"text":"Hello from SSE"}]}}
"""

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "text/event-stream"}
    mock_response.text = sse_data

    parsed = config._get_parsed_response(mock_response)

    assert parsed["content"] == "Hello from SSE"
    assert parsed["usage"] is not None
    assert parsed["usage"]["inputTokens"] == 10
    assert parsed["usage"]["outputTokens"] == 5
    assert parsed["usage"]["totalTokens"] == 15
    assert parsed["final_message"] is not None
    assert parsed["final_message"]["role"] == "assistant"


@pytest.mark.usefixtures("fake_provider_credentials")
def test_bedrock_agentcore_with_all_parameters():
    """
    Test AgentCore with all parameters: api_key, runtimeSessionId, runtimeUserId
    """
    import json

    litellm.turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    test_jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
                messages=[
                    {
                        "role": "user",
                        "content": "Complete test",
                    }
                ],
                api_key=test_jwt_token,
                runtimeSessionId="full-test-session-id",
                runtimeUserId="full-test-user-id",
                qualifier="LATEST",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify URL includes qualifier
        assert "url" in call_kwargs
        url = call_kwargs["url"]
        print(f"URL: {url}")
        assert "qualifier=LATEST" in url

        # Verify all headers are present
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")

        # Check Bearer token authorization
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {test_jwt_token}"

        # Check session and user IDs
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert (
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"]
            == "full-test-session-id"
        )
        assert "X-Amzn-Bedrock-AgentCore-Runtime-User-Id" in headers
        assert (
            headers["X-Amzn-Bedrock-AgentCore-Runtime-User-Id"] == "full-test-user-id"
        )

        # Verify JSON body
        assert "data" in call_kwargs
        request_data = json.loads(call_kwargs["data"])
        print(f"Request data: {json.dumps(request_data, indent=2)}")
        assert "prompt" in request_data
        assert request_data["prompt"] == "Complete test"


@pytest.mark.usefixtures("fake_provider_credentials")
def test_bedrock_agentcore_with_api_key_bearer_token():
    """
    Test AgentCore with api_key parameter for JWT/Bearer token authentication
    """
    import json

    litellm.turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    test_jwt_token = "test-jwt-token-header.payload.signature"

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
                messages=[
                    {
                        "role": "user",
                        "content": "Test JWT authentication",
                    }
                ],
                api_key=test_jwt_token,
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify Authorization header with Bearer token
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")
        assert "Authorization" in headers
        assert headers["Authorization"] == f"Bearer {test_jwt_token}"
        assert headers["Content-Type"] == "application/json"

        # Verify the request body is JSON-encoded (not SigV4 signed)
        assert "data" in call_kwargs
        request_data = json.loads(call_kwargs["data"])
        print(f"Request data: {json.dumps(request_data, indent=2)}")
        assert "prompt" in request_data
        assert request_data["prompt"] == "Test JWT authentication"


@pytest.mark.usefixtures("fake_provider_credentials")
def test_bedrock_agentcore_with_session_and_user():
    """
    Test AgentCore with both runtimeSessionId and runtimeUserId
    """
    import json

    litellm.turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
                messages=[
                    {
                        "role": "user",
                        "content": "Test message",
                    }
                ],
                runtimeSessionId="session-abc-123",
                runtimeUserId="user-xyz-789",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify headers contain both session and user IDs
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert (
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == "session-abc-123"
        )
        assert "X-Amzn-Bedrock-AgentCore-Runtime-User-Id" in headers
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-User-Id"] == "user-xyz-789"


def test_agentcore_parse_json_response():
    """
    Unit test for JSON response parsing (non-streaming)
    Verifies that content-type: application/json responses are parsed correctly
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig

    config = AmazonAgentCoreConfig()

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "result": {
            "role": "assistant",
            "content": [{"text": "Hello from JSON response"}],
        }
    }

    parsed = config._get_parsed_response(mock_response)

    assert parsed["content"] == "Hello from JSON response"
    assert parsed["usage"] is None
    assert parsed["final_message"] == mock_response.json.return_value["result"]


def test_agentcore_transform_response_json():
    """
    Integration test for transform_response with JSON response
    Verifies end-to-end transformation of JSON responses to ModelResponse
    """
    from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig
    from litellm.types.utils import ModelResponse

    config = AmazonAgentCoreConfig()

    mock_response = Mock(spec=httpx.Response)
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "result": {
            "role": "assistant",
            "content": [{"text": "Response from transform_response"}],
        }
    }
    mock_response.status_code = 200

    model_response = ModelResponse()

    mock_logging = MagicMock()

    result = config.transform_response(
        model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=mock_logging,
        request_data={},
        messages=[{"role": "user", "content": "test"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Response from transform_response"
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"
    assert result.choices[0].index == 0


@pytest.mark.usefixtures("fake_provider_credentials")
def test_bedrock_agentcore_with_custom_params():
    """
    Test AgentCore request structure with custom parameters
    """
    import json

    litellm.turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post", return_value=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC",
                messages=[
                    {
                        "role": "user",
                        "content": "Explain machine learning in simple terms",
                    }
                ],
                runtimeSessionId="litellm-test-session-id-12345678901234567890",
                qualifier="DEFAULT",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        print(f"mock_post.call_args.kwargs: {call_kwargs}")

        # Verify URL structure - should include ARN and qualifier
        assert "url" in call_kwargs
        url = call_kwargs["url"]
        print(f"URL: {url}")
        assert (
            "/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aus-west-2%3A888602223428%3Aruntime%2Fhosted_agent_r9jvp-3ySZuRHjLC/invocations"
            in url
        )
        assert "qualifier=DEFAULT" in url

        # Verify headers - session ID should be in header
        assert "headers" in call_kwargs
        headers = call_kwargs["headers"]
        print(f"Headers: {headers}")
        assert "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id" in headers
        assert (
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"]
            == "litellm-test-session-id-12345678901234567890"
        )

        # Verify the request body - should just be the payload
        assert "data" in call_kwargs or "json" in call_kwargs

        # Parse the request data
        if "data" in call_kwargs:
            request_data = json.loads(call_kwargs["data"])
        else:
            request_data = call_kwargs["json"]

        print(f"Request data: {json.dumps(request_data, indent=2)}")

        # Body should just contain the prompt
        assert "prompt" in request_data
        assert request_data["prompt"] == "Explain machine learning in simple terms"
