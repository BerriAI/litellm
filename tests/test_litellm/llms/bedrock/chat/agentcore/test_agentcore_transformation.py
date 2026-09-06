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


AGENTCORE_TEST_MODEL = (
    "agentcore/arn:aws:bedrock-agentcore:us-west-2:111111111111:runtime/test_agent"
)
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    },
}


class TestAgentCoreForwardTools:
    """Tests for transform_request forwarding the caller's OpenAI tool declarations.

    AgentCore Runtime is schemaless on the agent side — the agent author's
    @app.entrypoint handler parses whatever JSON arrives. An agent that implements
    a capability itself (its own search, its own retrieval) has no way today to
    learn whether the caller offered that capability for this turn: AgentCore
    declares no supported OpenAI params, so ``tools`` never reaches the transform.

    When the ``forward_tools`` litellm param is set, the OpenAI tools list is
    forwarded verbatim under a "tools" field. This is opt-in: an agent must be
    written to read payload["tools"]. Without the flag, the payload is
    byte-identical to the legacy {"prompt": "..."} shape.

    This is a one-way signal — AgentCore responses carry no tool-call channel, so
    none of this implies function-calling support.
    """

    @pytest.fixture
    def config(self):
        return AmazonAgentCoreConfig()

    @pytest.fixture
    def messages(self):
        return [{"role": "user", "content": "what happened today"}]

    @pytest.fixture
    def base_kwargs(self):
        """Default kwargs — forwarding is OFF (no opt-in flag)."""
        return {
            "model": "bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:111111111111:runtime/test_agent",
            "optional_params": {},
            "litellm_params": {},
            "headers": {},
        }

    def test_tools_not_forwarded_by_default(self, config, messages, base_kwargs):
        """Default (no opt-in flag): tools are NOT forwarded — backward compat."""
        base_kwargs["optional_params"] = {"tools": [WEB_SEARCH_TOOL]}

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert payload == {"prompt": "what happened today"}
        assert "tools" not in payload

    def test_tools_forwarded_when_opted_in(self, config, messages, base_kwargs):
        """Flag on + tools present → forwarded verbatim."""
        tools = [WEB_SEARCH_TOOL]
        base_kwargs["optional_params"] = {"tools": tools, "forward_tools": True}

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert payload["tools"] == tuple(tools)
        assert payload["prompt"] == "what happened today"

    def test_multiple_tools_forwarded_verbatim(self, config, messages, base_kwargs):
        """The list is not filtered or reshaped — the agent decides what it wants."""
        calculator = {"type": "function", "function": {"name": "calculator"}}
        tools = [WEB_SEARCH_TOOL, calculator]
        base_kwargs["optional_params"] = {"tools": tools, "forward_tools": True}

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert payload["tools"] == tuple(tools)

    def test_no_tools_key_when_opted_in_without_tools(
        self, config, messages, base_kwargs
    ):
        """Flag on but the caller declared nothing → no "tools" field."""
        base_kwargs["optional_params"] = {"forward_tools": True}

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert payload == {"prompt": "what happened today"}

    def test_empty_tools_list_not_forwarded(self, config, messages, base_kwargs):
        """Clients send tools=[] when a tool toggle is off — treat it as absent."""
        base_kwargs["optional_params"] = {"tools": [], "forward_tools": True}

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert payload == {"prompt": "what happened today"}
        assert "tools" not in payload

    def test_non_list_tools_not_forwarded(self, config, messages, base_kwargs):
        """A malformed tools value must not reach the payload."""
        base_kwargs["optional_params"] = {
            "tools": {"name": "web_search"},
            "forward_tools": True,
        }

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert "tools" not in payload

    @pytest.mark.parametrize("flag", ["true", "True", "1", "yes", "on", " TRUE "])
    def test_truthy_string_flags(self, config, messages, base_kwargs, flag):
        """Config/env values arrive as strings."""
        base_kwargs["optional_params"] = {
            "tools": [WEB_SEARCH_TOOL],
            "forward_tools": flag,
        }

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert payload["tools"] == (WEB_SEARCH_TOOL,)

    @pytest.mark.parametrize("flag", ["false", "0", "no", "off", ""])
    def test_falsy_string_flags(self, config, messages, base_kwargs, flag):
        base_kwargs["optional_params"] = {
            "tools": [WEB_SEARCH_TOOL],
            "forward_tools": flag,
        }

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert "tools" not in payload

    def test_flag_read_from_litellm_params(self, config, messages, base_kwargs):
        """litellm_params is the fallback source for direct transform_request callers.

        Deployment-level and per-request flags both arrive in optional_params (see
        TestAgentCoreForwardToolsEndToEnd), so this branch only serves callers that
        build litellm_params themselves.
        """
        base_kwargs["optional_params"] = {"tools": [WEB_SEARCH_TOOL]}
        base_kwargs["litellm_params"] = {"forward_tools": True}

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert payload["tools"] == (WEB_SEARCH_TOOL,)

    def test_optional_params_wins_over_litellm_params(
        self, config, messages, base_kwargs
    ):
        """optional_params is read first, so it wins over the litellm_params fallback."""
        base_kwargs["optional_params"] = {
            "tools": [WEB_SEARCH_TOOL],
            "forward_tools": False,
        }
        base_kwargs["litellm_params"] = {"forward_tools": True}

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert "tools" not in payload

    def test_payload_does_not_alias_optional_params(
        self, config, messages, base_kwargs
    ):
        """The payload holds an immutable copy, so it cannot alias the caller's list.

        The forwarded value serializes to the same JSON array either way, so the wire
        format is asserted here too rather than just the in-memory type.
        """
        tools = [WEB_SEARCH_TOOL]
        base_kwargs["optional_params"] = {"tools": tools, "forward_tools": True}

        payload = config.transform_request(messages=messages, **base_kwargs)

        assert isinstance(payload["tools"], tuple)
        with pytest.raises(AttributeError):
            payload["tools"].append({"type": "function", "function": {"name": "extra"}})
        assert tools == [WEB_SEARCH_TOOL]
        assert json.loads(json.dumps(payload))["tools"] == [WEB_SEARCH_TOOL]

    def test_tools_and_multimodal_content_are_independent(self, config, base_kwargs):
        """Both opt-ins together → both keys present, neither disturbs the other."""
        content = [
            {"type": "text", "text": "what is this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGVsbG8="},
            },
        ]
        tools = [WEB_SEARCH_TOOL]
        base_kwargs["optional_params"] = {
            "tools": tools,
            "forward_tools": True,
            "forward_multimodal_content": True,
        }

        payload = config.transform_request(
            messages=[{"role": "user", "content": content}], **base_kwargs
        )

        assert payload["tools"] == tuple(tools)
        assert payload["content"] == content
        assert payload["prompt"] == "what is this"

    def test_session_header_still_set_when_forwarding(
        self, config, messages, base_kwargs
    ):
        """Forwarding must not disturb the existing header contract."""
        base_kwargs["optional_params"] = {
            "tools": [WEB_SEARCH_TOOL],
            "forward_tools": True,
            "runtimeSessionId": "session-abc",
        }

        config.transform_request(messages=messages, **base_kwargs)

        assert (
            base_kwargs["headers"]["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"]
            == "session-abc"
        )


class TestAgentCoreForwardToolsEndToEnd:
    """``tools`` only reaches transform_request via allowed_openai_params.

    AgentCore reports no supported OpenAI params, so get_optional_params drops
    ``tools`` (or raises without drop_params). ``allowed_openai_params=["tools"]``
    is the documented escape hatch, and is what makes forwarding reachable.
    """

    def test_tools_dropped_without_allowed_openai_params(self):
        optional_params = litellm.utils.get_optional_params(
            model=AGENTCORE_TEST_MODEL,
            custom_llm_provider="bedrock",
            tools=[WEB_SEARCH_TOOL],
            drop_params=True,
        )

        assert "tools" not in optional_params

    def test_allowed_openai_params_carries_tools_to_the_transform(self):
        optional_params = litellm.utils.get_optional_params(
            model=AGENTCORE_TEST_MODEL,
            custom_llm_provider="bedrock",
            tools=[WEB_SEARCH_TOOL],
            allowed_openai_params=["tools"],
        )

        assert optional_params["tools"] == [WEB_SEARCH_TOOL]

        payload = AmazonAgentCoreConfig().transform_request(
            model=AGENTCORE_TEST_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={"forward_tools": True},
            headers={},
        )

        assert payload["tools"] == (WEB_SEARCH_TOOL,)

    def test_flag_reaches_the_transform_via_optional_params(self):
        """The flag itself is not an OpenAI param, so it lands in optional_params.

        get_litellm_params only forwards an allowlist of keys, which excludes
        forward_tools, so this is the path every real caller takes: both the flag and
        the tools list arrive in optional_params and litellm_params stays empty.
        """
        optional_params = litellm.utils.get_optional_params(
            model=AGENTCORE_TEST_MODEL,
            custom_llm_provider="bedrock",
            tools=[WEB_SEARCH_TOOL],
            allowed_openai_params=["tools"],
            forward_tools=True,
        )

        assert optional_params["forward_tools"] is True

        payload = AmazonAgentCoreConfig().transform_request(
            model=AGENTCORE_TEST_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert payload["tools"] == (WEB_SEARCH_TOOL,)

    def test_allowed_openai_params_alone_does_not_forward(self):
        """Both opt-ins are required; the escape hatch alone changes nothing."""
        optional_params = litellm.utils.get_optional_params(
            model=AGENTCORE_TEST_MODEL,
            custom_llm_provider="bedrock",
            tools=[WEB_SEARCH_TOOL],
            allowed_openai_params=["tools"],
        )

        payload = AmazonAgentCoreConfig().transform_request(
            model=AGENTCORE_TEST_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )

        assert payload == {"prompt": "hi"}
