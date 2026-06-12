"""
Tests for Snowflake Cortex native endpoint migration.

Covers:
  - SnowflakeConfig with auto-routing:
    - Non-Claude models → /chat/completions (OpenAI format)
    - Claude models → /messages (Anthropic format)

Run:
    pytest tests/test_litellm/llms/snowflake/test_snowflake_native_endpoints.py -v
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.snowflake.chat.transformation import (
    SnowflakeConfig,
    _is_claude_model,
)
from litellm.types.utils import ModelResponse


# ─── Fixtures ──────────────────────────────────────────────────────────────

ACCOUNT_ID = "myaccount"
API_BASE = f"https://{ACCOUNT_ID}.snowflakecomputing.com"
PAT_TOKEN = "pat/my-secret-pat-token"
JWT_TOKEN = "eyJhbGciOiJSUzI1NiJ9.test"


def _mock_logging():
    m = MagicMock()
    m.post_call = MagicMock()
    return m


def _make_openai_response(content: str = "Hello!") -> httpx.Response:
    body = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "model": "llama3.1-70b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return httpx.Response(200, json=body)


def _make_anthropic_response(content: str = "Hello!") -> httpx.Response:
    body = {
        "id": "msg_abc123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    return httpx.Response(200, json=body)


# ─── SnowflakeConfig (OpenAI-compatible) ───────────────────────────────────

class TestSnowflakeConfigURL:
    def setup_method(self):
        self.cfg = SnowflakeConfig()

    def test_url_with_account_id_in_optional_params(self):
        optional_params = {"account_id": ACCOUNT_ID}
        url = self.cfg.get_complete_url(
            api_base=None,
            api_key=JWT_TOKEN,
            model="snowflake/llama3.1-70b",
            optional_params=optional_params,
            litellm_params={},
        )
        assert url == f"https://{ACCOUNT_ID}.snowflakecomputing.com/api/v2/cortex/v1/chat/completions"

    def test_url_with_explicit_api_base(self):
        url = self.cfg.get_complete_url(
            api_base=API_BASE,
            api_key=JWT_TOKEN,
            model="snowflake/llama3.1-70b",
            optional_params={},
            litellm_params={},
        )
        assert url.endswith("/api/v2/cortex/v1/chat/completions")
        assert "cortex/inference:complete" not in url

    def test_url_never_uses_legacy_endpoint(self):
        url = self.cfg.get_complete_url(
            api_base=API_BASE,
            api_key=JWT_TOKEN,
            model="snowflake/llama3.1-70b",
            optional_params={},
            litellm_params={},
        )
        assert "inference:complete" not in url
        assert "/v1/chat/completions" in url

    def test_url_works_for_claude_models(self):
        url = self.cfg.get_complete_url(
            api_base=API_BASE,
            api_key=JWT_TOKEN,
            model="snowflake/claude-sonnet-4-5",
            optional_params={},
            litellm_params={},
        )
        assert "/cortex/v1/messages" in url

    def test_url_works_for_llama_models(self):
        url = self.cfg.get_complete_url(
            api_base=API_BASE,
            api_key=JWT_TOKEN,
            model="snowflake/llama3.1-70b",
            optional_params={},
            litellm_params={},
        )
        assert "/cortex/v1/chat/completions" in url


class TestSnowflakeConfigAuth:
    def setup_method(self):
        self.cfg = SnowflakeConfig()

    def test_pat_auth_strips_prefix_and_sets_header(self):
        headers = self.cfg.validate_environment(
            headers={},
            model="snowflake/llama3.1-70b",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=PAT_TOKEN,
        )
        assert headers["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
        assert headers["Authorization"] == "Bearer my-secret-pat-token"

    def test_jwt_auth_sets_keypair_header(self):
        headers = self.cfg.validate_environment(
            headers={},
            model="snowflake/llama3.1-70b",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=JWT_TOKEN,
        )
        assert headers["X-Snowflake-Authorization-Token-Type"] == "KEYPAIR_JWT"
        assert headers["Authorization"] == f"Bearer {JWT_TOKEN}"

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="Missing Snowflake JWT key"):
            self.cfg.validate_environment(
                headers={},
                model="snowflake/llama3.1-70b",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )


class TestSnowflakeConfigRequest:
    def setup_method(self):
        self.cfg = SnowflakeConfig()
        self.messages = [{"role": "user", "content": "hello"}]

    def test_request_uses_openai_tool_format(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        body = self.cfg.transform_request(
            model="snowflake/llama3.1-70b",
            messages=self.messages,
            optional_params={"tools": tools},
            litellm_params={},
            headers={},
        )
        assert body["tools"] == tools
        assert "tool_spec" not in json.dumps(body)

    def test_stream_defaults_to_false(self):
        body = self.cfg.transform_request(
            model="snowflake/llama3.1-70b",
            messages=self.messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["stream"] is False

    def test_stream_true_passes_through(self):
        body = self.cfg.transform_request(
            model="snowflake/llama3.1-70b",
            messages=self.messages,
            optional_params={"stream": True},
            litellm_params={},
            headers={},
        )
        assert body["stream"] is True

    def test_supported_params_includes_stream(self):
        params = self.cfg.get_supported_openai_params("snowflake/llama3.1-70b")
        assert "stream" in params

    def test_no_content_list_in_request(self):
        body = self.cfg.transform_request(
            model="snowflake/llama3.1-70b",
            messages=self.messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "content_list" not in body


class TestSnowflakeConfigResponse:
    def setup_method(self):
        self.cfg = SnowflakeConfig()

    def test_standard_response_parsed(self):
        raw = _make_openai_response("Hello from Snowflake!")
        result = self.cfg.transform_response(
            model="snowflake/llama3.1-70b",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=_mock_logging(),
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.choices[0].message.content == "Hello from Snowflake!"
        assert result.model.startswith("snowflake/")

    def test_model_prefixed_with_snowflake(self):
        raw = _make_openai_response()
        result = self.cfg.transform_response(
            model="snowflake/llama3.1-70b",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=_mock_logging(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.model.startswith("snowflake/")


# ─── SnowflakeConfig ────────────────────────────────────────

class TestAnthropicConfigURL:
    def setup_method(self):
        self.cfg = SnowflakeConfig()

    def test_url_routes_to_messages_endpoint(self):
        url = self.cfg.get_complete_url(
            api_base=API_BASE,
            api_key=PAT_TOKEN,
            model="snowflake/claude-sonnet-4-5",
            optional_params={},
            litellm_params={},
        )
        assert url.endswith("/api/v2/cortex/v1/messages")
        assert "chat/completions" not in url
        assert "inference:complete" not in url

    def test_url_with_account_id(self):
        url = self.cfg.get_complete_url(
            api_base=None,
            api_key=PAT_TOKEN,
            model="snowflake/claude-sonnet-4-5",
            optional_params={"account_id": ACCOUNT_ID},
            litellm_params={},
        )
        assert f"https://{ACCOUNT_ID}.snowflakecomputing.com/api/v2/cortex/v1/messages" == url


class TestAnthropicConfigAuth:
    def setup_method(self):
        self.cfg = SnowflakeConfig()

    def test_anthropic_version_header_set(self):
        headers = self.cfg.validate_environment(
            headers={},
            model="snowflake/claude-sonnet-4-5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=PAT_TOKEN,
        )
        assert headers["anthropic-version"] == "2023-06-01"

    def test_pat_auth_and_anthropic_version_combined(self):
        headers = self.cfg.validate_environment(
            headers={},
            model="snowflake/claude-sonnet-4-5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=PAT_TOKEN,
        )
        assert headers["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "Bearer" in headers["Authorization"]


class TestAnthropicConfigRequest:
    def setup_method(self):
        self.cfg = SnowflakeConfig()

    def test_system_message_extracted_to_top_level(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["system"] == "You are helpful."
        assert all(m["role"] != "system" for m in body["messages"])
        assert body["messages"][0] == {"role": "user", "content": "Hello"}

    def test_model_prefix_stripped(self):
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["model"] == "claude-sonnet-4-5"
        assert "snowflake/" not in body["model"]

    def test_max_tokens_defaulted_when_missing(self):
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "max_tokens" in body
        assert body["max_tokens"] == 4096

    def test_max_tokens_not_overridden_when_provided(self):
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"max_tokens": 500},
            litellm_params={},
            headers={},
        )
        assert body["max_tokens"] == 500

    def test_no_system_key_when_no_system_message(self):
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert "system" not in body


class TestAnthropicConfigResponse:
    def setup_method(self):
        self.cfg = SnowflakeConfig()

    def test_anthropic_response_to_openai_format(self):
        raw = _make_anthropic_response("Hi there!")
        result = self.cfg.transform_response(
            model="snowflake/claude-sonnet-4-5",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=_mock_logging(),
            request_data={},
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.choices[0].message.content == "Hi there!"
        assert result.choices[0].finish_reason == "stop"

    def test_usage_tokens_mapped(self):
        raw = _make_anthropic_response()
        result = self.cfg.transform_response(
            model="snowflake/claude-sonnet-4-5",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=_mock_logging(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_stop_reason_end_turn_maps_to_stop(self):
        raw = _make_anthropic_response()
        result = self.cfg.transform_response(
            model="snowflake/claude-sonnet-4-5",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=_mock_logging(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.choices[0].finish_reason == "stop"

    def test_tool_use_block_mapped_to_tool_calls(self):
        body = {
            "id": "msg_tool",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
        raw = httpx.Response(200, json=body)
        result = self.cfg.transform_response(
            model="snowflake/claude-sonnet-4-5",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=_mock_logging(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.choices[0].finish_reason == "tool_calls"
        tool_calls = result.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"city": "Paris"}


# ─── Model detection helper ────────────────────────────────────────────────

class TestIsClaudeModel:
    def test_claude_model_detected(self):
        assert _is_claude_model("snowflake/claude-sonnet-4-5") is True
        assert _is_claude_model("claude-3-haiku") is True
        assert _is_claude_model("snowflake/claude-opus-4") is True

    def test_non_claude_not_detected(self):
        assert _is_claude_model("snowflake/llama3.1-70b") is False
        assert _is_claude_model("snowflake/mistral-large") is False
        assert _is_claude_model("snowflake/deepseek-r1") is False
        assert _is_claude_model("snowflake/snowflake-arctic") is False


# ─── Anthropic Tool Transformation Tests ──────────────────────────────────

class TestAnthropicToolTransformation:
    def setup_method(self):
        self.cfg = SnowflakeConfig()

    def test_openai_tools_converted_to_anthropic_format(self):
        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=messages,
            optional_params={"tools": tools},
            litellm_params={},
            headers={},
        )
        assert len(body["tools"]) == 1
        tool = body["tools"][0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get current weather"
        assert "input_schema" in tool
        assert tool["input_schema"]["properties"]["city"]["type"] == "string"
        assert "function" not in tool
        assert "type" not in tool

    def test_tools_already_in_anthropic_format_pass_through(self):
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"name": "my_tool", "input_schema": {"type": "object", "properties": {}}}]
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=messages,
            optional_params={"tools": tools},
            litellm_params={},
            headers={},
        )
        assert body["tools"] == tools


class TestAnthropicMultiTurnToolMessages:
    def setup_method(self):
        self.cfg = SnowflakeConfig()

    def test_assistant_tool_calls_converted_to_tool_use_blocks(self):
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Sunny, 22°C",
            },
            {"role": "user", "content": "Thanks!"},
        ]
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        msgs = body["messages"]
        assert msgs[0] == {"role": "user", "content": "What's the weather in Paris?"}

        assistant_msg = msgs[1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        assert assistant_msg["content"][0]["type"] == "tool_use"
        assert assistant_msg["content"][0]["id"] == "call_123"
        assert assistant_msg["content"][0]["name"] == "get_weather"
        assert assistant_msg["content"][0]["input"] == {"city": "Paris"}

        tool_result_msg = msgs[2]
        assert tool_result_msg["role"] == "user"
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "call_123"
        assert tool_result_msg["content"][0]["content"] == "Sunny, 22°C"

        assert msgs[3] == {"role": "user", "content": "Thanks!"}

    def test_assistant_with_text_and_tool_calls(self):
        messages = [
            {"role": "user", "content": "Check weather"},
            {
                "role": "assistant",
                "content": "Let me check that for you.",
                "tool_calls": [
                    {
                        "id": "call_456",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }
                ],
            },
        ]
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assistant_msg = body["messages"][1]
        assert assistant_msg["content"][0] == {"type": "text", "text": "Let me check that for you."}
        assert assistant_msg["content"][1]["type"] == "tool_use"
        assert assistant_msg["content"][1]["name"] == "get_weather"

    def test_tool_role_never_in_output(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        for msg in body["messages"]:
            assert msg["role"] != "tool"

    def test_malformed_json_in_tool_arguments_handled_gracefully(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "broken_tool", "arguments": "not valid json{{{"},
                    }
                ],
            },
        ]
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assistant_msg = body["messages"][1]
        tool_use_block = assistant_msg["content"][0]
        assert tool_use_block["type"] == "tool_use"
        assert tool_use_block["name"] == "broken_tool"
        assert tool_use_block["input"] == {}

    def test_non_string_tool_arguments_pass_through(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_dict",
                        "type": "function",
                        "function": {"name": "dict_tool", "arguments": {"already": "parsed"}},
                    }
                ],
            },
        ]
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        tool_use_block = body["messages"][1]["content"][0]
        assert tool_use_block["input"] == {"already": "parsed"}

    def test_tool_result_with_non_string_content(self):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": {"result_key": "result_value"}},
        ]
        body = self.cfg.transform_request(
            model="snowflake/claude-sonnet-4-5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        tool_result = body["messages"][2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert json.loads(tool_result["content"]) == {"result_key": "result_value"}
