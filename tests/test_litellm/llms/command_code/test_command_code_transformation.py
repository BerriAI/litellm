"""
Unit tests for the Command Code chat transformation.

Fully mocked - no real network calls.
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from unittest.mock import MagicMock

from litellm.llms.command_code.chat.transformation import CommandCodeConfig
from litellm.llms.command_code.common_utils import (
    COMMAND_CODE_CLI_VERSION,
    COMMAND_CODE_DEFAULT_MAX_TOKENS,
    CommandCodeError,
    map_command_code_finish_reason,
)
from litellm.types.utils import ModelResponse


@pytest.fixture
def config() -> CommandCodeConfig:
    return CommandCodeConfig()


class TestTransformRequest:
    def test_request_body_shape(self, config):
        request = config.transform_request(
            model="command_code/deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={"stream": True, "max_tokens": 100},
            litellm_params={},
            headers={},
        )

        assert set(request.keys()) == {"config", "memory", "taste", "skills", "threadId", "params"}
        config_block = request["config"]
        assert config_block["workingDir"] == "/tmp"
        assert config_block["environment"] == "terminal"
        assert config_block["structure"] == []
        assert config_block["isGitRepo"] is False
        assert config_block["currentBranch"] == ""
        assert config_block["mainBranch"] == ""
        assert config_block["gitStatus"] == ""
        assert config_block["recentCommits"] == []
        assert request["memory"] is None
        assert request["taste"] is None
        assert request["skills"] is None
        assert isinstance(request["threadId"], str) and request["threadId"]

        params = request["params"]
        assert params["model"] == "deepseek/deepseek-v4-flash"  # provider prefix stripped
        assert params["messages"] == [{"role": "user", "content": "hello"}]
        assert params["max_tokens"] == 100
        assert params["stream"] is True

    def test_max_tokens_defaults(self, config):
        request = config.transform_request(
            model="command_code/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert request["params"]["max_tokens"] == COMMAND_CODE_DEFAULT_MAX_TOKENS
        assert "temperature" not in request["params"]

    def test_temperature_passthrough(self, config):
        request = config.transform_request(
            model="command_code/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"temperature": 0.7},
            litellm_params={},
            headers={},
        )
        assert request["params"]["temperature"] == 0.7

    def test_system_messages_flattened(self, config):
        request = config.transform_request(
            model="command_code/gpt-5.5",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Be brief."},
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )
        params = request["params"]
        assert params["system"] == "You are helpful.\n\nBe brief."
        assert all(m["role"] != "system" for m in params["messages"])
        assert len(params["messages"]) == 1

    def test_assistant_message_with_text_reasoning_and_tool_call(self, config):
        messages = [
            {"role": "user", "content": "weather in sf?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "reasoning_content": "The user wants weather.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "sf"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        request = config.transform_request(
            model="command_code/gpt-5.5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assistant = request["params"]["messages"][1]
        assert assistant["role"] == "assistant"
        assert {"type": "reasoning", "text": "The user wants weather."} in assistant["content"]
        assert {"type": "text", "text": "Let me check."} in assistant["content"]
        tool_call_parts = [p for p in assistant["content"] if p["type"] == "tool-call"]
        assert tool_call_parts == [
            {
                "type": "tool-call",
                "toolCallId": "call_1",
                "toolName": "get_weather",
                "input": {"city": "sf"},
            }
        ]

    def test_tool_result_message(self, config):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "sunny"},
        ]
        request = config.transform_request(
            model="command_code/gpt-5.5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        tool_message = request["params"]["messages"][1]
        assert tool_message == {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "call_1",
                    "toolName": "get_weather",
                    "output": {"type": "text", "value": "sunny"},
                }
            ],
        }

    def test_orphan_tool_calls_and_results_dropped(self, config):
        messages = [
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    {
                        "id": "call_orphan",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_unknown", "content": "stale result"},
            {"role": "user", "content": "hi"},
        ]
        request = config.transform_request(
            model="command_code/gpt-5.5",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        cc_messages = request["params"]["messages"]
        # orphan tool call dropped from assistant parts, orphan result dropped entirely
        assert all(m["role"] != "tool" for m in cc_messages)
        assistant = cc_messages[0]
        assert all(p["type"] != "tool-call" for p in assistant["content"])

    def test_tool_schema_uses_input_schema(self, config):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        request = config.transform_request(
            model="command_code/gpt-5.5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"tools": tools},
            litellm_params={},
            headers={},
        )
        assert request["params"]["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]


class TestValidateEnvironment:
    def test_sets_required_headers(self, config):
        headers = config.validate_environment(
            headers={},
            model="command_code/gpt-5.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-test",
        )
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["x-command-code-version"] == COMMAND_CODE_CLI_VERSION
        assert headers["x-cli-environment"] == "production"
        assert headers["Content-Type"] == "application/json"

    def test_missing_api_key_raises(self, config, monkeypatch):
        monkeypatch.delenv("COMMANDCODE_API_KEY", raising=False)
        with pytest.raises(CommandCodeError) as excinfo:
            config.validate_environment(
                headers={},
                model="command_code/gpt-5.5",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )
        assert excinfo.value.status_code == 401

    def test_api_key_from_env(self, config, monkeypatch):
        monkeypatch.setenv("COMMANDCODE_API_KEY", "sk-env")
        headers = config.validate_environment(
            headers={},
            model="command_code/gpt-5.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert headers["Authorization"] == "Bearer sk-env"


class TestGetCompleteUrl:
    def test_default_api_base(self, config, monkeypatch):
        monkeypatch.delenv("COMMANDCODE_API_BASE", raising=False)
        url = config.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="command_code/gpt-5.5",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.commandcode.ai/alpha/generate"

    def test_custom_api_base(self, config):
        url = config.get_complete_url(
            api_base="https://example.com/",
            api_key="sk-test",
            model="command_code/gpt-5.5",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://example.com/alpha/generate"


class TestSupportedParams:
    def test_supported_openai_params(self, config):
        supported = config.get_supported_openai_params(model="command_code/gpt-5.5")
        assert set(supported) == {"stream", "max_tokens", "max_completion_tokens", "temperature", "tools"}

    def test_map_openai_params(self, config):
        optional_params = config.map_openai_params(
            non_default_params={
                "stream": True,
                "max_completion_tokens": 42,
                "temperature": 0.1,
                "tools": [{"type": "function", "function": {"name": "f"}}],
            },
            optional_params={},
            model="command_code/gpt-5.5",
            drop_params=False,
        )
        assert optional_params["stream"] is True
        assert optional_params["max_tokens"] == 42
        assert optional_params["temperature"] == 0.1
        assert optional_params["tools"] == [{"type": "function", "function": {"name": "f"}}]


class TestStreamProperties:
    def test_has_custom_stream_wrapper(self, config):
        assert config.has_custom_stream_wrapper is True

    def test_stream_param_not_in_top_level_body(self, config):
        # stream lives in params.stream, set by transform_request
        assert config.supports_stream_param_in_request_body is False

    def test_should_fake_stream(self, config):
        assert config.should_fake_stream(model="command_code/gpt-5.5", stream=True) is False


class TestTransformResponse:
    def _raw_response(self, lines) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content="\n".join(lines).encode("utf-8"),
            request=httpx.Request("POST", "https://api.commandcode.ai/alpha/generate"),
        )

    def test_assembles_full_response_from_event_stream(self, config):
        raw_response = self._raw_response(
            [
                json.dumps({"type": "reasoning-delta", "text": "thinking "}),
                json.dumps({"type": "reasoning-delta", "text": "hard"}),
                json.dumps({"type": "text-delta", "text": "Hello"}),
                json.dumps({"type": "text-delta", "text": " world"}),
                json.dumps(
                    {
                        "type": "finish",
                        "finishReason": "stop",
                        "totalUsage": {
                            "inputTokens": 10,
                            "outputTokens": 5,
                            "inputTokenDetails": {"cacheReadTokens": 3, "cacheWriteTokens": 2},
                        },
                    }
                ),
            ]
        )
        result = config.transform_response(
            model="command_code/gpt-5.5",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        message = result.choices[0].message
        assert message.content == "Hello world"
        assert message.reasoning_content == "thinking hard"
        assert result.choices[0].finish_reason == "stop"
        usage = result.usage
        assert usage.prompt_tokens == 15  # 10 input + 3 cache read + 2 cache write
        assert usage.completion_tokens == 5
        assert usage.cache_read_input_tokens == 3
        assert usage.cache_creation_input_tokens == 2

    def test_tool_call_response(self, config):
        raw_response = self._raw_response(
            [
                json.dumps(
                    {
                        "type": "tool-call",
                        "toolCallId": "call_1",
                        "toolName": "get_weather",
                        "input": {"city": "sf"},
                    }
                ),
                json.dumps({"type": "finish", "finishReason": "tool-calls"}),
            ]
        )
        result = config.transform_response(
            model="command_code/gpt-5.5",
            raw_response=raw_response,
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.choices[0].finish_reason == "tool_calls"
        tool_calls = result.choices[0].message.tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {"city": "sf"}

    def test_error_event_raises(self, config):
        raw_response = self._raw_response([json.dumps({"type": "error", "error": {"message": "quota exceeded"}})])
        with pytest.raises(CommandCodeError, match="quota exceeded"):
            config.transform_response(
                model="command_code/gpt-5.5",
                raw_response=raw_response,
                model_response=ModelResponse(),
                logging_obj=MagicMock(),
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )


class TestFinishReasonMapping:
    @pytest.mark.parametrize(
        "command_code_reason,expected",
        [
            ("tool-calls", "tool_calls"),
            ("length", "length"),
            ("max_tokens", "length"),
            ("max-tokens", "length"),
            ("max_output_tokens", "length"),
            ("stop", "stop"),
            ("some-unknown-reason", "stop"),
            (None, "stop"),
        ],
    )
    def test_mapping(self, command_code_reason, expected):
        assert map_command_code_finish_reason(command_code_reason) == expected


class TestGetErrorClass:
    def test_returns_command_code_error(self, config):
        error = config.get_error_class(error_message="boom", status_code=429, headers={})
        assert isinstance(error, CommandCodeError)
        assert error.status_code == 429
