"""
Unit tests for GigaChat chat transformation.

Tests GigaChatConfig covering get_complete_url, validate_environment,
get_supported_openai_params, map_openai_params, _convert_tools_to_functions,
_map_tool_choice, _transform_messages, transform_request, transform_response,
get_model_response_iterator, and get_error_class.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.gigachat.chat.transformation import (
    GigaChatConfig,
    GigaChatError,
    is_valid_json,
)
from litellm.types.utils import ModelResponse, Usage

TRANSFORM_MODULE = "litellm.llms.gigachat.chat.transformation"


def _make_httpx_response(
    body: dict, status_code: int = 200
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
        request=httpx.Request(
            "POST",
            "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
        ),
    )


# ---------------------------------------------------------------------------
# is_valid_json
# ---------------------------------------------------------------------------


class TestIsValidJson:
    def test_valid_json_object(self):
        assert is_valid_json('{"key": "value"}') is True

    def test_valid_json_array(self):
        assert is_valid_json("[1, 2, 3]") is True

    def test_valid_json_string(self):
        assert is_valid_json('"hello"') is True

    def test_invalid_json(self):
        assert is_valid_json("{invalid}") is False

    def test_empty_string(self):
        assert is_valid_json("") is False


# ---------------------------------------------------------------------------
# GigaChatConfig
# ---------------------------------------------------------------------------


class TestGetCompleteUrl:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_uses_api_base_from_param(self):
        url = self.config.get_complete_url(
            api_base="https://custom.example.com",
            api_key=None,
            model="GigaChat",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == "https://custom.example.com/chat/completions"

    def test_uses_api_base_with_trailing_slash(self):
        url = self.config.get_complete_url(
            api_base="https://custom.example.com/",
            api_key=None,
            model="GigaChat",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        # get_api_base passes the value through without stripping the slash
        assert url == "https://custom.example.com//chat/completions"

    def test_uses_api_base_from_get_api_base_when_none(self):
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="GigaChat",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url.endswith("/chat/completions")


class TestValidateEnvironment:
    def setup_method(self):
        self.config = GigaChatConfig()

    @patch(f"{TRANSFORM_MODULE}.get_access_token", return_value="test-token")
    @patch(f"{TRANSFORM_MODULE}.get_secret_str", return_value=None)
    def test_sets_auth_headers(self, mock_get_secret, mock_get_token):
        headers: dict = {}
        result = self.config.validate_environment(
            headers=headers,
            model="GigaChat",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="creds",
            api_base="https://api.example.com",
        )
        assert result["Authorization"] == "Bearer test-token"
        assert result["Content-Type"] == "application/json"
        assert result["Accept"] == "application/json"

    @patch(f"{TRANSFORM_MODULE}.get_access_token", return_value="token")
    @patch(f"{TRANSFORM_MODULE}.get_secret_str", return_value=None)
    def test_stores_credentials_and_api_base_for_image_uploads(
        self, mock_get_secret, mock_get_token
    ):
        self.config.validate_environment(
            headers={},
            model="GigaChat",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="my-creds",
            api_base="https://my-api.example.com",
        )
        assert self.config._current_credentials == "my-creds"
        assert self.config._current_api_base == "https://my-api.example.com"

    @patch(f"{TRANSFORM_MODULE}.get_access_token", return_value="token")
    @patch(f"{TRANSFORM_MODULE}.get_secret_str")
    def test_falls_back_to_env_for_credentials(  # test-quality-ok: mock-echo of internal wiring
        self, mock_get_secret, mock_get_token
    ):
        mock_get_secret.return_value = "env-creds"
        self.config.validate_environment(
            headers={},
            model="GigaChat",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )
        mock_get_secret.assert_any_call("GIGACHAT_CREDENTIALS")  # test-quality-ok: mock-echo of internal wiring


class TestGetSupportedOpenAiParams:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_returns_expected_params(self):
        params = self.config.get_supported_openai_params("GigaChat")
        expected = [
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "tools",
            "tool_choice",
            "functions",
            "function_call",
            "response_format",
        ]
        assert params == expected


class TestMapOpenAiParams:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_stream(self):
        result = self.config.map_openai_params(
            non_default_params={"stream": True},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result["stream"] is True

    def test_temperature_zero_maps_to_top_p_zero(self):
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result["top_p"] == 0
        assert "temperature" not in result

    def test_temperature_non_zero(self):
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result["temperature"] == 0.7

    def test_top_p(self):
        result = self.config.map_openai_params(
            non_default_params={"top_p": 0.5},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result["top_p"] == 0.5

    def test_max_tokens(self):
        result = self.config.map_openai_params(
            non_default_params={"max_tokens": 100},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result["max_tokens"] == 100

    def test_max_completion_tokens(self):
        result = self.config.map_openai_params(
            non_default_params={"max_completion_tokens": 200},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result["max_tokens"] == 200

    def test_stop_is_dropped(self):
        result = self.config.map_openai_params(
            non_default_params={"stop": ["\n\n"]},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert "stop" not in result

    def test_tools_converted_to_functions(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
        result = self.config.map_openai_params(
            non_default_params={"tools": tools},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert "functions" in result
        assert result["functions"] == [
            {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}
        ]

    def test_tool_choice_auto(self):
        result = self.config.map_openai_params(
            non_default_params={"tool_choice": "auto"},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result.get("function_call") == "auto"

    def test_tool_choice_none(self):
        result = self.config.map_openai_params(
            non_default_params={"tool_choice": "none"},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result.get("function_call") == "none"

    def test_tool_choice_required(self):
        result = self.config.map_openai_params(
            non_default_params={"tool_choice": "required"},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result.get("function_call") == "auto"

    def test_tool_choice_dict(self):
        result = self.config.map_openai_params(
            non_default_params={
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                }
            },
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result.get("function_call") == {"name": "get_weather"}

    def test_functions(self):
        funcs = [{"name": "my_func", "description": "desc", "parameters": {}}]
        result = self.config.map_openai_params(
            non_default_params={"functions": funcs},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result["functions"] == funcs

    def test_function_call(self):
        result = self.config.map_openai_params(
            non_default_params={"function_call": {"name": "my_func"}},
            optional_params={},
            model="GigaChat",
            drop_params=False,
        )
        assert result["function_call"] == {"name": "my_func"}

    def test_response_format_json_schema(self):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
        result = self.config.map_openai_params(
            non_default_params={"response_format": response_format},
            optional_params={"functions": []},
            model="GigaChat",
            drop_params=False,
        )
        # Should add a function for the schema
        assert len(result["functions"]) == 1
        assert result["functions"][0]["name"] == "test_schema"
        assert result["function_call"] == {"name": "test_schema"}
        assert result["_structured_output"] is True


class TestConvertToolsToFunctions:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_converts_function_tools_only(self):
        tools = [
            {"type": "function", "function": {"name": "a", "description": "d", "parameters": {}}},
            {"type": "code_interpreter"},  # should be ignored
        ]
        result = self.config._convert_tools_to_functions(tools)
        assert len(result) == 1
        assert result[0]["name"] == "a"

    def test_empty_tools(self):
        assert self.config._convert_tools_to_functions([]) == []


class TestMapToolChoice:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_none(self):
        assert self.config._map_tool_choice("none") == "none"

    def test_auto(self):
        assert self.config._map_tool_choice("auto") == "auto"

    def test_required(self):
        assert self.config._map_tool_choice("required") == "auto"

    def test_dict_with_function(self):
        result = self.config._map_tool_choice(
            {"type": "function", "function": {"name": "get_weather"}}
        )
        assert result == {"name": "get_weather"}

    def test_dict_without_name(self):
        result = self.config._map_tool_choice(
            {"type": "function", "function": {}}
        )
        assert result is None

    def test_unknown_value(self):
        assert self.config._map_tool_choice("unknown") is None


class TestTransformMessages:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_developer_role_to_system(self):
        result = self.config._transform_messages(
            [{"role": "developer", "content": "be helpful"}]
        )
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "be helpful"

    def test_system_message_not_first_becomes_user(self):
        result = self.config._transform_messages([
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "instruction"},
        ])
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "instruction"

    def test_tool_role_to_function(self):
        result = self.config._transform_messages([
            {"role": "tool", "content": '{"result": "ok"}'}
        ])
        assert result[0]["role"] == "function"

    def test_tool_role_content_wraps_non_json(self):
        result = self.config._transform_messages([
            {"role": "tool", "content": "plain text"}
        ])
        assert result[0]["role"] == "function"
        assert is_valid_json(result[0]["content"])

    def test_none_content_becomes_empty_string(self):
        result = self.config._transform_messages([
            {"role": "user", "content": None}
        ])
        assert result[0]["content"] == ""

    def test_name_field_removed(self):
        result = self.config._transform_messages([
            {"role": "user", "content": "hi", "name": "John"}
        ])
        assert "name" not in result[0]

    def test_tool_calls_converted_to_function_call(self):
        result = self.config._transform_messages([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }
                ],
            }
        ])
        assert "tool_calls" not in result[0]
        assert result[0]["function_call"]["name"] == "get_weather"
        assert result[0]["function_call"]["arguments"] == {"city": "London"}

    def test_tool_calls_with_dict_arguments(self):
        result = self.config._transform_messages([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": {"query": "test"},
                        },
                    }
                ],
            }
        ])
        assert result[0]["function_call"]["arguments"] == {"query": "test"}

    def test_list_content_multimodal(self):
        content = [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/img.jpg"},
            },
        ]
        with patch.object(self.config, "_upload_image", return_value="file-123"):
            result = self.config._transform_messages([
                {"role": "user", "content": content}
            ])
        assert result[0]["content"] == "describe this"
        assert result[0]["attachments"] == ["file-123"]

    def test_list_content_with_image_url_string(self):
        content = [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": "https://example.com/img.jpg"},
        ]
        with patch.object(self.config, "_upload_image", return_value="file-456"):
            result = self.config._transform_messages([
                {"role": "user", "content": content}
            ])
        assert result[0]["content"] == "look"
        assert "file-456" in result[0]["attachments"]


class TestTransformRequest:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_builds_basic_request(self):
        body = self.config.transform_request(
            model="gigachat/GigaChat",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["model"] == "GigaChat"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["content"] == "hi"

    def test_model_prefix_stripped(self):
        body = self.config.transform_request(
            model="gigachat/GigaChat-Pro",
            messages=[{"role": "user", "content": "hello"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert body["model"] == "GigaChat-Pro"

    def test_includes_optional_params(self):
        body = self.config.transform_request(
            model="gigachat/GigaChat",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={
                "temperature": 0.5,
                "max_tokens": 100,
                "stream": True,
            },
            litellm_params={},
            headers={},
        )
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 100
        assert body["stream"] is True

    def test_includes_functions(self):
        body = self.config.transform_request(
            model="gigachat/GigaChat",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={
                "functions": [{"name": "my_func"}],
                "function_call": {"name": "my_func"},
            },
            litellm_params={},
            headers={},
        )
        assert body["functions"] == [{"name": "my_func"}]
        assert body["function_call"] == {"name": "my_func"}

    def test_skips_unsupported_params(self):
        body = self.config.transform_request(
            model="gigachat/GigaChat",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"n": 2, "user": "abc"},
            litellm_params={},
            headers={},
        )
        assert "n" not in body
        assert "user" not in body


class TestTransformResponse:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_basic_response(self):
        raw = _make_httpx_response({
            "id": "chatcmpl-123",
            "created": 1700000000,
            "model": "GigaChat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        })
        model_response = ModelResponse()
        result = self.config.transform_response(
            model="gigachat/GigaChat",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.choices[0].message.content == "Hello!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 5
        assert result.usage.total_tokens == 8

    def test_function_call_into_tool_calls(self):
        raw = _make_httpx_response({
            "id": "chatcmpl-456",
            "created": 1700000000,
            "model": "GigaChat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "get_weather",
                            "arguments": {"city": "Moscow"},
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        model_response = ModelResponse()
        result = self.config.transform_response(
            model="gigachat/GigaChat",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.choices[0].finish_reason == "tool_calls"
        tool_calls = result.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert '{"city": "Moscow"}' in tool_calls[0].function.arguments

    def test_function_call_structured_output(self):
        raw = _make_httpx_response({
            "id": "chatcmpl-789",
            "created": 1700000000,
            "model": "GigaChat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "test_schema",
                            "arguments": {"name": "John"},
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })
        model_response = ModelResponse()
        result = self.config.transform_response(
            model="gigachat/GigaChat",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={"_structured_output": True},
            litellm_params={},
            encoding=None,
        )
        # Structured output: function_call -> content
        assert result.choices[0].finish_reason == "stop"
        assert result.choices[0].message.content is not None
        assert '"name": "John"' in result.choices[0].message.content

    def test_function_call_string_arguments(self):
        raw = _make_httpx_response({
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "function_call": {
                            "name": "get_weather",
                            "arguments": '{"city": "Moscow"}',
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        model_response = ModelResponse()
        result = self.config.transform_response(
            model="gigachat/GigaChat",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        tc = result.choices[0].message.tool_calls[0]
        assert '{"city": "Moscow"}' in tc.function.arguments

    def test_cleans_up_gigachat_specific_fields(self):
        raw = _make_httpx_response({
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "functions_state_id": "some-state",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        model_response = ModelResponse()
        result = self.config.transform_response(
            model="gigachat/GigaChat",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        # functions_state_id should have been removed from the message data
        assert result.choices[0].message.content == "done"

    def test_raises_on_invalid_json(self):
        raw = httpx.Response(
            status_code=500,
            headers={"content-type": "text/plain"},
            content=b"not json",
            request=httpx.Request("POST", "https://example.com"),
        )
        model_response = ModelResponse()
        with pytest.raises(GigaChatError) as exc_info:
            self.config.transform_response(
                model="gigachat/GigaChat",
                raw_response=raw,
                model_response=model_response,
                logging_obj=MagicMock(),
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )
        assert "Invalid JSON response" in str(exc_info.value.message)

    def test_empty_choices(self):
        raw = _make_httpx_response({
            "choices": [],
            "usage": {},
        })
        model_response = ModelResponse()
        result = self.config.transform_response(
            model="gigachat/GigaChat",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.choices == []

    def test_function_call_with_non_dict_arguments(self):
        raw = _make_httpx_response({
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "function_call": {
                            "name": "say_hello",
                            "arguments": "hello",
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        model_response = ModelResponse()
        result = self.config.transform_response(
            model="gigachat/GigaChat",
            raw_response=raw,
            model_response=model_response,
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        tc = result.choices[0].message.tool_calls[0]
        assert tc.function.arguments == "hello"


class TestGetModelResponseIterator:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_returns_gigachat_iterator_sync(self):
        from litellm.llms.gigachat.chat.streaming import (
            GigaChatModelResponseIterator,
        )

        result = self.config.get_model_response_iterator(
            streaming_response=iter(["data"]),
            sync_stream=True,
            json_mode=False,
        )
        assert isinstance(result, GigaChatModelResponseIterator)


class TestGetErrorClass:
    def setup_method(self):
        self.config = GigaChatConfig()

    def test_returns_gigachat_error(self):
        error = self.config.get_error_class(
            error_message="something went wrong",
            status_code=400,
            headers={"x-request-id": "abc"},
        )
        assert isinstance(error, GigaChatError)
        assert error.status_code == 400
        assert error.message == "something went wrong"
        assert error.headers == {"x-request-id": "abc"}


class TestUploadImage:
    def setup_method(self):
        self.config = GigaChatConfig()

    @patch(f"{TRANSFORM_MODULE}.upload_file_sync", return_value="file-uploaded")
    def test_upload_image_success(self, mock_upload):
        self.config._current_credentials = "creds"
        self.config._current_api_base = "https://api.example.com"
        result = self.config._upload_image("https://example.com/img.jpg")
        assert result == "file-uploaded"
        mock_upload.assert_called_once_with(
            image_url="https://example.com/img.jpg",
            credentials="creds",
            api_base="https://api.example.com",
        )

    @patch(f"{TRANSFORM_MODULE}.upload_file_sync", side_effect=Exception("fail"))
    def test_upload_image_failure_returns_none(self, mock_upload):
        result = self.config._upload_image("https://example.com/img.jpg")
        assert result is None