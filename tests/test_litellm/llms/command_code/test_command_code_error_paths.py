"""
Error-path and edge-case tests for the command_code provider.

Fully mocked - no real network calls.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.llms.command_code.chat.transformation import CommandCodeConfig
from litellm.llms.command_code.common_utils import (
    CommandCodeError,
    parse_stream_event_line,
    parse_tool_call_input,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import ModelResponse

MODEL = "command_code/gpt-5.5"
URL = "https://api.commandcode.ai/alpha/generate"


@pytest.fixture
def config() -> CommandCodeConfig:
    return CommandCodeConfig()


class TestStreamWrapperErrorPaths:
    def test_sync_wrapper_raises_on_http_error(self, config):
        def fake_post(self, url, **kwargs):
            return httpx.Response(
                status_code=429,
                content=b"rate limited",
                request=httpx.Request("POST", url),
            )

        with patch.object(HTTPHandler, "post", fake_post):
            with pytest.raises(CommandCodeError) as excinfo:
                config.get_sync_custom_stream_wrapper(
                    model=MODEL,
                    custom_llm_provider="command_code",
                    logging_obj=MagicMock(),
                    api_base=URL,
                    headers={},
                    data={},
                    messages=[],
                )
        assert excinfo.value.status_code == 429

    @pytest.mark.asyncio
    async def test_async_wrapper_raises_on_http_error(self, config):
        async def fake_post(self, url, **kwargs):
            return httpx.Response(
                status_code=500,
                content=b"server error",
                request=httpx.Request("POST", url),
            )

        with patch.object(AsyncHTTPHandler, "post", fake_post):
            with pytest.raises(CommandCodeError) as excinfo:
                await config.get_async_custom_stream_wrapper(
                    model=MODEL,
                    custom_llm_provider="command_code",
                    logging_obj=MagicMock(),
                    api_base=URL,
                    headers={},
                    data={},
                    messages=[],
                )
        assert excinfo.value.status_code == 500

    def test_completion_http_error_surfaces(self):
        def fake_post(self, url, **kwargs):
            raise httpx.HTTPStatusError(
                "auth failed",
                request=httpx.Request("POST", url),
                response=httpx.Response(401, content=b"bad key", request=httpx.Request("POST", url)),
            )

        with patch.object(HTTPHandler, "post", fake_post):
            with pytest.raises(Exception):
                litellm.completion(
                    model=MODEL,
                    messages=[{"role": "user", "content": "hi"}],
                    api_key="sk-bad",
                )


class TestMessageContentShapes:
    def test_user_message_with_list_content(self, config):
        request = config.transform_request(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look at this"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                    ],
                }
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )
        user_message = request["params"]["messages"][0]
        # image part dropped, text part kept in Command Code shape
        assert user_message["content"] == [{"type": "text", "text": "look at this"}]

    def test_assistant_message_with_list_content(self, config):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}],
            },
        ]
        request = config.transform_request(
            model=MODEL,
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )
        assistant = request["params"]["messages"][1]
        assert {"type": "text", "text": "part one"} in assistant["content"]
        assert {"type": "text", "text": "part two"} in assistant["content"]

    def test_developer_role_flattened_into_system(self, config):
        request = config.transform_request(
            model=MODEL,
            messages=[
                {"role": "developer", "content": "dev instructions"},
                {"role": "user", "content": "hi"},
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert request["params"]["system"] == "dev instructions"

    def test_non_function_tools_skipped(self, config):
        request = config.transform_request(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"tools": [{"type": "web_search"}]},
            litellm_params={},
            headers={},
        )
        assert request["params"]["tools"] == []


class TestParsingEdgeCases:
    def test_parse_tool_call_input_invalid_json_string(self):
        assert parse_tool_call_input("{not json") == {}

    def test_parse_tool_call_input_non_dict_json(self):
        assert parse_tool_call_input("[1, 2]") == {}

    def test_parse_tool_call_input_none(self):
        assert parse_tool_call_input(None) == {}

    def test_parse_stream_event_line_non_object_json(self):
        assert parse_stream_event_line('["a", "b"]') is None

    def test_parse_stream_event_line_invalid_json(self):
        assert parse_stream_event_line("{broken") is None

    def test_transform_response_error_as_string(self, config):
        raw_response = httpx.Response(
            status_code=200,
            content=json.dumps({"type": "error", "error": "plain failure"}).encode(),
            request=httpx.Request("POST", URL),
        )
        with pytest.raises(CommandCodeError, match="plain failure"):
            config.transform_response(
                model=MODEL,
                raw_response=raw_response,
                model_response=ModelResponse(),
                logging_obj=MagicMock(),
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )


class TestApiBaseResolution:
    def test_api_base_from_env(self, config, monkeypatch):
        monkeypatch.setenv("COMMANDCODE_API_BASE", "https://proxy.internal")
        url = config.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model=MODEL,
            optional_params={},
            litellm_params={},
        )
        assert url == "https://proxy.internal/alpha/generate"

    def test_api_key_from_litellm_global(self, monkeypatch):
        monkeypatch.delenv("COMMANDCODE_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "command_code_key", "sk-global")
        captured = {}

        def fake_post(self, url, **kwargs):
            captured["headers"] = kwargs.get("headers")
            return httpx.Response(
                status_code=200,
                content=json.dumps({"type": "finish", "finishReason": "stop"}).encode() + b"\n",
                request=httpx.Request("POST", url),
            )

        with patch.object(HTTPHandler, "post", fake_post):
            litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
            )
        assert captured["headers"]["Authorization"] == "Bearer sk-global"
