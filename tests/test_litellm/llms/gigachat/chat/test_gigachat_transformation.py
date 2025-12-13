import json
import os
import sys
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.gigachat.chat.transformation import GigaChatConfig
from litellm.llms.gigachat.common_utils import BaseGigaChat


@pytest.fixture()
def gigachat_config(monkeypatch):
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "true")
    return GigaChatConfig()


def test_is_token_expired_seconds(monkeypatch):
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "true")
    base = BaseGigaChat()
    fixed_now = 1_700_000_000
    monkeypatch.setattr(
        "litellm.llms.gigachat.common_utils.time.time", lambda: fixed_now
    )

    base._token_cache = {"token": "cached", "expires_at": fixed_now + 60}
    assert base._is_token_expired() is False

    base._token_cache["expires_at"] = fixed_now - 1
    assert base._is_token_expired() is True


def test_is_token_expired_milliseconds(monkeypatch):
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "true")
    base = BaseGigaChat()
    fixed_now = 1_700_000_000
    monkeypatch.setattr(
        "litellm.llms.gigachat.common_utils.time.time", lambda: fixed_now
    )

    base._token_cache = {"token": "cached", "expires_at": (fixed_now * 1000) + 5_000}
    assert base._is_token_expired() is False

    base._token_cache["expires_at"] = (fixed_now * 1000) - 1
    assert base._is_token_expired() is True


def test_transform_messages_role_and_tool_conversion(gigachat_config):
    messages = [
        {"role": "developer", "content": "system prompt"},
        {"role": "system", "content": "second system message"},
        {"role": "tool", "content": {"result": 42}},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tool_1",
                    "function": {
                        "name": "lookup",
                        "arguments": json.dumps({"city": "Rome"}),
                    },
                }
            ],
        },
    ]

    transformed = gigachat_config._transform_messages(deepcopy(messages), headers={})

    assert transformed[0]["role"] == "system"
    assert transformed[1]["role"] == "user"
    assert transformed[2]["role"] == "function"
    assert transformed[2]["content"] == json.dumps({"result": 42}, ensure_ascii=False)
    assert transformed[3]["content"] == ""
    assert transformed[3]["function_call"]["arguments"] == {"city": "Rome"}


def test_limit_attachments_total_cap(gigachat_config):
    messages = [
        {"role": "user", "content": "", "attachments": ["a1", "a2", "a3"]},
        {"role": "assistant", "content": "", "attachments": ["b1", "b2", "b3"]},
        {"role": "assistant", "content": "", "attachments": ["c1", "c2"]},
    ]

    gigachat_config._limit_attachments(messages, max_total_attachments=4)

    assert messages[0]["attachments"] == ["a1", "a2", "a3"]
    assert messages[1]["attachments"] == ["b1"]
    assert messages[2]["attachments"] == []


def test_process_content_parts_limits_uploads(gigachat_config, monkeypatch):
    uploaded = []

    def fake_upload(image_url, headers, filename=None):
        uploaded.append((image_url, filename))
        return f"id_{len(uploaded)}"

    monkeypatch.setattr(gigachat_config, "upload_file", fake_upload)

    content_parts = [
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": {"url": "https://img/one.png"}},
        {"type": "text", "text": "second"},
        {"type": "image_url", "image_url": {"url": "https://img/two.png"}},
        {
            "type": "file",
            "file": {"filename": "doc.txt", "file_data": "data:text/plain;base64,AAA"},
        },
    ]

    texts, attachments = gigachat_config._process_content_parts(
        content_parts, headers={"Authorization": "Bearer test"}
    )

    assert texts == ["first", "second"]
    assert attachments == ["id_1", "id_2"]
    assert len(uploaded) == 3


def test_transform_request_maps_openai_params(gigachat_config):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "desc",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    response_format = {"json_schema": {"schema": {"type": "object", "properties": {}}}}

    request_body = gigachat_config.transform_request(
        model="gigachat-large",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "max_tokens": 256,
            "temperature": 0.4,
            "top_p": 0.9,
            "stream": True,
            "tools": tools,
            "response_format": response_format,
        },
        litellm_params={},
        headers={},
    )

    assert request_body["model"] == "gigachat-large"
    assert request_body["messages"][0]["role"] == "user"
    assert request_body["max_tokens"] == 256
    assert request_body["temperature"] == 0.4
    assert request_body["top_p"] == 0.9
    assert request_body["stream"] is True
    assert request_body["functions"][0]["name"] == "get_weather"
    assert request_body["response_format"]["type"] == "json_schema"


def test_transform_response_converts_function_calls(monkeypatch, gigachat_config):
    response_payload = {
        "id": "resp",
        "object": "chat.completion",
        "model": "gigachat-large",
        "choices": [
            {
                "index": 0,
                "finish_reason": None,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "lookup",
                        "arguments": {"location": "Rome"},
                    },
                    "functions_state_id": "state-1",
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    raw_response = MagicMock()
    raw_response.json.return_value = response_payload

    mock_model_response = MagicMock(return_value="transformed")
    monkeypatch.setattr(
        "litellm.llms.gigachat.chat.transformation.ModelResponse", mock_model_response
    )

    result = gigachat_config.transform_response(
        model="gigachat-large",
        raw_response=raw_response,
        model_response=None,
        logging_obj=MagicMock(),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key=None,
        json_mode=None,
    )

    assert result == "transformed"
    transformed_payload = mock_model_response.call_args.kwargs

    choice = transformed_payload["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert "tool_calls" in message
    assert "function_call" not in message
    assert message["tool_calls"][0]["function"]["arguments"] == '{"location": "Rome"}'
    assert (
        transformed_payload["usage"]["input_tokens_details"]["cached_tokens"] == 0
    )
    assert (
        transformed_payload["usage"]["output_tokens_details"]["reasoning_tokens"] == 0
    )

