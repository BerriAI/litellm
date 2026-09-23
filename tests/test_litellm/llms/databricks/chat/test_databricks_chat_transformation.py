import json
from typing import Final

import httpx
import respx

import litellm
from litellm.llms.databricks.chat.transformation import (
    DatabricksChatResponseIterator,
    DatabricksConfig,
)
from unittest.mock import MagicMock


def test_completion_merges_leading_system_and_developer_messages_for_chat_template_models(
    respx_mock: respx.MockRouter,
):
    upstream: Final = respx_mock.post("https://example.databricks.test/serving-endpoints/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "my-custom-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
            },
        )
    )

    response: Final = litellm.completion(
        model="databricks/my-custom-model",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "developer", "content": "Skills: none."},
            {"role": "user", "content": "Hello"},
        ],
        api_base="https://example.databricks.test/serving-endpoints",
        api_key="fake-databricks-api-key",
        num_retries=0,
    )

    assert upstream.call_count == 1
    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["messages"] == [
        {"role": "system", "content": "You are terse.\n\nSkills: none."},
        {"role": "user", "content": "Hello"},
    ]
    assert response.choices[0].message.content == "Answer"


def test_completion_merges_system_messages_when_one_has_empty_content(respx_mock: respx.MockRouter):
    upstream: Final = respx_mock.post("https://example.databricks.test/serving-endpoints/chat/completions").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "my-custom-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
            },
        )
    )

    litellm.completion(
        model="databricks/my-custom-model",
        messages=[
            {"role": "system", "content": "You are terse."},
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello"},
        ],
        api_base="https://example.databricks.test/serving-endpoints",
        api_key="fake-databricks-api-key",
        num_retries=0,
    )

    request_body: Final = json.loads(upstream.calls[0].request.read())
    assert request_body["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hello"},
    ]


def test_get_optional_params_passes_service_tier_through() -> None:
    optional_params: Final = litellm.utils.get_optional_params(
        model="databricks-claude-opus-5",
        custom_llm_provider="databricks",
        service_tier="priority",
        drop_params=True,
    )

    assert optional_params["service_tier"] == "priority"


def test_chunk_parser_carries_service_tier() -> None:
    iterator: Final = DatabricksChatResponseIterator(None, sync_stream=True)
    chunk: Final = {
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "lit-qa-deepseek-v4-flash",
        "service_tier": "priority",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    parsed: Final = iterator.chunk_parser(chunk)

    assert parsed.service_tier == "priority"


def test_transform_response_carries_service_tier() -> None:
    config: Final = DatabricksConfig()
    raw_response: Final = MagicMock()
    raw_response.json.return_value = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "dbrx",
        "service_tier": "priority",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    result: Final = config.transform_response(
        model="databricks/dbrx",
        raw_response=raw_response,
        model_response=litellm.ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result.service_tier == "priority"
