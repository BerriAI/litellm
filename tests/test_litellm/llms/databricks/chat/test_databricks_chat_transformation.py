import json
from typing import Final

import httpx
import respx

import litellm


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
    assert parsed.choices[0].delta.reasoning_content == "We need answer"
    assert parsed.choices[0].delta.content is None
    assert parsed.choices[0].delta.reasoning_content == "We need answer"
    assert parsed.choices[0].delta.content is None


def test_get_optional_params_passes_service_tier_through() -> None:
    optional_params = litellm.utils.get_optional_params(
        model="databricks-claude-opus-5",
        custom_llm_provider="databricks",
        service_tier="priority",
        drop_params=True,
    )

    assert optional_params["service_tier"] == "priority"


def test_chunk_parser_carries_service_tier() -> None:
    iterator = DatabricksChatResponseIterator(None, sync_stream=True)
    chunk = {
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "lit-qa-deepseek-v4-flash",
        "service_tier": "priority",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    parsed = iterator.chunk_parser(chunk)

    assert parsed.service_tier == "priority"
