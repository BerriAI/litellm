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
