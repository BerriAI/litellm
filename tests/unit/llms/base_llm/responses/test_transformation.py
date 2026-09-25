"""The shared Responses API config contract."""

import json

import httpx
import pytest

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams


@pytest.mark.asyncio
async def test_default_async_transform_delegates_to_the_sync_transform():
    """A config that overrides only the sync transform gets the same request from the async hook,
    so the async handler can always await the hook."""
    cfg = OpenAIResponsesAPIConfig()
    input_with_cache_marker = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
        }
    ]
    sync_body = cfg.transform_responses_api_request(
        model="gpt-5",
        input=input_with_cache_marker,
        response_api_optional_request_params={"max_output_tokens": 64},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    async_body = await cfg.async_transform_responses_api_request(
        model="gpt-5",
        input=input_with_cache_marker,
        response_api_optional_request_params={"max_output_tokens": 64},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert async_body == sync_body
    assert "cache_control" not in async_body["input"][0]["content"][0]


def test_responses_sends_a_caller_extra_body_over_the_request_unchanged(respx_mock, monkeypatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "m",
                "output": [],
            },
        )
    )

    litellm.responses(
        model="openai/gpt-5",
        input="hi",
        api_key="sk-test",
        metadata={"a": "1"},
        extra_body={"foo": 1, "metadata": {"b": "2"}},
    )

    body = json.loads(route.calls.last.request.content)
    assert body["foo"] == 1
    assert body["metadata"] == {"b": "2"}
