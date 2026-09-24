"""The shared Responses API config contract."""

import pytest

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


def test_default_merge_extra_body_shallow_merges_and_lets_extra_body_replace_nested_metadata():
    cfg = OpenAIResponsesAPIConfig()
    request = {"model": "m", "input": "hi", "metadata": {"from_request": "1"}, "max_output_tokens": 5}
    extra_body = {"metadata": {"from_extra_body": "2"}, "vendor_only_field": "x"}

    assert cfg.merge_extra_body(dict(request), extra_body) == {
        "model": "m",
        "input": "hi",
        "metadata": {"from_extra_body": "2"},
        "max_output_tokens": 5,
        "vendor_only_field": "x",
    }
    assert cfg.merge_extra_body(dict(request), None) == request
    assert cfg.merge_extra_body(dict(request), {}) == request
