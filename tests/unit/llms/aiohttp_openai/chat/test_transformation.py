from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientResponse
from pydantic import ValidationError

from litellm.llms.aiohttp_openai.chat.transformation import AiohttpOpenAIChatConfig
from litellm.types.utils import ModelResponse


async def _transform(body: object) -> ModelResponse:
    raw_response = Mock(spec=ClientResponse)
    raw_response.json = AsyncMock(return_value=body)
    return await AiohttpOpenAIChatConfig().transform_response(
        model="gpt-4o",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=Mock(),
        request_data={},
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )


async def test_transform_response_copies_the_openai_body_onto_the_model_response():
    response = await _transform(
        {
            "id": "chatcmpl-1",
            "created": 1700000000,
            "model": "gpt-4o-2024",
            "object": "chat.completion",
            "system_fingerprint": "fp_1",
            "choices": [
                {"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "Hi"}},
                {
                    "index": 1,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                        ],
                    },
                },
            ],
        }
    )

    assert response.id == "chatcmpl-1"
    assert response.created == 1700000000
    assert response.model == "gpt-4o-2024"
    assert response.object == "chat.completion"
    assert response.system_fingerprint == "fp_1"
    assert [(choice.index, choice.finish_reason, choice.message.content) for choice in response.choices] == [
        (0, "length", "Hi"),
        (1, "tool_calls", None),
    ]
    assert response.choices[1].message.tool_calls[0].function.name == "lookup"


async def test_transform_response_fills_choice_defaults_for_an_empty_choice():
    response = await _transform({"choices": [{}]})

    assert [(choice.index, choice.finish_reason, choice.message.role) for choice in response.choices] == [
        (0, "stop", "assistant")
    ]
    assert response.id is None


async def test_transform_response_returns_no_choices_for_an_empty_choices_list():
    response = await _transform({"id": "chatcmpl-1", "choices": []})

    assert response.choices == []
    assert response.id == "chatcmpl-1"


@pytest.mark.parametrize(
    "body",
    [
        {"id": "chatcmpl-1"},
        {"choices": None},
        {"choices": 7},
        {"choices": "secret-completion"},
        {"choices": {"message": "secret-completion"}},
        {"choices": ["secret-completion"]},
        {"choices": [{"index": 0}, ["secret-completion"]]},
    ],
)
async def test_transform_response_rejects_choices_that_are_not_a_list_of_objects_without_echoing_them(body: object):
    with pytest.raises(ValidationError) as exc_info:
        await _transform(body)

    assert "secret-completion" not in str(exc_info.value)


async def test_transform_response_rejects_a_choice_whose_message_is_not_an_object():
    with pytest.raises(ValidationError, match="validation error for Choices"):
        await _transform({"choices": [{"message": "text"}]})


async def test_transform_response_forwards_usage_including_prompt_tokens_details():
    """Regression test for https://github.com/BerriAI/litellm/issues/33967:
    the provider's usage - prompt_tokens_details.cached_tokens included -
    must reach the client response."""
    response = await _transform(
        {
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 10,
                "total_tokens": 2010,
                "prompt_tokens_details": {"cached_tokens": 1088},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        }
    )

    usage = response.usage
    assert usage.prompt_tokens == 2000
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 2010
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 1088
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 4


async def test_transform_response_without_usage_key_keeps_usage_unset():
    response = await _transform(
        {
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}],
        }
    )

    assert response.id == "chatcmpl-1"
    assert getattr(response, "usage", None) is None


async def test_transform_response_with_null_usage_keeps_usage_unset():
    response = await _transform(
        {
            "id": "chatcmpl-1",
            "usage": None,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}],
        }
    )

    assert response.id == "chatcmpl-1"
    assert getattr(response, "usage", None) is None


async def test_transform_response_coerces_null_token_values_to_zero():
    """Providers that send null token counts get the same 0 coercion the
    openai-like path applies in _sanitize_usage_obj."""
    response = await _transform(
        {
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "prompt_tokens_details": {"cached_tokens": 1088},
            },
        }
    )

    usage = response.usage
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 1088


async def test_transform_response_maps_provider_cache_extras_into_prompt_tokens_details():
    """Providers that report cache hits outside prompt_tokens_details
    (DeepSeek's prompt_cache_hit_tokens, Anthropic's cache_read_input_tokens)
    get the same Usage normalization every other chat path gets."""
    for extra in ({"prompt_cache_hit_tokens": 8}, {"cache_read_input_tokens": 7}):
        response = await _transform(
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, **extra},
            }
        )

        expected = extra[next(iter(extra))]
        assert response.usage.prompt_tokens_details is not None
        assert response.usage.prompt_tokens_details.cached_tokens == expected
