"""
Test that the 422 drop_params retry loop raises after exhaustion
instead of implicitly returning None.

When an OpenAI-compatible endpoint returns HTTP 422 with a body that has no
structured `detail` param list (e.g. content-moderation rejections),
`drop_params_from_unprocessable_entity_error` cannot drop anything, the
retry loop exhausts, and prior to the fix the handler fell off the end of
the function and returned None - which the proxy then serialized as
HTTP 200 with body `null`.
"""

import os
import sys
from unittest.mock import patch

import httpx
import openai
import pytest
from openai import AsyncOpenAI, OpenAI

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm


def _make_unprocessable_entity_error() -> openai.UnprocessableEntityError:
    """Build a 422 error whose body has no structured `detail` param list."""
    request = httpx.Request(
        method="POST", url="https://api.openai.com/v1/chat/completions"
    )
    response = httpx.Response(
        status_code=422,
        request=request,
        json={"error": {"message": "content moderation rejected the request"}},
    )
    return openai.UnprocessableEntityError(
        message="Error code: 422 - content moderation rejected the request",
        response=response,
        body={"message": "content moderation rejected the request"},
    )


@pytest.fixture
def enable_drop_params():
    """Enable drop_params globally, as `litellm_settings: drop_params: true` does on the proxy."""
    original = litellm.drop_params
    litellm.drop_params = True
    yield
    litellm.drop_params = original


@pytest.mark.asyncio
async def test_acompletion_raises_after_422_retry_exhaustion(enable_drop_params):
    """
    If every attempt raises a 422 that drop_params cannot fix,
    acompletion must raise - not return None.
    """
    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response,
        "create",
        side_effect=_make_unprocessable_entity_error(),
    ) as mock_create:
        with pytest.raises(Exception) as exc_info:
            await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                client=client,
            )

    # both attempts of the retry loop were consumed
    assert mock_create.call_count == 2
    # the surfaced error preserves the upstream 422
    assert getattr(exc_info.value, "status_code", None) == 422


def test_completion_raises_after_422_retry_exhaustion(enable_drop_params):
    """Same as above for the sync path."""
    client = OpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response,
        "create",
        side_effect=_make_unprocessable_entity_error(),
    ) as mock_create:
        with pytest.raises(Exception) as exc_info:
            litellm.completion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                client=client,
            )

    assert mock_create.call_count == 2
    assert getattr(exc_info.value, "status_code", None) == 422


@pytest.mark.asyncio
async def test_acompletion_streaming_raises_after_422_retry_exhaustion(
    enable_drop_params,
):
    """Same as above for the async streaming path (async_streaming retry loop)."""
    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response,
        "create",
        side_effect=_make_unprocessable_entity_error(),
    ) as mock_create:
        with pytest.raises(Exception) as exc_info:
            await litellm.acompletion(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                client=client,
            )

    assert mock_create.call_count == 2
    assert getattr(exc_info.value, "status_code", None) == 422
