"""
Regression tests for issue #32221: with drop_params=True, two consecutive 422s
with an unstructured body must raise instead of silently returning None.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest

from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.openai import OpenAIChatCompletion, OpenAIConfig
from litellm.types.utils import ModelResponse


def _unstructured_422_error() -> openai.UnprocessableEntityError:
    return openai.UnprocessableEntityError(
        message="Error code: 422 - content moderation rejected the request",
        response=httpx.Response(
            422,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
            json={"error": {"message": "content moderation rejected the request"}},
        ),
        body={"message": "content moderation rejected the request"},
    )


@pytest.mark.asyncio
async def test_acompletion_raises_after_unstructured_422_retry_exhaustion():
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=_unstructured_422_error())

    with pytest.raises((openai.UnprocessableEntityError, OpenAIError)) as exc_info:
        await OpenAIChatCompletion().acompletion(
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            provider_config=OpenAIConfig(),
            model="gpt-4o",
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            timeout=30.0,
            api_key="sk-test",
            client=mock_client,
            drop_params=True,
        )

    assert exc_info.value.status_code == 422
    assert mock_client.chat.completions.with_raw_response.create.await_count == 2


def test_completion_raises_after_unstructured_422_retry_exhaustion():
    mock_client = MagicMock(spec=openai.OpenAI)
    mock_client.api_key = "sk-test"
    mock_client._base_url = MagicMock()
    mock_client.chat.completions.with_raw_response.create.side_effect = _unstructured_422_error()

    with pytest.raises((openai.UnprocessableEntityError, OpenAIError)) as exc_info:
        OpenAIChatCompletion().completion(
            model_response=ModelResponse(),
            timeout=30.0,
            optional_params={},
            litellm_params={},
            logging_obj=MagicMock(),
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            client=mock_client,
            drop_params=True,
        )

    assert exc_info.value.status_code == 422
    assert mock_client.chat.completions.with_raw_response.create.call_count == 2


@pytest.mark.asyncio
async def test_async_streaming_raises_after_unstructured_422_retry_exhaustion():
    mock_client = MagicMock()
    mock_client.chat.completions.with_raw_response.create = AsyncMock(side_effect=_unstructured_422_error())

    with pytest.raises((openai.UnprocessableEntityError, OpenAIError)) as exc_info:
        await OpenAIChatCompletion().async_streaming(
            timeout=30.0,
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            provider_config=OpenAIConfig(),
            model="gpt-4o",
            logging_obj=MagicMock(),
            api_key="sk-test",
            client=mock_client,
            drop_params=True,
        )

    assert exc_info.value.status_code == 422
    assert mock_client.chat.completions.with_raw_response.create.await_count == 2
