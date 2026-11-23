"""Regression test for the non-streaming RejectedRequestError response.

A pre-call hook that returns a string raises ``RejectedRequestError``, and the
route answers with that string instead of calling the LLM. The non-streaming
response used to be a bare ``ModelResponse()`` with ``model=None``, which the
OpenAI SDK rejects (``model: Input should be a valid string``).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response
from openai.types.chat import ChatCompletion

from litellm.exceptions import RejectedRequestError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import chat_completion, completion


async def _run_rejected_request(route, body):
    exception = RejectedRequestError(
        message="cached answer",
        model=body["model"],
        llm_provider="",
        request_data=dict(body),
    )
    with (
        patch(
            "litellm.proxy.proxy_server.read_request_body",
            new_callable=AsyncMock,
            return_value=dict(body),
        ),
        patch(
            "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            new_callable=AsyncMock,
            side_effect=exception,
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_proxy_logging,
    ):
        mock_proxy_logging.post_call_failure_hook = AsyncMock()

        return await route(
            request=MagicMock(spec=Request),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=UserAPIKeyAuth(),
        )


@pytest.mark.asyncio
async def test_rejected_chat_completion_response_has_model():
    response = await _run_rejected_request(
        chat_completion,
        {"model": "my-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.model == "my-model"
    ChatCompletion.model_validate(response.model_dump())


@pytest.mark.asyncio
async def test_rejected_text_completion_response_has_model():
    response = await _run_rejected_request(completion, {"model": "my-model", "prompt": "hi"})

    # finish_reason is still None here, so this is not validated against
    # openai.types.Completion the way the chat response is.
    assert response.model == "my-model"
