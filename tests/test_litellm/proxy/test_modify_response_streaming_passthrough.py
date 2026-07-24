"""Regression test for the ModifyResponseException streaming passthrough.

When a guardrail blocks a *streaming* request pre-call by raising
``ModifyResponseException``, the chat-completion route streams the violation
message back as a 200 by building a ``CustomStreamWrapper``. The logging object
must be read from ``e.request_data`` (the processor's data, which carries
``litellm_logging_obj``) and NOT from the outer request body returned by
``_read_request_body`` -- the two diverge at ``function_setup`` and only the
processor copy gets ``litellm_logging_obj`` attached.

Reading it from the outer body passed ``logging_obj=None`` to
``CustomStreamWrapper.__init__``, which dereferences
``logging_obj.model_call_details`` and 500s with
``AttributeError: 'NoneType' object has no attribute 'model_call_details'``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response

from litellm.exceptions import RejectedRequestError
from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import chat_completion


async def _run_streaming_block_and_get_wrapper(exception):
    """Drive chat_completion's streaming guardrail-passthrough handler for the
    given pre-call block exception and return the patched CustomStreamWrapper.

    The outer request body (what _read_request_body returns) is a streaming
    request that does NOT carry litellm_logging_obj -- mirroring production,
    where the outer body diverges from the processor's data at function_setup.
    Only the processor copy (exposed as exception.request_data) carries it.
    """
    request = MagicMock(spec=Request)
    fastapi_response = MagicMock(spec=Response)
    user_api_key_dict = UserAPIKeyAuth()
    outer_body = {"model": "gpt-4o", "messages": [], "stream": True}

    with patch(
        "litellm.proxy.proxy_server._read_request_body",
        new_callable=AsyncMock,
        return_value=outer_body,
    ), patch(
        "litellm.proxy.proxy_server.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
        side_effect=exception,
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_proxy_logging, patch(
        "litellm.proxy.proxy_server.select_data_generator",
        return_value=iter([]),
    ), patch(
        "litellm.CustomStreamWrapper"
    ) as mock_csw:
        mock_proxy_logging.post_call_failure_hook = AsyncMock()

        await chat_completion(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
        )

    return mock_csw


@pytest.mark.asyncio
async def test_streaming_modify_response_uses_request_data_logging_obj():
    sentinel_logging_obj = MagicMock(name="litellm_logging_obj")
    exception = ModifyResponseException(
        message="blocked by guardrail",
        model="gpt-4o",
        request_data={
            "model": "gpt-4o",
            "stream": True,
            "litellm_logging_obj": sentinel_logging_obj,
        },
        guardrail_name="test-guardrail",
    )

    mock_csw = await _run_streaming_block_and_get_wrapper(exception)

    # The wrapper must be built with the logging object from e.request_data,
    # NOT None (which is what the outer body would have yielded).
    mock_csw.assert_called_once()
    assert mock_csw.call_args.kwargs["logging_obj"] is sentinel_logging_obj


@pytest.mark.asyncio
async def test_streaming_rejected_request_uses_request_data_logging_obj():
    # RejectedRequestError gets the identical fix in its own streaming
    # passthrough handler, so it needs the same regression guard.
    sentinel_logging_obj = MagicMock(name="litellm_logging_obj")
    exception = RejectedRequestError(
        message="rejected by guardrail",
        model="gpt-4o",
        llm_provider="openai",
        request_data={
            "model": "gpt-4o",
            "stream": True,
            "litellm_logging_obj": sentinel_logging_obj,
        },
    )

    mock_csw = await _run_streaming_block_and_get_wrapper(exception)

    mock_csw.assert_called_once()
    assert mock_csw.call_args.kwargs["logging_obj"] is sentinel_logging_obj
