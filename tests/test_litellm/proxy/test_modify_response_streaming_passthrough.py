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

from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import chat_completion


@pytest.mark.asyncio
async def test_streaming_modify_response_uses_request_data_logging_obj():
    request = MagicMock(spec=Request)
    fastapi_response = MagicMock(spec=Response)
    user_api_key_dict = UserAPIKeyAuth()

    # The outer request body (what _read_request_body returns) is a streaming
    # request that does NOT carry litellm_logging_obj.
    outer_body = {"model": "gpt-4o", "messages": [], "stream": True}

    # The exception carries the processor's data, which DOES carry the logging
    # object (litellm attaches it after function_setup).
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

    # The wrapper must be built with the logging object from e.request_data,
    # NOT None (which is what the outer body would have yielded).
    mock_csw.assert_called_once()
    assert mock_csw.call_args.kwargs["logging_obj"] is sentinel_logging_obj
