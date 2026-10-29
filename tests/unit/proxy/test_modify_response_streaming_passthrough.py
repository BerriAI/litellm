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
        "litellm.proxy.proxy_server.read_request_body",
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


def _late_block_request_data():
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    return {
        "model": "gpt-4o",
        "stream": True,
        "litellm_logging_obj": LiteLLMLoggingObj(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            call_type="acompletion",
            start_time=datetime.now(),
            litellm_call_id="late-block-call-id",
            function_id="late-block-function-id",
        ),
    }


async def _serialize_stream(response, **_kwargs):
    async for chunk in response:
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def _drive_streamed_block(endpoint, outer_body, keepalive_interval):
    import asyncio
    import json

    import litellm
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    block = ModifyResponseException(
        message="blocked by guardrail",
        model="gpt-4o",
        request_data=_late_block_request_data(),
        guardrail_name="test-guardrail",
    )

    async def slow_block(self, **_kwargs):
        await asyncio.sleep(0.25)
        raise block

    with patch.object(litellm, "sse_keepalive_ping_interval_seconds", keepalive_interval), patch(
        "litellm.proxy.proxy_server.read_request_body",
        new_callable=AsyncMock,
        return_value=outer_body,
    ), patch.object(ProxyBaseLLMRequestProcessing, "_process_llm_request", slow_block), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_proxy_logging, patch(
        "litellm.proxy.proxy_server.select_data_generator", side_effect=_serialize_stream
    ):
        mock_proxy_logging.post_call_failure_hook = AsyncMock(return_value=None)
        response = await endpoint(
            request=MagicMock(spec=Request),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=UserAPIKeyAuth(),
        )
        frames = [chunk if isinstance(chunk, str) else chunk.decode() async for chunk in response.body_iterator]

    mock_proxy_logging.post_call_failure_hook.assert_awaited_once()
    assert mock_proxy_logging.post_call_failure_hook.await_args.kwargs["original_exception"] is block
    assert (frames[0] == ": ping\n\n") is (keepalive_interval is not None)
    assert frames[-1] == "data: [DONE]\n\n"
    return [json.loads(frame.removeprefix("data: ")) for frame in frames if frame.startswith("data: {")]


KEEPALIVE_INTERVALS = pytest.mark.parametrize(
    "keepalive_interval", [0.05, None], ids=["blocked-after-a-keepalive-ping", "keepalive-off"]
)


@pytest.mark.asyncio
@KEEPALIVE_INTERVALS
async def test_a_blocked_chat_stream_gets_the_block_message_and_a_content_filter_finish(keepalive_interval):
    chunks = await _drive_streamed_block(
        chat_completion, {"model": "gpt-4o", "messages": [], "stream": True}, keepalive_interval
    )

    assert not any("error" in chunk for chunk in chunks)
    assert "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks) == "blocked by guardrail"
    assert chunks[-1]["choices"][0]["finish_reason"] == "content_filter"


@pytest.mark.asyncio
@KEEPALIVE_INTERVALS
async def test_a_blocked_text_completion_stream_gets_the_block_message(keepalive_interval):
    from litellm.proxy.proxy_server import completion

    chunks = await _drive_streamed_block(
        completion, {"model": "gpt-4o", "prompt": "hi", "stream": True}, keepalive_interval
    )

    assert not any("error" in chunk for chunk in chunks)
    assert "".join(chunk["choices"][0].get("text") or "" for chunk in chunks) == "blocked by guardrail"
    assert chunks[-1]["choices"][0]["finish_reason"] == "content_filter"


async def _drive_unstreamed_block(endpoint, outer_body):
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    block = ModifyResponseException(
        message="blocked by guardrail",
        model="gpt-4o",
        request_data={"model": "gpt-4o", "stream": False},
        guardrail_name="test-guardrail",
    )

    async def blocking(self, **_kwargs):
        raise block

    with patch(
        "litellm.proxy.proxy_server.read_request_body",
        new_callable=AsyncMock,
        return_value=outer_body,
    ), patch.object(ProxyBaseLLMRequestProcessing, "_process_llm_request", blocking), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_proxy_logging:
        mock_proxy_logging.post_call_failure_hook = AsyncMock(return_value=None)
        response = await endpoint(
            request=MagicMock(spec=Request),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=UserAPIKeyAuth(),
        )

    mock_proxy_logging.post_call_failure_hook.assert_awaited_once()
    assert mock_proxy_logging.post_call_failure_hook.await_args.kwargs["original_exception"] is block
    return response


@pytest.mark.asyncio
async def test_a_blocked_chat_request_without_streaming_gets_the_block_message_and_a_content_filter_finish():
    response = await _drive_unstreamed_block(chat_completion, {"model": "gpt-4o", "messages": [], "stream": False})

    assert response.choices[0].message.content == "blocked by guardrail"
    assert response.choices[0].finish_reason == "content_filter"


@pytest.mark.asyncio
async def test_a_blocked_text_completion_without_streaming_gets_the_block_text_and_a_content_filter_finish():
    from litellm.proxy.proxy_server import completion

    response = await _drive_unstreamed_block(completion, {"model": "gpt-4o", "prompt": "hi", "stream": False})

    assert response.choices[0].text == "blocked by guardrail"
    assert response.choices[0].finish_reason == "content_filter"
