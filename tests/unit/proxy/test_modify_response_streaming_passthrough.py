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

import asyncio
import json
from collections.abc import AsyncGenerator, Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.types import Message

import litellm
import litellm.proxy.proxy_server as proxy_server
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import RejectedRequestError
from litellm.integrations.custom_guardrail import CustomGuardrail, ModifyResponseException
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import chat_completion, completion
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import ModelResponse, ModelResponseStream, TextCompletionResponse

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


_BLOCK_TEXT: Final = "blocked by guardrail"
_PING: Final = ": ping"


class _BlockEveryRequestOnceReleased(CustomGuardrail):
    def __init__(self, released: asyncio.Event) -> None:
        super().__init__(guardrail_name="deny-list", event_hook=GuardrailEventHooks.pre_call, default_on=True)
        self.released: Final = released
        self.failures: tuple[Exception, ...] = ()

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> None:
        await self.released.wait()
        self.raise_passthrough_exception(violation_message=_BLOCK_TEXT, request_data=data)

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        self.failures = (*self.failures, original_exception)


class _RejectEveryRequest(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: str
    ) -> None:
        raise RejectedRequestError(
            message="rejected by the pre-call hook", model="gpt-4o", llm_provider="openai", request_data=data
        )


def _post(path: str, body: Mapping[str, object]) -> Request:
    async def receive() -> Message:
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )


async def _frames_releasing_the_block(response: StreamingResponse, released: asyncio.Event) -> tuple[str, ...]:
    async def chunks() -> AsyncGenerator[str, None]:
        async for chunk in response.body_iterator:
            released.set()
            yield chunk if isinstance(chunk, str) else bytes(chunk).decode()

    streamed: Final = "".join([chunk async for chunk in chunks()])
    return tuple(frame for frame in streamed.split("\n\n") if frame)


async def _reply(
    path: str,
    body: Mapping[str, object],
    rail: CustomLogger,
    released: asyncio.Event,
    keepalive_interval: float | None,
) -> tuple[object, tuple[str, ...]]:
    endpoint: Final = chat_completion if path == "/v1/chat/completions" else completion
    with (
        patch.object(
            proxy_server, "proxy_logging_obj", new=ProxyLogging(user_api_key_cache=DualCache())
        ),  # test-quality-ok: module global imported at call time; no injection seam
        patch.object(litellm, "callbacks", new=[rail]),  # test-quality-ok: guardrails register through this global
        patch.object(
            litellm, "sse_keepalive_ping_interval_seconds", new=keepalive_interval
        ),  # test-quality-ok: the operator's keepalive interval is read from this global
    ):
        response: Final = await endpoint(
            request=_post(path, body), fastapi_response=Response(), user_api_key_dict=UserAPIKeyAuth()
        )
        if not isinstance(response, StreamingResponse):
            return response, ()
        return response, await _frames_releasing_the_block(response, released)


async def _blocked_reply(
    path: str, body: Mapping[str, object], keepalive_interval: float | None
) -> tuple[object, tuple[str, ...], _BlockEveryRequestOnceReleased]:
    released: Final = asyncio.Event()
    if keepalive_interval is None:
        released.set()
    blocker: Final = _BlockEveryRequestOnceReleased(released)
    response, frames = await _reply(path, body, blocker, released, keepalive_interval)
    return response, frames, blocker


def _data_frames(frames: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(frame.removeprefix("data: ") for frame in frames if frame.startswith("data: {"))


KEEPALIVE_INTERVALS: Final = pytest.mark.parametrize(
    "keepalive_interval", [0.05, None], ids=["blocked-after-a-keepalive-ping", "keepalive-off"]
)


@pytest.mark.asyncio
@KEEPALIVE_INTERVALS
async def test_a_blocked_chat_stream_gets_the_block_message_and_a_content_filter_finish(
    keepalive_interval: float | None,
) -> None:
    _, frames, blocker = await _blocked_reply(
        "/v1/chat/completions", {"model": "gpt-4o", "messages": [], "stream": True}, keepalive_interval
    )

    chunks: Final = tuple(ModelResponseStream.model_validate_json(frame) for frame in _data_frames(frames))
    assert (frames[0] == _PING) is (keepalive_interval is not None)
    assert frames[-1] == "data: [DONE]"
    assert not any('"error"' in frame for frame in frames)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == _BLOCK_TEXT
    assert chunks[-1].choices[0].finish_reason == "content_filter"
    assert [type(failure) for failure in blocker.failures] == [ModifyResponseException]


@pytest.mark.asyncio
@KEEPALIVE_INTERVALS
async def test_a_blocked_text_completion_stream_gets_the_block_message(keepalive_interval: float | None) -> None:
    _, frames, blocker = await _blocked_reply(
        "/v1/completions", {"model": "gpt-4o", "prompt": "hi", "stream": True}, keepalive_interval
    )

    chunks: Final = tuple(TextCompletionResponse.model_validate_json(frame) for frame in _data_frames(frames))
    assert (frames[0] == _PING) is (keepalive_interval is not None)
    assert frames[-1] == "data: [DONE]"
    assert not any('"error"' in frame for frame in frames)
    assert "".join(chunk.choices[0].text or "" for chunk in chunks) == _BLOCK_TEXT
    assert chunks[-1].choices[0].finish_reason == "content_filter"
    assert [type(failure) for failure in blocker.failures] == [ModifyResponseException]


@pytest.mark.asyncio
async def test_a_blocked_chat_request_without_streaming_gets_the_block_message_and_a_content_filter_finish() -> None:
    response, _, blocker = await _blocked_reply(
        "/v1/chat/completions", {"model": "gpt-4o", "messages": [], "stream": False}, None
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == _BLOCK_TEXT
    assert response.choices[0].finish_reason == "content_filter"
    assert [type(failure) for failure in blocker.failures] == [ModifyResponseException]


@pytest.mark.asyncio
async def test_a_blocked_text_completion_without_streaming_gets_the_block_text_and_a_content_filter_finish() -> None:
    response, _, blocker = await _blocked_reply(
        "/v1/completions", {"model": "gpt-4o", "prompt": "hi", "stream": False}, None
    )

    assert isinstance(response, TextCompletionResponse)
    assert response.choices[0].text == _BLOCK_TEXT
    assert response.choices[0].finish_reason == "content_filter"
    assert [type(failure) for failure in blocker.failures] == [ModifyResponseException]


@pytest.mark.asyncio
async def test_a_rejected_text_completion_stream_ends_with_a_stop_finish() -> None:
    released: Final = asyncio.Event()
    released.set()

    _, frames = await _reply(
        "/v1/completions", {"model": "gpt-4o", "prompt": "hi", "stream": True}, _RejectEveryRequest(), released, None
    )

    chunks: Final = tuple(TextCompletionResponse.model_validate_json(frame) for frame in _data_frames(frames))
    assert frames[-1] == "data: [DONE]"
    assert chunks[-1].choices[0].finish_reason == "stop"
