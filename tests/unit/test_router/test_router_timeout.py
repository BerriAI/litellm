from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Final
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import openai
import pytest
import pytest_asyncio
import respx
from openai.types.chat import ChatCompletionChunk

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.router_api import ChangeSet
from litellm.router_rust.core import RustCore
from litellm.router_rust.invoke.streaming import StreamAttempt, open_attempt_stream
from litellm.router_rust.surface.health import logged_call
from litellm.router_utils.cooldown_handlers import is_caller_timeout_408, is_stream_ttft_expiry
from litellm.types.integrations.custom_logger import WEBSEARCH_INTERCEPTION_PREFIX, as_converted_stream
from litellm.types.llms.openai import AllMessageValues
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service
from tests.unit.test_router.core.streaming_helpers import request_logging
from tests.unit.llms.bedrock.slow_upstream import (
    CONVERSE_ANSWER,
    STREAM_TIMEOUT_SECONDS,
    UPSTREAM_FIRST_BYTE_SECONDS,
    slow_upstream_answer,
    slow_upstream_async_client,
)


@pytest.fixture
def restore_request_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "request_timeout", litellm.request_timeout)


@pytest.mark.parametrize(
    "num_retries, expected_call_count",
    [(0, 1), (1, 2), (2, 3), (3, 4)],
)
@pytest.mark.usefixtures("fake_provider_credentials", "restore_request_timeout")
def test_router_timeout_with_retries_anthropic_model(num_retries, expected_call_count):
    """
    If request hits custom timeout, ensure it's retried.
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm.num_retries = num_retries
    litellm.request_timeout = 0.000001

    router = Router(
        model_list=[
            {
                "model_name": "claude-3-haiku",
                "litellm_params": {
                    "model": f"anthropic/{os.environ.get('CI_CD_DEFAULT_ANTHROPIC_MODEL', 'claude-haiku-4-5-20251001')}",
                },
            }
        ],
    )

    custom_client = HTTPHandler()

    with patch.object(custom_client, "post", new=MagicMock()) as mock_client:
        try:

            def delayed_response(*args, **kwargs):
                time.sleep(0.01)  # Exceeds the 0.000001 timeout
                raise TimeoutError("Request timed out.")

            mock_client.side_effect = delayed_response

            router.completion(
                model="claude-3-haiku",
                messages=[{"role": "user", "content": "hello, who are u"}],
                client=custom_client,
            )
        except litellm.Timeout:
            pass

        assert mock_client.call_count == expected_call_count


async def _timeout_sent_to_the_provider_call(router: Router, request_kwargs: Mapping[str, object]) -> object:
    sent: Final[list[object]] = []
    real_acompletion: Final = litellm.acompletion

    async def spy(*args: object, **kwargs: object) -> object:
        # A stream is bounded by the stream timeout litellm.acompletion is handed, any other call by its timeout
        stream_timeout = kwargs.get("stream_timeout")
        sent.append(
            stream_timeout if kwargs.get("stream") is True and stream_timeout is not None else kwargs.get("timeout")
        )
        return await real_acompletion(*args, **kwargs)

    with patch.object(litellm, "acompletion", spy):
        response = await router.acompletion(
            model="openai-gpt-5.1", messages=[{"role": "user", "content": "hi"}], mock_response="ok", **request_kwargs
        )
        if request_kwargs.get("stream"):
            _ = [chunk async for chunk in response]
    assert len(sent) == 1
    return sent[0]


@pytest.mark.parametrize(
    "request_kwargs, expected_timeout",
    [
        ({"service_tier": "flex"}, 900.0),
        ({"service_tier": "default"}, 610.0),
        ({}, 610.0),
        ({"service_tier": "flex", "timeout": 20}, 20),
        ({"service_tier": "flex", "stream": True}, 950.0),
        ({"service_tier": "default", "stream": True}, 610.0),
        ({"service_tier": "flex", "stream": True, "stream_timeout": 30}, 30),
    ],
)
@pytest.mark.asyncio
async def test_the_deployment_timeout_for_the_requests_service_tier_wins_over_its_general_timeout(
    request_kwargs: dict[str, object], expected_timeout: float
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai-gpt-5.1",
                "litellm_params": {
                    "model": "openai/gpt-5.1",
                    "api_key": "sk-test",
                    "timeout": 610,
                    "stream_timeout": 610,
                    "timeout_by_service_tier": {"flex": 900},
                    "stream_timeout_by_service_tier": {"flex": 950},
                },
            }
        ]
    )

    assert await _timeout_sent_to_the_provider_call(router, request_kwargs) == expected_timeout


@pytest.mark.asyncio
async def test_a_stream_keeps_the_deployment_stream_timeout_when_only_the_request_timeout_map_names_its_tier() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "openai-gpt-5.1",
                "litellm_params": {
                    "model": "openai/gpt-5.1",
                    "api_key": "sk-test",
                    "timeout": 610,
                    "stream_timeout": 30,
                    "timeout_by_service_tier": {"flex": 900},
                },
            }
        ]
    )

    assert await _timeout_sent_to_the_provider_call(router, {"service_tier": "flex", "stream": True}) == 30.0
    assert await _timeout_sent_to_the_provider_call(router, {"service_tier": "flex"}) == 900.0


async def _drain_responses_stream(router: Router) -> None:
    response: Final = await router.aresponses(model="gpt", input="hi", stream=True)
    _ = [event async for event in response]


@pytest.mark.asyncio
async def test_live_responses_stream_through_the_router_fails_at_the_stream_timeout(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/responses").mock(side_effect=slow_upstream_answer())
    router: Final = Router(
        model_list=[{"model_name": "gpt", "litellm_params": {"model": "openai/gpt-4.1", "api_key": "fake"}}],
        timeout=UPSTREAM_FIRST_BYTE_SECONDS * 2,
        stream_timeout=STREAM_TIMEOUT_SECONDS,
        num_retries=0,
    )

    with pytest.raises(litellm.Timeout):
        await _drain_responses_stream(router)


@pytest.mark.asyncio
async def test_prompt_managed_stream_keeps_the_target_deployments_own_stream_timeout() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-agent",
                "litellm_params": {"model": "litellm_agent/claude-target"},
            },
            {
                "model_name": "claude-target",
                "litellm_params": {
                    "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
                    "aws_access_key_id": "fake",
                    "aws_secret_access_key": "fake",
                    "aws_region_name": "us-east-1",
                    "stream_timeout": STREAM_TIMEOUT_SECONDS,
                },
            },
        ],
        timeout=UPSTREAM_FIRST_BYTE_SECONDS * 2,
        stream_timeout=UPSTREAM_FIRST_BYTE_SECONDS * 2,
        num_retries=0,
    )
    messages: Final[list[AllMessageValues]] = [{"role": "user", "content": "hi"}]

    with pytest.raises(litellm.Timeout):
        await router.acompletion(
            model="claude-agent",
            messages=messages,
            stream=True,
            client=slow_upstream_async_client(),
            litellm_logging_obj=LiteLLMLogging(
                model="claude-agent",
                messages=messages,
                stream=True,
                call_type="acompletion",
                start_time=datetime.now(),
                litellm_call_id="prompt-managed-stream",
                function_id="f",
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_params", "target_params", "request_params"),
    [
        ({}, {"timeout": UPSTREAM_FIRST_BYTE_SECONDS * 2}, {}),
        ({"timeout": STREAM_TIMEOUT_SECONDS}, {}, {"request_timeout": UPSTREAM_FIRST_BYTE_SECONDS * 2}),
    ],
    ids=["target_deployment_timeout", "caller_request_timeout"],
)
async def test_prompt_managed_fake_stream_keeps_the_timeout_its_caller_or_target_set(
    agent_params: Mapping[str, float], target_params: Mapping[str, float], request_params: Mapping[str, float]
) -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "jamba-agent",
                "litellm_params": {"model": "litellm_agent/jamba-target", **agent_params},
            },
            {
                "model_name": "jamba-target",
                "litellm_params": {
                    "model": "bedrock/converse/ai21.jamba-1-5-large-v1:0",
                    "aws_access_key_id": "fake",
                    "aws_secret_access_key": "fake",
                    "aws_region_name": "us-east-1",
                    **target_params,
                },
            },
        ],
        timeout=STREAM_TIMEOUT_SECONDS,
        stream_timeout=STREAM_TIMEOUT_SECONDS,
        num_retries=0,
    )
    messages: Final[list[AllMessageValues]] = [{"role": "user", "content": "hi"}]

    response: Final = await router.acompletion(
        model="jamba-agent",
        messages=messages,
        stream=True,
        client=slow_upstream_async_client(CONVERSE_ANSWER),
        **request_params,
        litellm_logging_obj=LiteLLMLogging(
            model="jamba-agent",
            messages=messages,
            stream=True,
            call_type="acompletion",
            start_time=datetime.now(),
            litellm_call_id="prompt-managed-fake-stream",
            function_id="f",
        ),
    )

    assert isinstance(response, litellm.CustomStreamWrapper)
    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == "hi"


class _AnswersStreamsInOnePiece(CustomLogger):
    async def async_pre_call_deployment_hook(self, kwargs: dict, call_type: object) -> dict:
        return as_converted_stream(kwargs, WEBSEARCH_INTERCEPTION_PREFIX) if kwargs.get("stream") else kwargs


_INVOKE_ANSWER: Final = json.dumps(
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
).encode()


async def _streamed_chat_text(router: Router) -> str:
    response: Final = await router.acompletion(
        model="claude", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    return "".join([chunk.choices[0].delta.content or "" async for chunk in response])


async def _streamed_messages_text(router: Router) -> str:
    response: Final = await router.aanthropic_messages(
        model="claude", messages=[{"role": "user", "content": "hi"}], max_tokens=10, stream=True
    )
    events: Final = [
        json.loads(line.removeprefix(b"data: "))
        for line in b"".join([chunk async for chunk in response]).splitlines()
        if line.startswith(b"data: ")
    ]
    return "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta")


@contextmanager
def _slow_upstream(respx_mock: respx.MockRouter, served_by_rust: bool, answer: bytes = b"") -> Iterator[str | None]:
    if not served_by_rust:
        respx_mock.post(url__regex=r"https://(bedrock-runtime\.us-east-1\.amazonaws\.com|api\.openai\.com)/.*").mock(
            side_effect=slow_upstream_answer(answer)
        )
        yield None
        return
    with recording_service() as server:
        server.expected_requests = None
        server.default_response = ResponseSpec(body=answer, delay=UPSTREAM_FIRST_BYTE_SECONDS)
        yield server.base_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "answer", "streamed_text", "served_by_rust"),
    [
        ("bedrock/converse/anthropic.claude-sonnet-4-5-v1:0", CONVERSE_ANSWER, _streamed_chat_text, False),
        ("bedrock/invoke/anthropic.claude-sonnet-4-5-v1:0", _INVOKE_ANSWER, _streamed_messages_text, True),
    ],
    ids=["chat_completions", "anthropic_messages"],
)
async def test_stream_a_deployment_hook_answers_in_one_piece_gets_the_longer_limit(
    model: str,
    answer: bytes,
    streamed_text: Callable[[Router], Awaitable[str]],
    served_by_rust: bool,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [_AnswersStreamsInOnePiece()])
    with _slow_upstream(respx_mock, served_by_rust, answer) as api_base:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "claude",
                    "litellm_params": {
                        "model": model,
                        "api_base": api_base,
                        "aws_access_key_id": "fake",
                        "aws_secret_access_key": "fake",
                        "aws_region_name": "us-east-1",
                        "timeout": STREAM_TIMEOUT_SECONDS,
                        "stream_timeout": UPSTREAM_FIRST_BYTE_SECONDS * 2,
                    },
                }
            ],
            num_retries=0,
        )

        assert await streamed_text(router) == "hi"


class _FailureDetails(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.logged: Final = asyncio.Event()
        self.details: Mapping[str, object] = {}

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.details = kwargs
        self.logged.set()


async def _drain_messages_stream(router: Router) -> None:
    response: Final = await router.aanthropic_messages(
        model="native",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        stream=True,
        client_side_timeout=True,
    )
    _ = [chunk async for chunk in response]


async def _drain_native_responses_stream(router: Router) -> None:
    response: Final = await router.aresponses(model="native", input="hi", stream=True, client_side_timeout=True)
    _ = [event async for event in response]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "drain", "served_by_rust"),
    [
        ("bedrock/invoke/anthropic.claude-sonnet-4-5-v1:0", _drain_messages_stream, True),
        ("openai/gpt-4.1", _drain_native_responses_stream, False),
    ],
    ids=["anthropic_messages", "responses"],
)
async def test_native_live_stream_cut_at_the_stream_timeout_is_not_blamed_on_the_deployment(
    model: str,
    drain: Callable[[Router], Awaitable[None]],
    served_by_rust: bool,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    failure_details: Final = _FailureDetails()
    monkeypatch.setattr(litellm, "callbacks", [failure_details])
    with _slow_upstream(respx_mock, served_by_rust) as api_base:
        router: Final = Router(
            model_list=[
                {
                    "model_name": "native",
                    "litellm_params": {
                        "model": model,
                        "api_base": api_base,
                        "api_key": "fake",
                        "aws_access_key_id": "fake",
                        "aws_secret_access_key": "fake",
                        "aws_region_name": "us-east-1",
                    },
                }
            ],
            timeout=UPSTREAM_FIRST_BYTE_SECONDS * 2,
            stream_timeout=STREAM_TIMEOUT_SECONDS,
            num_retries=0,
        )

        with pytest.raises(litellm.Timeout):
            await drain(router)
        await asyncio.wait_for(failure_details.logged.wait(), timeout=10)

    started: Final = failure_details.details.get("api_call_start_time")
    assert isinstance(started, datetime)
    assert is_caller_timeout_408(
        failure_details.details, 408, ended=started + timedelta(seconds=STREAM_TIMEOUT_SECONDS)
    )


@pytest.mark.parametrize(
    "params, elapsed, status, expired, expected",
    (
        ({"client_side_stream_ttft_timeout": True, "stream_ttft_timeout": 0.1}, 1.0, 408, False, False),
        ({"client_side_stream_ttft_timeout": True, "stream_ttft_timeout": 0.1}, 0.0, 408, True, True),
        ({"stream_ttft_timeout": 0.1}, 1.0, 408, True, False),
        ({"client_side_timeout": True, "timeout": 600, "stream_ttft_timeout": 0.1}, 1.0, 408, True, False),
        ({"client_side_stream_ttft_timeout": True, "stream_ttft_timeout": 0.1}, 1.0, 500, True, False),
        (
            {
                "client_side_timeout": True,
                "timeout": 600,
                "client_side_stream_ttft_timeout": True,
                "stream_ttft_timeout": 0.1,
            },
            1.0,
            408,
            False,
            False,
        ),
        (
            {
                "client_side_timeout": True,
                "timeout": 0.5,
                "client_side_stream_ttft_timeout": True,
                "stream_ttft_timeout": 600,
            },
            1.0,
            408,
            False,
            True,
        ),
    ),
)
def test_caller_stream_ttft_timeout_408_requires_router_expiry(
    params: Mapping[str, object],
    elapsed: float,
    status: int,
    expired: bool,
    expected: bool,
) -> None:
    started: Final = datetime(2024, 1, 1)
    details: Final = {
        "litellm_params": params,
        "api_call_start_time": started,
        "end_time": started + timedelta(seconds=elapsed),
    }
    error: Final = litellm.Timeout(message="timeout", model="ttft-primary", llm_provider="openai")
    setattr(error, "stream_ttft_expired", expired)
    assert is_caller_timeout_408(details, status, exception=error) is expected


@pytest.mark.asyncio
async def test_bedrock_timeout_preserves_the_provider_timeout_error(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    model_id: Final = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    route: Final = respx_mock.post(url__regex=r"^https://bedrock-runtime\.us-east-1\.amazonaws\.com/.*").mock(
        side_effect=httpx.ReadTimeout("upstream timeout")
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "bedrock-test",
                "litellm_params": {
                    "model": f"bedrock/{model_id}",
                    "aws_region_name": "us-east-1",
                    "timeout": 0.0001,
                },
            }
        ],
        num_retries=0,
    )

    with pytest.raises(openai.APITimeoutError) as exc_info:
        await router.acompletion(
            model="bedrock-test",
            messages=[{"role": "user", "content": "timeout test"}],
        )

    assert route.call_count == 1
    assert route.calls[0].request.url.path.endswith(f"/model/{model_id}/converse")


def test_stream_timeout_uses_the_configured_fallback(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    stream_body: Final = "\n\n".join(
        (
            f"data: {json.dumps({'id': 'chatcmpl-fallback', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'fallback', 'choices': [{'index': 0, 'delta': {'content': 'fallback'}, 'finish_reason': 'stop'}]})}",
            "data: [DONE]",
            "",
        )
    )
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.ReadTimeout("primary stream timed out"),
            httpx.Response(200, text=stream_body),
        ]
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/primary", "api_key": "test-key"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {"model": "openai/fallback", "api_key": "test-key"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
    )

    response: Final = router.completion(
        model="primary",
        messages=[{"role": "user", "content": "stream timeout test"}],
        stream=True,
    )
    chunks: Final = tuple(response)

    assert tuple(chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content) == ("fallback",)
    request_bodies: Final = tuple(json.loads(call.request.content) for call in route.calls)
    assert route.call_count == 2
    assert tuple(body["model"] for body in request_bodies) == ("primary", "fallback")
    assert tuple(body["stream"] for body in request_bodies) == (True, True)


def test_openai_timeout_raises_provider_timeout_for_scripted_failure(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=httpx.ReadTimeout("scripted upstream timeout")
    )

    with pytest.raises(openai.APITimeoutError):
        litellm.completion(
            model="openai/gpt-4o-mini",
            api_key="scripted-timeout-key",
            timeout=0.01,
            messages=[{"role": "user", "content": "timeout"}],
            num_retries=0,
        )

    assert route.call_count == 1


@pytest.mark.parametrize(
    ("async_mode", "stream"),
    ((False, False), (False, True), (True, False), (True, True)),
)
@pytest.mark.asyncio
async def test_anthropic_timeout_raises_for_sync_and_streaming(
    async_mode: bool,
    stream: bool,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=httpx.ReadTimeout("scripted Anthropic timeout")
    )
    request_kwargs: Final = {
        "model": "anthropic/claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "timeout contract"}],
        "api_key": "scripted-anthropic-key",
        "timeout": 0.001,
        "stream": stream,
    }

    async def run_async_request() -> None:
        response: Final = await litellm.acompletion(**request_kwargs)
        if stream:
            _ = tuple([chunk async for chunk in response])

    def run_sync_request() -> None:
        response: Final = litellm.completion(**request_kwargs)
        if stream:
            _ = tuple(response)

    if async_mode:
        with pytest.raises(litellm.Timeout):
            await run_async_request()
    else:
        with pytest.raises(litellm.Timeout):
            run_sync_request()

    assert route.call_count == 1


def test_openai_router_timeout_raises_after_httpx_read_timeout(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=httpx.ReadTimeout("scripted OpenAI timeout")
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "timeout-model",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "scripted-openai-key",
                    "timeout": 0.001,
                },
            }
        ],
        num_retries=0,
    )

    with pytest.raises(litellm.Timeout):
        router.completion(
            model="timeout-model",
            messages=[{"role": "user", "content": "timeout contract"}],
        )

    assert route.call_count == 1


def test_azure_router_timeout_raises_after_httpx_read_timeout(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://azure.example.com/openai/deployments/gpt-4o-mini/chat/completions").mock(
        side_effect=httpx.ReadTimeout("scripted Azure timeout")
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "azure-timeout",
                "litellm_params": {
                    "model": "azure/gpt-4o-mini",
                    "api_base": "https://azure.example.com",
                    "api_version": "2024-10-21",
                    "api_key": "scripted-azure-key",
                    "timeout": 0.001,
                },
            }
        ],
        num_retries=0,
    )

    with pytest.raises(litellm.Timeout):
        router.completion(
            model="azure-timeout",
            messages=[{"role": "user", "content": "timeout contract"}],
        )

    assert route.call_count == 1


class _ManualClock:
    __slots__ = ("now",)

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    async def elapse(self, seconds: float) -> None:
        steps: Final = max(1, round(seconds / 0.01))
        for _ in range(steps):
            self.now += seconds / steps
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)


@pytest_asyncio.fixture(loop_scope="function")
async def stream_clock(monkeypatch: pytest.MonkeyPatch) -> _ManualClock:
    loop: Final = asyncio.get_running_loop()
    clock: Final = _ManualClock(loop.time())
    monkeypatch.setattr(loop, "time", clock)
    return clock


@pytest.fixture
def ttft_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


class _TTFTObserver(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.timeout: object = None
        self.logging_params: Mapping[str, object] = {}
        self.logging_ready: Final = asyncio.Event()
        self.logging_loop: Final = asyncio.get_running_loop()

    def log_pre_api_call(self, model: object, messages: object, kwargs: Mapping[str, object]) -> None:
        self.logging_params = as_str_mapping(kwargs.get("litellm_params")) or {}
        self.logging_loop.call_soon_threadsafe(self.logging_ready.set)

    async def async_pre_call_deployment_hook(
        self, kwargs: dict[str, object], call_type: object
    ) -> dict[str, object] | None:
        self.timeout = kwargs.get("stream_ttft_timeout")
        return None


def _ttft_chat_chunk(delta: Mapping[str, object], finish: str | None = None) -> bytes:
    return (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-ttft",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "ttft-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )
        + "\n\n"
    ).encode()


_OMITTED: Final = object()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_value, expected",
    (
        (_OMITTED, 0.2),
        (None, 0.2),
        ("not-a-number", 0.2),
        (True, 0.2),
        (10**400, 0.2),
        ("nan", 0.2),
        ("1e400", 0.2),
        ("0.3", 0.3),
    ),
    ids=(
        "omitted",
        "explicit-null",
        "non-numeric",
        "boolean",
        "int-past-float-range",
        "nan",
        "infinite",
        "request-value",
    ),
)
async def test_stream_ttft_timeout_is_resolved_per_attempt_and_kept_out_of_provider_body(
    request_value: object,
    expected: float,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    observer: Final = _TTFTObserver()
    monkeypatch.setattr(litellm, "callbacks", [observer])
    route: Final = respx_mock.post("https://ttft.invalid/v1/chat/completions").respond(
        200,
        content=_ttft_chat_chunk({"content": "answer"}, "stop") + b"data: [DONE]\n\n",
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "ttft",
                "litellm_params": {
                    "model": "openai/ttft-model",
                    "api_key": "fake",
                    "api_base": "https://ttft.invalid/v1",
                    "stream_ttft_timeout": "0.2",
                },
            }
        ],
        num_retries=0,
    )
    request: Final = {
        "client_side_stream_ttft_timeout": True,
        **({} if request_value is _OMITTED else {"stream_ttft_timeout": request_value}),
    }
    response: Final = await router.acompletion(
        model="ttft",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        **request,
    )
    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == "answer"
    assert observer.timeout == expected
    assert "stream_ttft_timeout" not in json.loads(route.calls.last.request.content)
    assert "client_side_stream_ttft_timeout" not in json.loads(route.calls.last.request.content)


class _TTFTByteStream(httpx.AsyncByteStream):
    def __init__(self, clock: _ManualClock, frames: tuple[tuple[float, bytes], ...]) -> None:
        self.clock: Final = clock
        self.frames: Final = frames

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for delay, frame in self.frames:
            if delay:
                await self.clock.elapse(delay)
            yield frame


def _ttft_router(
    provider: str = "openai", *, enabled: bool = True, fallback: bool = True, messages: bool = False
) -> Router:
    return Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": f"{provider}/ttft-{name}",
                    "api_key": "fake",
                    "api_base": "https://ttft.invalid/v1",
                    "stream_timeout": 5,
                    **({"stream_ttft_timeout": 0.1} if enabled else {}),
                },
                "model_info": {"id": f"ttft-{name}", **({"supported_endpoints": ["/v1/messages"]} if messages else {})},
            }
            for name in ("primary", "fallback")
        ],
        fallbacks=[{"primary": ["fallback"]}] if fallback else [],
        num_retries=0,
        allowed_fails=100,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ("role", "keepalive", "fetch"))
async def test_stream_ttft_chat_stall_falls_back_and_records_failure(
    prefix: str,
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    frames: Final = (
        ((0.0, _ttft_chat_chunk({"role": "assistant"})),)
        if prefix == "role"
        else ((0.04, b": ping\n\n"),) * 3
        if prefix == "keepalive"
        else ()
    )

    async def slow(_request: httpx.Request) -> httpx.Response:
        if prefix == "fetch":
            await stream_clock.elapse(0.3)
        return httpx.Response(
            200,
            stream=_TTFTByteStream(
                stream_clock,
                (
                    *frames,
                    (0.3, _ttft_chat_chunk({"content": "primary"}, "stop")),
                    (0.0, b"data: [DONE]\n\n"),
                ),
            ),
        )

    route: Final = respx_mock.post("https://ttft.invalid/v1/chat/completions").mock(
        side_effect=[
            slow,
            httpx.Response(200, content=_ttft_chat_chunk({"content": "fallback"}, "stop") + b"data: [DONE]\n\n"),
        ]
    )
    router: Final = _ttft_router()
    response: Final = await router.acompletion(
        model="primary", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == "fallback"
    assert route.call_count == 2
    assert router.health().failures_this_minute("ttft-primary") == 1


class _TTFTChatStream:
    def __init__(self, clock: _ManualClock) -> None:
        self.clock: Final = clock
        self.closed: Final = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        await self.clock.elapse(0.3)
        yield ChatCompletionChunk.model_validate_json(
            _ttft_chat_chunk({"content": "late"}, "stop").removeprefix(b"data: ")
        )

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.closed.set()


class _TTFTProviderTimeoutStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise TimeoutError("provider")
        yield b""


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ("chat/completions", "messages", "responses"))
async def test_provider_timeout_before_output_is_not_caller_ttft_expiry(
    endpoint: str,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
) -> None:
    respx_mock.post(f"https://ttft.invalid/v1/{endpoint}").respond(200, stream=_TTFTProviderTimeoutStream())
    router: Final = _ttft_router(messages=endpoint == "messages", fallback=False)
    kwargs: Final = {
        "model": "primary",
        "stream": True,
        "stream_ttft_timeout": 10,
        "client_side_stream_ttft_timeout": True,
    }

    async def drain() -> None:
        if endpoint == "responses":
            responses_stream: Final = await router.aresponses(input="hi", **kwargs)
            _ = tuple([chunk async for chunk in responses_stream])
        elif endpoint == "messages":
            messages_stream: Final = await router.aanthropic_messages(
                messages=[{"role": "user", "content": "hi"}], max_tokens=10, **kwargs
            )
            _ = tuple([chunk async for chunk in messages_stream])
        else:
            chat_stream: Final = await router.acompletion(messages=[{"role": "user", "content": "hi"}], **kwargs)
            _ = tuple([chunk async for chunk in chat_stream])

    with pytest.raises((TimeoutError, litellm.APIConnectionError, litellm.Timeout)) as caught:
        await drain()
    assert not is_stream_ttft_expiry(caught.value)
    assert not is_caller_timeout_408({"litellm_params": kwargs}, 408, exception=caught.value)


@pytest.mark.asyncio
async def test_provider_timeout_during_chat_fetch_is_not_caller_ttft_expiry() -> None:
    router: Final = _ttft_router(fallback=False)
    core: Final = router._core
    assert isinstance(core, RustCore)
    kwargs: Final = {
        "model": "openai/ttft-primary",
        "stream": True,
        "stream_ttft_timeout": 10,
        "client_side_stream_ttft_timeout": True,
    }
    attempt: Final = StreamAttempt(
        core=core,
        op="acompletion",
        request=kwargs,
        kwargs=kwargs,
        group="primary",
        order=None,
        depth=0,
        level=0,
        deployment_id="ttft-primary",
        attempted_retries=0,
    )

    async def make_call(*, client: object) -> _TTFTProviderTimeoutStream:
        raise TimeoutError("provider")

    response: Final = CustomStreamWrapper(
        completion_stream=None,
        model="ttft-primary",
        custom_llm_provider="openai",
        make_call=make_call,
        logging_obj=LiteLLMLogging(
            model="ttft-primary",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="acompletion",
            start_time=datetime(2024, 1, 1),
            litellm_call_id="ttft-provider-fetch",
            function_id="ttft-provider-fetch",
        ),
    )
    with pytest.raises(TimeoutError, match="provider") as caught:
        await open_attempt_stream(attempt, response)
    assert not is_stream_ttft_expiry(caught.value)
    assert not is_caller_timeout_408({"litellm_params": kwargs}, 408, exception=caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("during_fetch", (False, True), ids=("already-open", "fetch"))
async def test_stream_ttft_chat_expiry_closes_abandoned_stream(stream_clock: _ManualClock, during_fetch: bool) -> None:
    router: Final = _ttft_router()
    core: Final = router._core
    assert isinstance(core, RustCore)
    kwargs: Final = {"model": "openai/ttft-primary", "stream": True, "stream_ttft_timeout": 0.1}
    attempt: Final = StreamAttempt(
        core=core,
        op="acompletion",
        request=kwargs,
        kwargs=kwargs,
        group="primary",
        order=None,
        depth=0,
        level=0,
        deployment_id="ttft-primary",
        attempted_retries=0,
    )
    upstream: Final = _TTFTChatStream(stream_clock)

    async def make_call(*, client: object) -> _TTFTChatStream:
        response.completion_stream = upstream
        await stream_clock.elapse(0.3)
        return upstream

    response: Final = CustomStreamWrapper(
        completion_stream=None if during_fetch else upstream,
        model="ttft-primary",
        custom_llm_provider="openai",
        make_call=make_call if during_fetch else None,
        logging_obj=LiteLLMLogging(
            model="ttft-primary",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="acompletion",
            start_time=datetime(2024, 1, 1),
            litellm_call_id="ttft-fetch-close",
            function_id="ttft-fetch-close",
        ),
    )
    started: Final = stream_clock()
    with pytest.raises(litellm.Timeout, match="stream_ttft_timeout") as caught:
        await open_attempt_stream(attempt, response)
    assert 0.08 <= stream_clock() - started <= 0.12
    assert upstream.closed.is_set()
    assert is_caller_timeout_408(
        {"litellm_params": {"client_side_stream_ttft_timeout": True}}, 408, exception=caught.value
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled, output_first", ((True, True), (False, False)), ids=("R5-output", "R6-disabled"))
async def test_stream_ttft_chat_allows_output_pause_and_disabled_deadline(
    enabled: bool,
    output_first: bool,
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://ttft.invalid/v1/chat/completions").respond(
        200,
        stream=_TTFTByteStream(
            stream_clock,
            (
                (0.0, _ttft_chat_chunk({"content": "first"} if output_first else {"role": "assistant"})),
                (0.3, _ttft_chat_chunk({"content": "last"}, "stop")),
                (0.0, b"data: [DONE]\n\n"),
            ),
        ),
    )
    router: Final = _ttft_router(enabled=enabled)
    response: Final = await router.acompletion(
        model="primary", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == (
        "firstlast" if output_first else "last"
    )
    assert route.call_count == 1


async def _drain_ttft_chat_stream(router: Router) -> None:
    response: Final = await router.acompletion(
        model="primary", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    _ = tuple([chunk async for chunk in response])


class _ProviderTimeoutAfterOutput(httpx.AsyncByteStream):
    def __init__(self, clock: _ManualClock) -> None:
        self.clock: Final = clock

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield _ttft_chat_chunk({"content": "first"})
        await self.clock.elapse(0.3)
        raise httpx.ReadTimeout("provider timeout after output")


@pytest.mark.asyncio
async def test_provider_timeout_after_first_output_is_not_exempted_by_caller_ttft(
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
) -> None:
    respx_mock.post("https://ttft.invalid/v1/chat/completions").mock(
        return_value=httpx.Response(200, stream=_ProviderTimeoutAfterOutput(stream_clock))
    )
    router: Final = _ttft_router(fallback=False)
    response: Final = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_ttft_timeout=0.1,
        client_side_stream_ttft_timeout=True,
    )
    first: Final = await anext(response)
    assert first.choices[0].delta.content == "first"
    with pytest.raises(litellm.Timeout) as caught:
        await anext(response)
    started: Final = datetime(2024, 1, 1)
    assert not is_caller_timeout_408(
        {
            "litellm_params": {"client_side_stream_ttft_timeout": True, "stream_ttft_timeout": 0.1},
            "api_call_start_time": started,
            "end_time": started + timedelta(seconds=0.3),
        },
        408,
        exception=caught.value,
    )


@pytest.mark.asyncio
async def test_stream_ttft_chat_without_fallback_raises_provider_timeout(
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://ttft.invalid/v1/chat/completions").respond(
        200,
        stream=_TTFTByteStream(
            stream_clock,
            ((0.0, _ttft_chat_chunk({"role": "assistant"})), (0.3, _ttft_chat_chunk({"content": "late"}, "stop"))),
        ),
    )
    router: Final = _ttft_router(fallback=False)
    started: Final = stream_clock()
    with pytest.raises(litellm.Timeout, match="stream_ttft_timeout"):
        await _drain_ttft_chat_stream(router)
    assert stream_clock() - started == pytest.approx(0.1, abs=0.02)


def _ttft_anthropic_frame(kind: str, **fields: object) -> bytes:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **fields})}\n\n".encode()


@pytest.mark.asyncio
async def test_stream_ttft_messages_split_delta_disarms_deadline(
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
) -> None:
    start: Final = _ttft_anthropic_frame("message_start", message={"id": "msg-ttft"})
    delta: Final = _ttft_anthropic_frame("content_block_delta", index=0, delta={"type": "text_delta", "text": "answer"})
    stop: Final = _ttft_anthropic_frame("message_stop")
    respx_mock.post("https://ttft.invalid/v1/messages").respond(
        200,
        stream=_TTFTByteStream(
            stream_clock,
            ((0.0, start), (0.0, delta[:17]), (0.0, delta[17:]), (0.3, stop)),
        ),
    )
    router: Final = _ttft_router(messages=True, fallback=False)
    response: Final = await router.aanthropic_messages(
        model="primary", messages=[{"role": "user", "content": "hi"}], max_tokens=10, stream=True
    )
    assert b"".join([chunk async for chunk in response]) == start + delta + stop
    assert router.health().failures_this_minute("ttft-primary") == 0


@pytest.mark.asyncio
async def test_stream_ttft_messages_forwards_pings_then_falls_back_without_duplicate_start(
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    start: Final = _ttft_anthropic_frame("message_start", message={"id": "msg-ttft"})
    ping: Final = _ttft_anthropic_frame("ping")
    output: Final = _ttft_anthropic_frame(
        "content_block_delta", index=0, delta={"type": "text_delta", "text": "fallback"}
    )
    route: Final = respx_mock.post("https://ttft.invalid/v1/messages").mock(
        side_effect=[
            httpx.Response(
                200,
                stream=_TTFTByteStream(
                    stream_clock,
                    (
                        (0.0, start),
                        (0.04, ping),
                        (0.04, ping),
                        (0.04, ping),
                        (0.0, output),
                    ),
                ),
            ),
            httpx.Response(200, content=start + output + _ttft_anthropic_frame("message_stop")),
        ]
    )
    router: Final = _ttft_router(messages=True)
    response: Final = await router.aanthropic_messages(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        stream=True,
    )
    assert await anext(response) == ping
    body: Final = b"".join([chunk async for chunk in response])
    assert body.count(b"event: message_start") == 1
    assert b'"text": "fallback"' in body
    assert route.call_count == 2
    assert router.health().failures_this_minute("ttft-primary") == 1


async def _drain_ttft_messages_stream(router: Router) -> None:
    response: Final = await router.aanthropic_messages(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        stream=True,
    )
    _ = tuple([chunk async for chunk in response])


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback, lifecycle_count", ((False, 1), (True, 201)), ids=("no-policy", "buffer-cap"))
async def test_stream_ttft_messages_deadline_ignores_wrapper_commit_flag(
    fallback: bool,
    lifecycle_count: int,
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://ttft.invalid/v1/messages").respond(
        200,
        stream=_TTFTByteStream(
            stream_clock,
            (
                *(
                    (
                        0.0,
                        _ttft_anthropic_frame(
                            "content_block_start", index=0, content_block={"type": "text", "text": ""}
                        ),
                    ),
                )
                * lifecycle_count,
                (0.3, _ttft_anthropic_frame("message_stop")),
            ),
        ),
    )
    router: Final = _ttft_router(fallback=fallback, messages=True)
    with pytest.raises(litellm.Timeout, match="stream_ttft_timeout"):
        await _drain_ttft_messages_stream(router)


def _ttft_response_event(kind: str, model: str, *, background: bool = False) -> bytes:
    response: Final = {
        "id": f"resp_{model}",
        "object": "response",
        "created_at": 1,
        "model": model,
        "status": "completed" if kind == "response.completed" else "in_progress",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "background": background,
    }
    event: Final = (
        {
            "type": kind,
            "delta": model,
            "item_id": "msg-ttft",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
        }
        if kind == "response.output_text.delta"
        else {"type": kind, "response": response, "sequence_number": 2 if kind == "response.completed" else 0}
    )
    return f"data: {json.dumps(event)}\n\n".encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", (False, True), ids=("R4-fallback", "background-exempt"))
async def test_stream_ttft_responses_holds_lifecycle_until_output_and_exempts_background_runs(
    background: bool,
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
) -> None:
    route: Final = respx_mock.post("https://ttft.invalid/v1/responses").mock(
        side_effect=[
            httpx.Response(
                200,
                stream=_TTFTByteStream(
                    stream_clock,
                    (
                        (0.0, _ttft_response_event("response.created", "primary", background=background)),
                        (0.3, _ttft_response_event("response.output_text.delta", "primary")),
                        (0.0, _ttft_response_event("response.completed", "primary")),
                    ),
                ),
            ),
            httpx.Response(
                200,
                stream=_TTFTByteStream(
                    stream_clock,
                    (
                        (0.0, _ttft_response_event("response.created", "fallback")),
                        (0.05, _ttft_response_event("response.output_text.delta", "fallback")),
                        (0.3, _ttft_response_event("response.completed", "fallback")),
                    ),
                ),
            ),
        ]
    )
    router: Final = _ttft_router()
    response: Final = await router.aresponses(model="primary", input="hi", stream=True, background=background)
    events: Final = tuple([event async for event in response])
    assert tuple(event.type for event in events) == (
        "response.created",
        "response.output_text.delta",
        "response.completed",
    )
    assert tuple(event.delta for event in events if event.type == "response.output_text.delta") == (
        "primary" if background else "fallback",
    )
    assert route.call_count == (1 if background else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", (False, True), ids=("deployment", "caller"))
async def test_stream_ttft_middle_hop_cools_only_deployment_deadline_with_shared_logging(
    caller: bool,
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
) -> None:
    route: Final = respx_mock.post("https://ttft.invalid/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500, json={"error": {"message": "provider failed", "type": "server_error"}}),
            httpx.Response(
                200,
                stream=_TTFTByteStream(
                    stream_clock,
                    (
                        (0.0, b": ping\n\n"),
                        (0.04, b": ping\n\n"),
                        (0.04, b": ping\n\n"),
                        (0.3, _ttft_chat_chunk({"content": "mid"}, "stop")),
                    ),
                ),
            ),
            httpx.Response(200, content=_ttft_chat_chunk({"content": "last"}, "stop") + b"data: [DONE]\n\n"),
        ]
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": name,
                "litellm_params": {
                    "model": f"openai/ttft-{name}",
                    "api_key": "fake",
                    "api_base": "https://ttft.invalid/v1",
                    "stream_timeout": 5,
                    **({"stream_ttft_timeout": 0.1} if name == "mid" and not caller else {}),
                },
                "model_info": {"id": f"ttft-{name}"},
            }
            for name in ("primary", "mid", "last")
        ],
        fallbacks=[{"primary": ["mid"]}, {"mid": ["last"]}],
        num_retries=0,
        allowed_fails=0,
    )
    response: Final = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        litellm_logging_obj=request_logging("primary"),
        **({"stream_ttft_timeout": 0.1, "client_side_stream_ttft_timeout": True} if caller else {}),
    )
    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == "last"
    assert router.health().failures_this_minute("ttft-mid") == (0 if caller else 1)
    cooldowns: Final = await router.health().cooldowns.async_get_active_cooldowns(["ttft-mid"], parent_otel_span=None)
    assert tuple(deployment_id for deployment_id, _ in cooldowns) == (() if caller else ("ttft-mid",))
    assert route.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", (False, True), ids=("deployment", "caller"))
async def test_stream_ttft_timeout_cools_only_a_deployment_configured_stall(
    caller: bool,
    stream_clock: _ManualClock,
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
) -> None:
    respx_mock.post("https://ttft.invalid/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                stream=_TTFTByteStream(
                    stream_clock,
                    (
                        (0.0, _ttft_chat_chunk({"role": "assistant"})),
                        (0.3, _ttft_chat_chunk({"content": "primary"}, "stop")),
                    ),
                ),
            ),
            httpx.Response(200, content=_ttft_chat_chunk({"content": "fallback"}, "stop") + b"data: [DONE]\n\n"),
        ]
    )
    router: Final = _ttft_router()
    router.apply(ChangeSet(assign={"allowed_fails": 0}))
    response: Final = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        **({"stream_ttft_timeout": 0.1, "client_side_stream_ttft_timeout": True} if caller else {}),
    )
    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == "fallback"
    cooldowns: Final = await router.health().cooldowns.async_get_active_cooldowns(
        ["ttft-primary"], parent_otel_span=None
    )
    assert tuple(deployment_id for deployment_id, _ in cooldowns) == (() if caller else ("ttft-primary",))
    assert router.health().failures_this_minute("ttft-primary") == (0 if caller else 1)


@pytest.mark.asyncio
async def test_stream_ttft_timeout_and_caller_marker_are_in_logging_params(
    respx_mock: respx.MockRouter,
    ttft_httpx_transport: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer: Final = _TTFTObserver()
    monkeypatch.setattr(litellm, "callbacks", [observer])
    respx_mock.post("https://ttft.invalid/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_ttft_chat_chunk({"content": "answer"}, "stop") + b"data: [DONE]\n\n"),
    )
    router: Final = _ttft_router()
    response: Final = await router.acompletion(
        model="primary",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_ttft_timeout=0.5,
        client_side_stream_ttft_timeout=True,
    )
    _ = tuple([chunk async for chunk in response])
    await asyncio.wait_for(observer.logging_ready.wait(), timeout=10)
    params: Final = observer.logging_params
    assert params["stream_ttft_timeout"] == 0.5
    assert params["client_side_stream_ttft_timeout"] is True


@pytest.mark.parametrize("caller", (True, False), ids=("caller", "deployment"))
@pytest.mark.parametrize("after_lifecycle", (False, True), ids=("first-chunk", "stream-report"))
def test_stream_ttft_timeout_failure_exits_preserve_deployment_health_signals(
    caller: bool,
    after_lifecycle: bool,
) -> None:
    router: Final = _ttft_router()
    core: Final = router._core
    assert isinstance(core, RustCore)
    started: Final = datetime(2024, 1, 1)
    kwargs: Final = {
        "metadata": {"model_group": "primary"},
        "model_info": {"id": "ttft-primary"},
        "stream_ttft_timeout": 0.5,
        "timeout": 600,
        **({"client_side_stream_ttft_timeout": True} if caller else {"client_side_timeout": True}),
    }
    attempt: Final = StreamAttempt(
        core=core,
        op="acompletion",
        request=kwargs,
        kwargs=kwargs,
        group="primary",
        order=None,
        depth=0,
        level=0,
        deployment_id="ttft-primary",
        attempted_retries=0,
    )
    call: Final = logged_call("acompletion", kwargs, started=started)
    logged: Final = {
        **call,
        **(attempt.logged() if after_lifecycle else {}),
        "end_time": started + timedelta(seconds=1),
    }
    error: Final = litellm.Timeout(
        message="No model output within stream_ttft_timeout=0.5s", model="ttft-primary", llm_provider="openai"
    )
    setattr(error, "stream_ttft_expired", True)
    assert core.failure_exits(error, logged, "ttft-primary", kwargs)["outside_deployment"] is caller
