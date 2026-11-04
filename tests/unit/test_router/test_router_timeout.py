from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from collections.abc import Awaitable, Callable, Mapping
from typing import Final
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest
import respx

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.router_utils.cooldown_handlers import is_caller_timeout_408
from litellm.types.integrations.custom_logger import WEBSEARCH_INTERCEPTION_PREFIX, as_converted_stream
from litellm.types.llms.openai import AllMessageValues
import time
from unittest.mock import patch, MagicMock

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


@pytest.mark.parametrize(
    "stream",
    [
        True,
        False,
    ],
)
def test_unit_test_streaming_timeout(stream):
    import os
    from dotenv import load_dotenv
    import litellm
    from litellm.router import Router, RetryPolicy, AllowedFailsPolicy

    litellm.set_verbose = True

    model_list = [
        {
            "model_name": "llama3",
            "litellm_params": {
                "model": "watsonx/meta-llama/llama-3-1-8b-instruct",
                "api_base": os.getenv("WATSONX_URL_US_SOUTH"),
                "api_key": os.getenv("WATSONX_API_KEY"),
                "project_id": os.getenv("WATSONX_PROJECT_ID_US_SOUTH"),
                "timeout": 0.01,
                "stream_timeout": 0.0000001,
            },
        },
        {
            "model_name": "bedrock-anthropic",
            "litellm_params": {
                "model": "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
                "timeout": 0.01,
                "stream_timeout": 0.0000001,
            },
        },
        {
            "model_name": "llama3-fallback",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        },
    ]

    router = Router(model_list=model_list)

    stream_timeout = 0.0000001
    normal_timeout = 0.01

    args = {
        "kwargs": {"stream": stream},
        "data": {"timeout": normal_timeout, "stream_timeout": stream_timeout},
    }

    assert router._get_stream_timeout(**args) == stream_timeout

    assert router._get_non_stream_timeout(**args) == normal_timeout

    stream_timeout_val = router._get_timeout(
        kwargs={"stream": stream},
        data={"timeout": normal_timeout, "stream_timeout": stream_timeout},
    )

    if stream:
        assert stream_timeout_val == stream_timeout
    else:
        assert stream_timeout_val == normal_timeout


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
def test_the_deployment_timeout_for_the_requests_service_tier_wins_over_its_general_timeout(
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
    deployment_params: Final = router.model_list[0]["litellm_params"]

    assert router._get_timeout(kwargs=request_kwargs, data=deployment_params) == expected_timeout


def test_a_stream_keeps_the_deployment_stream_timeout_when_only_the_request_timeout_map_names_its_tier() -> None:
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
    deployment_params: Final = router.model_list[0]["litellm_params"]

    assert router._get_timeout(kwargs={"service_tier": "flex", "stream": True}, data=deployment_params) == 30.0
    assert router._get_timeout(kwargs={"service_tier": "flex"}, data=deployment_params) == 900.0


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
    response: Final = await router.acompletion(model="claude", messages=[{"role": "user", "content": "hi"}], stream=True)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "answer", "streamed_text"),
    [
        ("bedrock/converse/anthropic.claude-sonnet-4-5-v1:0", CONVERSE_ANSWER, _streamed_chat_text),
        ("bedrock/invoke/anthropic.claude-sonnet-4-5-v1:0", _INVOKE_ANSWER, _streamed_messages_text),
    ],
    ids=["chat_completions", "anthropic_messages"],
)
async def test_stream_a_deployment_hook_answers_in_one_piece_gets_the_longer_limit(
    model: str,
    answer: bytes,
    streamed_text: Callable[[Router], Awaitable[str]],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "callbacks", [_AnswersStreamsInOnePiece()])
    respx_mock.post(url__regex=r"https://bedrock-runtime\.us-east-1\.amazonaws\.com/.*").mock(
        side_effect=slow_upstream_answer(answer)
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "claude",
                "litellm_params": {
                    "model": model,
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
        model="native", messages=[{"role": "user", "content": "hi"}], max_tokens=10, stream=True, client_side_timeout=True
    )
    _ = [chunk async for chunk in response]


async def _drain_native_responses_stream(router: Router) -> None:
    response: Final = await router.aresponses(model="native", input="hi", stream=True, client_side_timeout=True)
    _ = [event async for event in response]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "drain"),
    [
        ("bedrock/invoke/anthropic.claude-sonnet-4-5-v1:0", _drain_messages_stream),
        ("openai/gpt-4.1", _drain_native_responses_stream),
    ],
    ids=["anthropic_messages", "responses"],
)
async def test_native_live_stream_cut_at_the_stream_timeout_is_not_blamed_on_the_deployment(
    model: str,
    drain: Callable[[Router], Awaitable[None]],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    failure_details: Final = _FailureDetails()
    monkeypatch.setattr(litellm, "callbacks", [failure_details])
    respx_mock.post(url__regex=r"https://(bedrock-runtime\.us-east-1\.amazonaws\.com|api\.openai\.com)/.*").mock(
        side_effect=slow_upstream_answer()
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "native",
                "litellm_params": {
                    "model": model,
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
    assert is_caller_timeout_408(failure_details.details, 408, ended=started + timedelta(seconds=STREAM_TIMEOUT_SECONDS))

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
    route: Final = respx_mock.post(
        url__regex=r"^https://bedrock-runtime\.us-east-1\.amazonaws\.com/.*"
    ).mock(side_effect=httpx.ReadTimeout("upstream timeout"))
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

    assert tuple(chunk.choices[0].delta.content for chunk in chunks if chunk.choices[0].delta.content) == (
        "fallback",
    )
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
    route: Final = respx_mock.post(
        "https://api.anthropic.com/v1/messages"
    ).mock(side_effect=httpx.ReadTimeout("scripted Anthropic timeout"))
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
    route: Final = respx_mock.post(
        "https://api.openai.com/v1/chat/completions"
    ).mock(side_effect=httpx.ReadTimeout("scripted OpenAI timeout"))
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
    route: Final = respx_mock.post(
        "https://azure.example.com/openai/deployments/gpt-4o-mini/chat/completions"
    ).mock(side_effect=httpx.ReadTimeout("scripted Azure timeout"))
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
