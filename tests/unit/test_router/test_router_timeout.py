from __future__ import annotations

import asyncio
import json
import os
from typing import Final
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest
import respx

import litellm
from litellm import Router
import time
from unittest.mock import patch, MagicMock


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
