"""Tests for `BedrockConverseLLM.completion`.

AWS credential resolution is stubbed so nothing reaches STS.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone
from itertools import chain
from typing import Final
from unittest.mock import MagicMock, patch

import boto3
import httpx
import openai
import pytest
import respx
from botocore.credentials import Credentials
from botocore.exceptions import ClientError

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.bedrock.chat.converse_handler import BedrockConverseLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.rust_bridge import configuration
from litellm.types.utils import AdapterCompletionStreamWrapper, ModelResponse
from tests.unit.llms.bedrock.event_loop_probe import EventLoopProbe
from tests.unit.llms.bedrock.slow_upstream import (
    CONVERSE_ANSWER,
    STREAM_TIMEOUT_SECONDS,
    UPSTREAM_FIRST_BYTE_SECONDS,
    slow_upstream_answer,
    slow_upstream_async_client,
    slow_upstream_sync_client,
)

RESOLVED_CREDENTIALS = Credentials(
    access_key="AKIARESOLVED",
    secret_key="resolved-secret",
    token="resolved-token",
)


@pytest.fixture(autouse=True)
def reset_rust_configuration(monkeypatch):
    monkeypatch.setenv("LITELLM_RUST", "1")
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


def _completion_kwargs(**overrides):
    kwargs = {
        "model": "bedrock/us-east-1/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
        "api_base": None,
        "custom_prompt_dict": {},
        "model_response": ModelResponse(),
        "encoding": None,
        "logging_obj": MagicMock(),
        "optional_params": {"maxTokens": 16},
        "acompletion": False,
        "timeout": 30.0,
        "litellm_params": {},
        "extra_headers": None,
        "client": None,
        "api_key": None,
    }
    kwargs.update(overrides)
    return kwargs


def _run(*, credentials: Credentials | None = RESOLVED_CREDENTIALS, **overrides):
    with patch.object(BedrockConverseLLM, "get_credentials", return_value=credentials):
        return BedrockConverseLLM().completion(**_completion_kwargs(**overrides))


CONVERSE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
}


async def _drive_async_completion(
    *,
    skip_pre_call_logging: bool,
    logging_obj,
    credentials: Credentials = RESOLVED_CREDENTIALS,
    outer_dispatch: bool = False,
):
    """Run the real `async_completion` with a stubbed transport."""
    import httpx as _httpx

    client = MagicMock()

    async def post(**_kwargs):
        return _httpx.Response(
            200,
            json=CONVERSE_RESPONSE,
            request=_httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
        )

    client.post = post
    client.__class__ = AsyncHTTPHandler

    if outer_dispatch:
        return await _run(credentials=credentials, acompletion=True, client=client, logging_obj=logging_obj)

    return await BedrockConverseLLM().async_completion(
        model="anthropic.claude-sonnet-4-5-v1:0",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://bedrock-runtime.us-west-2.amazonaws.com/model/m/converse",
        model_response=ModelResponse(),
        timeout=30.0,
        encoding=None,
        logging_obj=logging_obj,
        stream=None,
        optional_params={"maxTokens": 16},
        litellm_params={"aws_region_name": "us-west-2"},
        credentials=credentials,
        headers={},
        client=client,
        skip_pre_call_logging=skip_pre_call_logging,
    )


@pytest.mark.asyncio
async def test_async_completion_honors_the_pre_call_suppression():
    logging_obj = MagicMock()
    await _drive_async_completion(skip_pre_call_logging=True, logging_obj=logging_obj)
    assert logging_obj.pre_call.call_count == 0


@pytest.mark.asyncio
async def test_async_completion_logs_pre_call_by_default():
    """The suppression must be opt-in, so every existing caller keeps its log."""
    logging_obj = MagicMock()
    await _drive_async_completion(skip_pre_call_logging=False, logging_obj=logging_obj)
    assert logging_obj.pre_call.call_count == 1


@pytest.mark.asyncio
async def test_async_completion_signs_off_the_event_loop(monkeypatch):
    """Regression for issue #40165: botocore refreshes expiring credentials inside SigV4 signing with a
    blocking HTTP call, so `async_completion` must sign on a worker thread to keep the loop serving."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    probe = EventLoopProbe()
    release = asyncio.create_task(probe.release_refresh_from_the_loop())

    response = await _drive_async_completion(
        skip_pre_call_logging=False, logging_obj=MagicMock(), credentials=probe.credentials()
    )
    await release

    assert response.choices[0].message.content == "hi"
    assert probe.served_during_refresh is True


@pytest.mark.asyncio
@pytest.mark.parametrize("rust_enabled", (False, True))
async def test_python_only_async_dispatch_refreshes_credentials_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, rust_enabled: bool
) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("LITELLM_RUST", "1" if rust_enabled else "0")
    configuration.rust(rust_enabled)
    probe: Final = EventLoopProbe()
    release: Final = asyncio.create_task(probe.release_refresh_from_the_loop())

    response: Final = await _drive_async_completion(
        skip_pre_call_logging=False, logging_obj=MagicMock(), credentials=probe.credentials(), outer_dispatch=True
    )
    await release

    assert response.choices[0].message.content == "hi"
    assert probe.served_during_refresh is True


def _sync_client_returning_converse_response():
    client = MagicMock()
    client.post.side_effect = lambda **_kwargs: httpx.Response(
        200,
        json=CONVERSE_RESPONSE,
        request=httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
    )
    client.__class__ = HTTPHandler
    return client


def test_the_sync_python_path_logs_pre_call_once():
    logging_obj = MagicMock()
    response = _run(
        logging_obj=logging_obj,
        client=_sync_client_returning_converse_response(),
    )

    assert response.choices[0].message.content == "hi"
    assert logging_obj.pre_call.call_count == 1


def test_bearer_token_auth_serves_when_boto3_resolves_no_sigv4_credentials(monkeypatch):
    """With only `AWS_BEARER_TOKEN_BEDROCK` configured boto3 resolves no
    credentials at all. The handler must not dereference that None: the bearer
    token signs the request on its own."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-bearer-token")
    client = _sync_client_returning_converse_response()

    response = _run(credentials=None, litellm_params={}, client=client)

    assert response.choices[0].message.content == "hi"
    sent_headers = client.post.call_args.kwargs["headers"]
    assert sent_headers["Authorization"] == "Bearer bedrock-bearer-token"


@pytest.mark.parametrize("configured_through", ["env_var", "api_key"])
def test_bearer_token_auth_never_runs_the_sigv4_credential_chain(monkeypatch, configured_through):
    """The deployment's AWS profile does not exist, so resolving SigV4 credentials
    raises; a bearer-token deployment must still serve the request, since the
    bearer token alone signs it."""
    if configured_through == "env_var":
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-bearer-token")
    else:
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    client = _sync_client_returning_converse_response()

    response = BedrockConverseLLM().completion(
        **_completion_kwargs(
            optional_params={"maxTokens": 16, "aws_profile_name": "litellm-no-such-aws-profile"},
            litellm_params={},
            client=client,
            api_key="bedrock-bearer-token" if configured_through == "api_key" else None,
        )
    )

    assert response.choices[0].message.content == "hi"
    assert client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer bedrock-bearer-token"


def test_session_tags_sign_the_request_and_stay_out_of_the_body(monkeypatch):
    """The tagged STS session signs the Converse call and the tags never reach the request body (#34069)."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    tags = [{"Key": "team", "Value": "genai"}]

    class FakeSTSClient:
        def get_caller_identity(self):
            return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

        def assume_role(self, **params):
            if list(params.get("Tags", ())) != tags:
                raise ClientError(
                    {"Error": {"Code": "AccessDenied", "Message": "is not authorized to perform: sts:TagSession"}},
                    "AssumeRole",
                )
            return {
                "Credentials": {
                    "AccessKeyId": "ASIACONVERSETAGGED",
                    "SecretAccessKey": "assumed-secret",
                    "SessionToken": "assumed-session-token",
                    "Expiration": datetime.now(timezone.utc) + timedelta(minutes=30),
                }
            }

    client = _sync_client_returning_converse_response()
    with patch.object(boto3, "client", return_value=FakeSTSClient()):
        response = BedrockConverseLLM().completion(
            **_completion_kwargs(
                optional_params={
                    "maxTokens": 16,
                    "aws_region_name": "us-east-1",
                    "aws_access_key_id": "AKIACONVERSECALLER",
                    "aws_secret_access_key": "pod-caller-secret",
                    "aws_role_name": "arn:aws:iam::999999999999:role/litellm-converse-role",
                    "aws_session_name": "litellm-converse-session",
                    "aws_session_tags": tags,
                },
                litellm_params={},
                client=client,
            )
        )

    assert response.choices[0].message.content == "hi"
    sent = client.post.call_args.kwargs
    assert "Credential=ASIACONVERSETAGGED/" in sent["headers"]["Authorization"]
    assert "aws_session_tags" not in sent["data"]


def _converse_streaming_kwargs() -> dict[str, object]:
    return {
        "model": "bedrock/anthropic.claude-sonnet-4-5-v1:0",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "timeout": STREAM_TIMEOUT_SECONDS,
        "aws_access_key_id": "fake",
        "aws_secret_access_key": "fake",
        "aws_region_name": "us-east-1",
    }


@pytest.mark.asyncio
async def test_async_converse_streaming_fails_at_the_request_timeout_not_the_upstreams_pace() -> None:
    with pytest.raises(litellm.Timeout):
        await litellm.acompletion(client=slow_upstream_async_client(), **_converse_streaming_kwargs())


def test_sync_converse_streaming_fails_at_the_request_timeout_not_the_upstreams_pace() -> None:
    with pytest.raises(litellm.Timeout):
        litellm.completion(client=slow_upstream_sync_client(), **_converse_streaming_kwargs())


@pytest.mark.asyncio
async def test_live_stream_with_a_text_stream_timeout_reports_the_stream_timeout_it_was_cut_at() -> None:
    with pytest.raises(litellm.Timeout) as excinfo:
        await litellm.acompletion(
            client=slow_upstream_async_client(),
            **{
                **_converse_streaming_kwargs(),
                "timeout": UPSTREAM_FIRST_BYTE_SECONDS * 2,
                "stream_timeout": str(STREAM_TIMEOUT_SECONDS),
            },
        )
    assert getattr(excinfo.value, "timeout", None) == STREAM_TIMEOUT_SECONDS



_FAKE_STREAMED_REQUESTS: Final = pytest.mark.parametrize(
    ("model", "request_params"),
    [
        ("bedrock/converse/ai21.jamba-1-5-large-v1:0", {}),
        (
            "bedrock/converse/meta.llama3-3-70b-instruct-v1:0",
            {"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]},
        ),
        (
            "bedrock/converse/meta.llama3-3-70b-instruct-v1:0",
            {"response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}}},
        ),
        ("bedrock/invoke/ai21.j2-ultra-v1", {}),
    ],
    ids=[
        "model_without_streaming",
        "request_shape_without_streaming",
        "json_schema_without_native_structured_output",
        "invoke_route_without_streaming",
    ],
)


def _bedrock_router(model: str, *, timeout: float, stream_timeout: float) -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "bedrock",
                "litellm_params": {
                    "model": model,
                    "aws_access_key_id": "fake",
                    "aws_secret_access_key": "fake",
                    "aws_region_name": "us-east-1",
                },
            }
        ],
        timeout=timeout,
        stream_timeout=stream_timeout,
        num_retries=0,
    )


def _router_with_a_stream_timeout_shorter_than_the_upstreams_first_byte(model: str) -> litellm.Router:
    return _bedrock_router(model, timeout=UPSTREAM_FIRST_BYTE_SECONDS * 2, stream_timeout=STREAM_TIMEOUT_SECONDS)


@_FAKE_STREAMED_REQUESTS
def test_sync_fake_stream_is_bounded_by_the_request_timeout_not_the_stream_timeout(
    model: str, request_params: Mapping[str, object]
) -> None:
    router: Final = _router_with_a_stream_timeout_shorter_than_the_upstreams_first_byte(model)
    response: Final = router.completion(
        model="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=slow_upstream_sync_client(CONVERSE_ANSWER),
        **request_params,
    )

    assert "".join(chunk.choices[0].delta.content or "" for chunk in response) == "hi"


@pytest.mark.asyncio
@_FAKE_STREAMED_REQUESTS
async def test_async_fake_stream_is_bounded_by_the_request_timeout_not_the_stream_timeout(
    model: str, request_params: Mapping[str, object]
) -> None:
    router: Final = _router_with_a_stream_timeout_shorter_than_the_upstreams_first_byte(model)
    response: Final = await router.acompletion(
        model="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=slow_upstream_async_client(CONVERSE_ANSWER),
        **request_params,
    )

    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == "hi"


@pytest.mark.asyncio
async def test_fake_stream_keeps_a_stream_timeout_longer_than_the_request_timeout() -> None:
    router: Final = _bedrock_router(
        "bedrock/converse/ai21.jamba-1-5-large-v1:0",
        timeout=STREAM_TIMEOUT_SECONDS,
        stream_timeout=UPSTREAM_FIRST_BYTE_SECONDS * 2,
    )
    response: Final = await router.acompletion(
        model="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=slow_upstream_async_client(CONVERSE_ANSWER),
    )

    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == "hi"


_BEDROCK_RUNTIME_URL: Final = r"https://bedrock-runtime\.us-east-1\.amazonaws\.com/.*"


async def _text_completion_text(router: litellm.Router) -> str:
    response: Final = await router.atext_completion(model="bedrock", prompt="hi", stream=True)
    return "".join([chunk.choices[0].text or "" async for chunk in response])


async def _anthropic_messages_text(router: litellm.Router) -> str:
    response: Final = await router.aanthropic_messages(
        model="bedrock", messages=[{"role": "user", "content": "hi"}], max_tokens=10, stream=True
    )
    sse: Final = b"".join([chunk async for chunk in response]).decode()
    events: Final = [json.loads(line.removeprefix("data: ")) for line in sse.splitlines() if line.startswith("data: ")]
    return "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta")


async def _responses_text(router: litellm.Router) -> str:
    response: Final = await router.aresponses(model="bedrock", input="hi", stream=True)
    return "".join([event.delta async for event in response if event.type == "response.output_text.delta"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream_text",
    [_text_completion_text, _anthropic_messages_text, _responses_text],
    ids=["text_completion", "anthropic_messages", "responses"],
)
async def test_fake_stream_on_each_router_surface_is_bounded_by_the_request_timeout(
    stream_text: Callable[[litellm.Router], Awaitable[str]],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=_BEDROCK_RUNTIME_URL).mock(side_effect=slow_upstream_answer(CONVERSE_ANSWER))
    router: Final = _router_with_a_stream_timeout_shorter_than_the_upstreams_first_byte(
        "bedrock/converse/ai21.jamba-1-5-large-v1:0"
    )

    assert await stream_text(router) == "hi"


class _ChatCompletionAdapter(CustomLogger):
    def translate_completion_input_params(self, kwargs: Mapping[str, object]) -> Mapping[str, object]:
        return kwargs

    def translate_completion_output_params_streaming(self, completion_stream: object) -> AdapterCompletionStreamWrapper:
        return AdapterCompletionStreamWrapper(completion_stream=completion_stream)


@pytest.mark.asyncio
async def test_fake_stream_through_a_router_adapter_is_bounded_by_the_request_timeout(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "adapters", [{"id": "chat", "adapter": _ChatCompletionAdapter()}])
    respx_mock.post(url__regex=_BEDROCK_RUNTIME_URL).mock(side_effect=slow_upstream_answer(CONVERSE_ANSWER))
    router: Final = _router_with_a_stream_timeout_shorter_than_the_upstreams_first_byte(
        "bedrock/converse/ai21.jamba-1-5-large-v1:0"
    )

    response: Final = await router.aadapter_completion(
        adapter_id="chat", model="bedrock", messages=[{"role": "user", "content": "hi"}], stream=True
    )

    assert isinstance(response, AsyncIterator)
    contents: Final = [
        chunk.choices[0].delta.content or "" async for chunk in response if isinstance(chunk, litellm.ModelResponseStream)
    ]
    assert "".join(contents) == "hi"


@pytest.mark.asyncio
async def test_live_anthropic_messages_stream_through_the_router_fails_at_the_stream_timeout(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=_BEDROCK_RUNTIME_URL).mock(side_effect=slow_upstream_answer())
    router: Final = _router_with_a_stream_timeout_shorter_than_the_upstreams_first_byte(
        "bedrock/invoke/anthropic.claude-sonnet-4-5-v1:0"
    )

    with pytest.raises(litellm.Timeout) as excinfo:
        await _anthropic_messages_text(router)
    assert getattr(excinfo.value, "timeout", None) == STREAM_TIMEOUT_SECONDS


def test_sync_live_stream_through_the_router_fails_at_the_stream_timeout() -> None:
    router: Final = _router_with_a_stream_timeout_shorter_than_the_upstreams_first_byte(
        "bedrock/anthropic.claude-sonnet-4-5-v1:0"
    )
    with pytest.raises(litellm.Timeout):
        router.completion(
            model="bedrock", messages=[{"role": "user", "content": "hi"}], stream=True, client=slow_upstream_sync_client()
        )


@pytest.mark.asyncio
async def test_async_live_stream_through_the_router_fails_at_the_stream_timeout() -> None:
    router: Final = _router_with_a_stream_timeout_shorter_than_the_upstreams_first_byte(
        "bedrock/anthropic.claude-sonnet-4-5-v1:0"
    )
    with pytest.raises(litellm.Timeout) as excinfo:
        await router.acompletion(
            model="bedrock", messages=[{"role": "user", "content": "hi"}], stream=True, client=slow_upstream_async_client()
        )
    assert getattr(excinfo.value, "timeout", None) == STREAM_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_failed_fake_stream_reports_the_request_timeout_it_was_bounded_by() -> None:
    request_timeout: Final = UPSTREAM_FIRST_BYTE_SECONDS / 2
    router: Final = _bedrock_router(
        "bedrock/converse/ai21.jamba-1-5-large-v1:0", timeout=request_timeout, stream_timeout=STREAM_TIMEOUT_SECONDS
    )
    with pytest.raises(litellm.Timeout) as excinfo:
        await router.acompletion(
            model="bedrock",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=slow_upstream_async_client(CONVERSE_ANSWER),
        )
    assert getattr(excinfo.value, "timeout", None) == request_timeout


_NATIVE_OUTPUT_REJECTION: Final = json.dumps(
    {"message": "The model returned the following errors: output_config.format: Extra inputs are not permitted"}
).encode()


_ANSWER_SCHEMA: Final = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _rejects_native_output_then_answers_slowly(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/converse-stream"):
        return httpx.Response(400, content=_NATIVE_OUTPUT_REJECTION, request=request)
    return slow_upstream_answer(CONVERSE_ANSWER)(request)


async def _stream_json_schema_with_thinking(router: litellm.Router) -> str:
    response: Final = await router.acompletion(
        model="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        thinking={"type": "adaptive"},
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": _ANSWER_SCHEMA}},
    )
    return "".join([chunk.choices[0].delta.content or "" async for chunk in response])


def _sent_paths(respx_mock: respx.MockRouter) -> list[str]:
    return [call.request.url.path.rsplit("/", 1)[-1] for call in respx_mock.calls]


@pytest.mark.asyncio
async def test_native_output_retry_is_bounded_by_the_request_timeout_not_the_stream_timeout(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=_BEDROCK_RUNTIME_URL).mock(side_effect=_rejects_native_output_then_answers_slowly)
    router: Final = _bedrock_router(
        "bedrock/converse/us.anthropic.claude-opus-4-8",
        timeout=UPSTREAM_FIRST_BYTE_SECONDS * 2,
        stream_timeout=STREAM_TIMEOUT_SECONDS,
    )

    assert await _stream_json_schema_with_thinking(router) == "hi"
    assert _sent_paths(respx_mock) == ["converse-stream", "converse"]


@pytest.mark.asyncio
async def test_failed_native_output_retry_reports_the_request_timeout_it_was_bounded_by(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=_BEDROCK_RUNTIME_URL).mock(side_effect=_rejects_native_output_then_answers_slowly)
    request_timeout: Final = UPSTREAM_FIRST_BYTE_SECONDS / 2
    router: Final = _bedrock_router(
        "bedrock/converse/us.anthropic.claude-opus-4-8", timeout=request_timeout, stream_timeout=STREAM_TIMEOUT_SECONDS
    )

    with pytest.raises(litellm.Timeout) as excinfo:
        await _stream_json_schema_with_thinking(router)
    assert getattr(excinfo.value, "timeout", None) == request_timeout
    assert _sent_paths(respx_mock) == ["converse-stream", "converse"]



@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream_text",
    [_text_completion_text, _anthropic_messages_text, _responses_text],
    ids=["text_completion", "anthropic_messages", "responses"],
)
@pytest.mark.parametrize(
    ("model", "upstream_answer", "bounding_timeout"),
    [
        ("bedrock/converse/ai21.jamba-1-5-large-v1:0", CONVERSE_ANSWER, UPSTREAM_FIRST_BYTE_SECONDS / 2),
        ("bedrock/anthropic.claude-sonnet-4-5-v1:0", b"", STREAM_TIMEOUT_SECONDS),
    ],
    ids=["fake_stream", "live_stream"],
)
async def test_failed_stream_on_each_router_surface_reports_the_timeout_it_was_bounded_by(
    stream_text: Callable[[litellm.Router], Awaitable[str]],
    model: str,
    upstream_answer: bytes,
    bounding_timeout: float,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=_BEDROCK_RUNTIME_URL).mock(side_effect=slow_upstream_answer(upstream_answer))
    router: Final = _bedrock_router(model, timeout=UPSTREAM_FIRST_BYTE_SECONDS / 2, stream_timeout=STREAM_TIMEOUT_SECONDS)

    with pytest.raises(openai.APIError) as excinfo:
        await stream_text(router)
    assert getattr(excinfo.value, "timeout", None) == bounding_timeout

RECORDED_REASONING: Final = "Tokyo proper has about 14 million people."
RECORDED_THINKING_TOKENS: Final = 309
JSON_MODE_THINKING_CONVERSE_RESPONSE: Final = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": RECORDED_REASONING,
                            "signature": "EqoBCkgIBxABGAIiQL1",
                        }
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "tooluse_kZJMlvQmRJ6eAyJE5GIl7Q",
                        "name": "json_tool_call",
                        "input": {"city": "Tokyo", "population_millions": 14.0},
                    }
                },
            ],
        }
    },
    "stopReason": "tool_use",
    "usage": {"inputTokens": 473, "outputTokens": 390, "totalTokens": 863},
    "additionalModelResponseFields": {
        "usage": {"output_tokens_details": {"thinking_tokens": RECORDED_THINKING_TOKENS}}
    },
}


def _answer_with_json_mode_thinking_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=JSON_MODE_THINKING_CONVERSE_RESPONSE, request=request)


def _bedrock_claude_without_native_structured_output() -> str:
    return next(
        f"bedrock/{model}"
        for model, info in litellm.model_cost.items()
        if info.get("litellm_provider") == "bedrock_converse"
        and model.startswith("us.anthropic.claude")
        and info.get("supports_native_structured_output") is False
        and info.get("supports_reasoning") is True
    )


def _json_mode_thinking_kwargs() -> dict[str, object]:
    return {
        "model": _bedrock_claude_without_native_structured_output(),
        "messages": [{"role": "user", "content": "How many people live in Tokyo, in millions?"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "city_fact",
                "schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
                    "required": ["city", "population_millions"],
                },
            },
        },
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "aws_access_key_id": "fake",
        "aws_secret_access_key": "fake",
        "aws_region_name": "us-east-1",
    }


def _answer_fields(response: ModelResponse) -> tuple[object, ...]:
    choice: Final = response.choices[0]
    return (
        choice.finish_reason,
        choice.message.content,
        choice.message.tool_calls,
        getattr(choice.message, "reasoning_content", None),
        getattr(choice.message, "thinking_blocks", None),
        getattr(response.usage.completion_tokens_details, "reasoning_tokens", None),
    )


def _streamed_thinking_blocks(chunks: list[litellm.ModelResponseStream]) -> list[object]:
    deltas: Final = [chunk.choices[0].delta for chunk in chunks if chunk.choices]
    block_lists: Final = [getattr(delta, "thinking_blocks", None) or [] for delta in deltas]
    return list(chain.from_iterable(block_lists))


def test_fake_streamed_json_mode_answer_keeps_the_reasoning_of_the_non_streamed_answer() -> None:
    """Without native structured output, json_schema plus streaming fakes the stream from one Converse call. That fake
    stream dropped the reasoning, the signed thinking blocks and the reasoning tokens, and ended on tool_calls"""
    transport: Final = httpx.MockTransport(_answer_with_json_mode_thinking_response)
    client: Final = HTTPHandler(client=httpx.Client(transport=transport))
    kwargs: Final = _json_mode_thinking_kwargs()

    answer: Final = litellm.completion(client=client, **kwargs)
    chunks: Final = list(
        litellm.completion(client=client, stream=True, stream_options={"include_usage": True}, **kwargs)
    )
    rebuilt: Final = litellm.stream_chunk_builder(chunks, messages=kwargs["messages"])

    finish_reason, _, _, reasoning_content, _, reasoning_tokens = _answer_fields(answer)
    assert (finish_reason, reasoning_content, reasoning_tokens) == (
        "stop",
        RECORDED_REASONING,
        RECORDED_THINKING_TOKENS,
    )
    assert _answer_fields(rebuilt) == _answer_fields(answer)
    assert _streamed_thinking_blocks(chunks) == [
        {**block, "index": index} for index, block in enumerate(answer.choices[0].message.thinking_blocks)
    ]


@pytest.mark.asyncio
async def test_async_fake_streamed_json_mode_answer_keeps_the_reasoning_of_the_non_streamed_answer() -> None:
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_answer_with_json_mode_thinking_response))
    kwargs: Final = _json_mode_thinking_kwargs()

    answer: Final = await litellm.acompletion(client=client, **kwargs)
    stream: Final = await litellm.acompletion(
        client=client, stream=True, stream_options={"include_usage": True}, **kwargs
    )
    chunks: Final = [chunk async for chunk in stream]
    rebuilt: Final = litellm.stream_chunk_builder(chunks, messages=kwargs["messages"])

    finish_reason, _, _, reasoning_content, _, reasoning_tokens = _answer_fields(answer)
    assert (finish_reason, reasoning_content, reasoning_tokens) == (
        "stop",
        RECORDED_REASONING,
        RECORDED_THINKING_TOKENS,
    )
    assert _answer_fields(rebuilt) == _answer_fields(answer)
    assert _streamed_thinking_blocks(chunks) == [
        {**block, "index": index} for index, block in enumerate(answer.choices[0].message.thinking_blocks)
    ]


LLAMA_TEXT_TOOL_CALL_CONVERSE_RESPONSE: Final = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": '{"type": "function", "name": "get_weather", "parameters": {"city": "Paris"}}'}],
        }
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 175, "outputTokens": 23, "totalTokens": 198},
}


def _answer_with_llama_text_tool_call(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=LLAMA_TEXT_TOOL_CALL_CONVERSE_RESPONSE, request=request)


def _llama_tool_kwargs() -> dict[str, object]:
    return {
        "model": "bedrock/us.meta.llama3-3-70b-instruct-v1:0",
        "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "aws_access_key_id": "fake",
        "aws_secret_access_key": "fake",
        "aws_region_name": "us-east-1",
    }


def _tool_answer(response: ModelResponse) -> tuple[object, ...]:
    choice: Final = response.choices[0]
    return (
        choice.finish_reason,
        choice.message.content,
        [(call.function.name, call.function.arguments) for call in choice.message.tool_calls or []],
    )


def test_fake_streamed_text_tool_call_arrives_as_the_tool_call_of_the_non_streamed_answer() -> None:
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_answer_with_llama_text_tool_call)))
    kwargs: Final = _llama_tool_kwargs()

    answer: Final = litellm.completion(client=client, **kwargs)
    chunks: Final = list(litellm.completion(client=client, stream=True, **kwargs))

    assert _tool_answer(answer) == ("tool_calls", None, [("get_weather", '{"city": "Paris"}')])
    assert _tool_answer(litellm.stream_chunk_builder(chunks, messages=kwargs["messages"])) == _tool_answer(answer)


@pytest.mark.asyncio
async def test_async_fake_streamed_text_tool_call_arrives_as_the_tool_call_of_the_non_streamed_answer() -> None:
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_answer_with_llama_text_tool_call))
    kwargs: Final = _llama_tool_kwargs()

    answer: Final = await litellm.acompletion(client=client, **kwargs)
    stream: Final = await litellm.acompletion(client=client, stream=True, **kwargs)
    chunks: Final = [chunk async for chunk in stream]

    assert _tool_answer(answer) == ("tool_calls", None, [("get_weather", '{"city": "Paris"}')])
    assert _tool_answer(litellm.stream_chunk_builder(chunks, messages=kwargs["messages"])) == _tool_answer(answer)


def test_fake_streamed_json_mode_answer_closes_merged_reasoning_before_the_answer() -> None:
    """A fake stream sends reasoning and answer in one chunk, so merging reasoning into content must open and close
    the think block around the reasoning in that chunk, before the answer"""
    client: Final = HTTPHandler(
        client=httpx.Client(transport=httpx.MockTransport(_answer_with_json_mode_thinking_response))
    )
    kwargs: Final = _json_mode_thinking_kwargs()

    answer: Final = litellm.completion(client=client, **kwargs)
    chunks: Final = list(
        litellm.completion(client=client, stream=True, merge_reasoning_content_in_choices=True, **kwargs)
    )

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == (
        f"<think>{answer.choices[0].message.reasoning_content}</think>{answer.choices[0].message.content}"
    )


_UPSTREAM_ERROR_BODY: Final = '{"message":"The request is invalid."}'


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize(
    "model", ["us.anthropic.claude-sonnet-4-5-20250929-v1:0", "us.anthropic.claude-opus-4-8"], ids=["sonnet", "opus"]
)
@pytest.mark.parametrize(
    ("upstream_status", "error_class", "status_code"),
    [
        (400, litellm.BadRequestError, 400),
        (429, litellm.RateLimitError, 429),
        (500, litellm.ServiceUnavailableError, 503),
        (503, litellm.ServiceUnavailableError, 503),
    ],
)
def test_sync_converse_streaming_upstream_error_carries_the_upstream_body_as_text(
    model: str,
    upstream_status: int,
    error_class: type[litellm.BadRequestError | litellm.RateLimitError | litellm.ServiceUnavailableError],
    status_code: int,
) -> None:
    client: Final = HTTPHandler(
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(upstream_status, content=_UPSTREAM_ERROR_BODY.encode())
            )
        )
    )

    with pytest.raises(error_class) as raised:
        litellm.completion(
            model=f"bedrock/converse/{model}",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            num_retries=0,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
            client=client,
        )

    assert raised.value.status_code == status_code
    assert raised.value.message == f"litellm.{error_class.__name__}: BedrockException - {_UPSTREAM_ERROR_BODY}"


@pytest.mark.parametrize("stream", (False, True))
def test_converse_profile_override_records_cost_without_changing_response_model(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("LITELLM_RUST", "0")
    configuration.rust(False)
    model: Final = "openai.gpt-5.6-sol"
    profile: Final = f"us.{model}"
    prices: Final = litellm.get_model_info(profile, custom_llm_provider="bedrock")

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/model/{profile}/converse"
        return httpx.Response(200, json=CONVERSE_RESPONSE)

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        response: Final = litellm.completion(
            model=f"bedrock/converse/{model}",
            model_id=profile,
            messages=[{"role": "user", "content": "hi"}],
            api_key="test-bearer",
            aws_region_name="us-east-1",
            client=HTTPHandler(client=transport),
            stream=stream,
            fake_stream=stream,
            max_tokens=16,
        )
        result: Final = litellm.stream_chunk_builder(list(response)) if stream else response

    assert result is not None
    assert result.model == model
    assert result.choices[0].message.content == "hi"
    cost: Final = (
        response.logging_obj.response_cost_calculator(result) if stream else result._hidden_params.get("response_cost")
    )
    assert cost == pytest.approx(
        CONVERSE_RESPONSE["usage"]["inputTokens"] * prices["input_cost_per_token"]
        + CONVERSE_RESPONSE["usage"]["outputTokens"] * prices["output_cost_per_token"]
    )
