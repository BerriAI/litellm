"""The Converse route end to end through `litellm.completion`, with the upstream scripted in process.

AWS credentials are static fakes or come from a stubbed STS or boto3 session, so nothing reaches AWS.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from datetime import datetime, timedelta, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import httpx
import openai
import pytest
import respx
from botocore.credentials import RefreshableCredentials
from botocore.exceptions import ClientError

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.rust_bridge import configuration
from litellm.types.utils import AdapterCompletionStreamWrapper, ModelResponse
from tests._support.stream_chunk_size import DEFAULT_CHUNKING_REQUESTS, ROUTER_CHUNK_SIZE_CASES, keys_at_every_depth
from tests.unit.llms.bedrock.event_loop_probe import EventLoopProbe
from tests.unit.llms.bedrock.slow_upstream import (
    CONVERSE_ANSWER,
    STREAM_TIMEOUT_SECONDS,
    UPSTREAM_FIRST_BYTE_SECONDS,
    slow_upstream_answer,
    slow_upstream_async_client,
    slow_upstream_sync_client,
)


@pytest.fixture(autouse=True)
def reset_rust_configuration(monkeypatch):
    monkeypatch.setenv("LITELLM_RUST", "1")
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


@pytest.fixture(autouse=True)
def isolate_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for env_var in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN", "AWS_PROFILE_NAME"):
        monkeypatch.delenv(env_var, raising=False)
    BaseAWSLLM._shared_iam_cache.flush_cache()
    yield
    BaseAWSLLM._shared_iam_cache.flush_cache()


CONVERSE_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
}

STATIC_AWS_CREDENTIALS: Final = MappingProxyType(
    {"aws_access_key_id": "AKIACONVERSETEST", "aws_secret_access_key": "converse-test-secret"}
)
HAIKU: Final = "bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0"
HI: Final = ({"role": "user", "content": "hi"},)


class RecordingUpstream:
    """Answers every request with one canned response and keeps each request it was sent."""

    def __init__(
        self,
        status_code: int = 200,
        body: bytes = CONVERSE_ANSWER,
        headers: Mapping[str, str] = MappingProxyType({"content-type": "application/json"}),
    ) -> None:
        self.status_code: Final = status_code
        self.body: Final = body
        self.headers: Final = headers
        self.requests: tuple[httpx.Request, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(self.status_code, content=self.body, headers=dict(self.headers), request=request)

    def sync_client(self) -> HTTPHandler:
        return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(self)))

    def async_client(self) -> AsyncHTTPHandler:
        return AsyncHTTPHandler(transport=httpx.MockTransport(self))

    @property
    def only_request(self) -> httpx.Request:
        assert len(self.requests) == 1, self.requests
        return self.requests[0]


async def complete(upstream: RecordingUpstream, *, sync: bool, **request: object) -> object:
    if sync:
        return litellm.completion(client=upstream.sync_client(), **request)
    return await litellm.acompletion(client=upstream.async_client(), **request)


SYNC_AND_ASYNC: Final = pytest.mark.parametrize("sync", [True, False], ids=["sync", "async"])


class PreCallCounter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.pre_calls = 0

    def log_pre_api_call(self, model, messages, kwargs):
        self.pre_calls += 1


@pytest.mark.asyncio
@SYNC_AND_ASYNC
async def test_each_request_is_logged_pre_call_exactly_once(monkeypatch: pytest.MonkeyPatch, sync: bool) -> None:
    counter: Final = PreCallCounter()
    monkeypatch.setattr(litellm, "input_callback", [counter])
    upstream: Final = RecordingUpstream()

    response: Final = await complete(
        upstream, sync=sync, model=HAIKU, messages=list(HI), aws_region_name="us-east-1", **STATIC_AWS_CREDENTIALS
    )

    assert response.choices[0].message.content == "hi"
    assert counter.pre_calls == 1


class _ProfileSession:
    """boto3.Session for a profile whose credentials are the probe's refreshable ones."""

    def __init__(self, credentials: RefreshableCredentials) -> None:
        self.credentials: Final = credentials

    def __call__(self, **_kwargs: object) -> "_ProfileSession":
        return self

    def get_credentials(self) -> RefreshableCredentials:
        return self.credentials


@pytest.mark.asyncio
@pytest.mark.parametrize("rust_enabled", (False, True))
async def test_async_completion_refreshes_credentials_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, rust_enabled: bool
) -> None:
    """Regression for issue #40165: botocore refreshes expiring credentials inside SigV4 signing with a
    blocking HTTP call, so the async route must sign on a worker thread to keep the loop serving."""
    monkeypatch.setenv("LITELLM_RUST", "1" if rust_enabled else "0")
    configuration.rust(rust_enabled)
    probe: Final = EventLoopProbe()
    upstream: Final = RecordingUpstream()
    release: Final = asyncio.create_task(probe.release_refresh_from_the_loop())

    with patch.object(boto3, "Session", _ProfileSession(probe.credentials())):
        response: Final = await litellm.acompletion(
            model=HAIKU,
            messages=list(HI),
            aws_profile_name="litellm-event-loop-probe",
            aws_region_name="us-west-2",
            client=upstream.async_client(),
        )
    await release

    assert response.choices[0].message.content == "hi"
    assert "Credential=AKIAREFRESHED/" in upstream.only_request.headers["Authorization"]
    assert probe.served_during_refresh is True


@pytest.mark.parametrize("configured_through", ["env_var", "api_key"])
def test_bearer_token_auth_never_runs_the_sigv4_credential_chain(
    monkeypatch: pytest.MonkeyPatch, configured_through: str
) -> None:
    """The deployment's AWS profile does not exist, so resolving SigV4 credentials raises. A bearer-token
    deployment must still serve the request, since the bearer token alone signs it."""
    if configured_through == "env_var":
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-bearer-token")
    upstream: Final = RecordingUpstream()

    response: Final = litellm.completion(
        model=HAIKU,
        messages=list(HI),
        aws_profile_name="litellm-no-such-aws-profile",
        aws_region_name="us-east-1",
        api_key="bedrock-bearer-token" if configured_through == "api_key" else None,
        client=upstream.sync_client(),
    )

    assert response.choices[0].message.content == "hi"
    assert upstream.only_request.headers["Authorization"] == "Bearer bedrock-bearer-token"


class _TrustPolicySTS:
    """STS whose role trust policy admits only an AssumeRole carrying exactly the expected extra params."""

    def __init__(self, expected: Mapping[str, object], access_key: str) -> None:
        self.expected: Final = expected
        self.access_key: Final = access_key

    def get_caller_identity(self) -> Mapping[str, str]:
        return {"Arn": "arn:aws:iam::111111111111:user/litellm-proxy-pod"}

    def assume_role(self, **params: object) -> Mapping[str, object]:
        extra: Final = {key: value for key, value in params.items() if key not in ("RoleArn", "RoleSessionName")}
        if json.dumps(extra, sort_keys=True) != json.dumps(self.expected, sort_keys=True):
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": f"trust policy rejected {extra}"}}, "AssumeRole"
            )
        return {
            "Credentials": {
                "AccessKeyId": self.access_key,
                "SecretAccessKey": "assumed-secret",
                "SessionToken": "assumed-session-token",
                "Expiration": datetime.now(timezone.utc) + timedelta(minutes=30),
            }
        }


_SESSION_TAGS: Final = [{"Key": "team", "Value": "genai"}]


@pytest.mark.parametrize(
    ("role_params", "sts_expects"),
    [
        ({"aws_session_tags": _SESSION_TAGS}, {"Tags": _SESSION_TAGS}),
        ({"aws_external_id": "converse-external-id"}, {"ExternalId": "converse-external-id"}),
    ],
    ids=["session_tags", "external_id"],
)
def test_assumed_role_session_signs_the_request_and_its_params_stay_out_of_the_body(
    role_params: Mapping[str, object], sts_expects: Mapping[str, object]
) -> None:
    """The role session STS admits signs the Converse call, and the role params never reach the body (#34069)."""
    upstream: Final = RecordingUpstream()
    sts: Final = _TrustPolicySTS(sts_expects, access_key="ASIACONVERSEROLE")

    with patch.object(boto3, "client", return_value=sts):
        response: Final = litellm.completion(
            model=HAIKU,
            messages=list(HI),
            aws_region_name="us-east-1",
            aws_access_key_id="AKIACONVERSECALLER",
            aws_secret_access_key="pod-caller-secret",
            aws_role_name="arn:aws:iam::999999999999:role/litellm-converse-role",
            aws_session_name="litellm-converse-session",
            client=upstream.sync_client(),
            **role_params,
        )

    assert response.choices[0].message.content == "hi"
    assert "Credential=ASIACONVERSEROLE/" in upstream.only_request.headers["Authorization"]
    assert keys_at_every_depth(json.loads(upstream.only_request.content)).isdisjoint(role_params)


_OUT_OF_BODY_PARAMS: Final = MappingProxyType(
    {
        "aws_access_key_id": "AKIAOUTOFBODY",
        "aws_secret_access_key": "out-of-body-secret",
        "aws_session_token": "out-of-body-session-token",
        "aws_sts_endpoint": "https://sts.out-of-body.example",
        "aws_external_id": "out-of-body-external-id",
        "aws_region_name": "eu-central-1",
        "aws_bedrock_runtime_endpoint": "https://bedrock-runtime.out-of-body.example",
        "model_id": "anthropic.claude-haiku-4-5-20251001-v1:0",
    }
)


@pytest.mark.asyncio
@SYNC_AND_ASYNC
@pytest.mark.parametrize(
    "stream_params",
    [{}, {"stream": True}, {"stream": True, "fake_stream": True}],
    ids=["non_stream", "live_stream", "fake_stream"],
)
async def test_transport_params_reach_the_url_and_signature_but_never_the_body(
    sync: bool, stream_params: Mapping[str, object]
) -> None:
    live_stream: Final = stream_params == {"stream": True}
    upstream: Final = RecordingUpstream(body=b"" if live_stream else CONVERSE_ANSWER)

    await complete(upstream, sync=sync, model=HAIKU, messages=list(HI), **_OUT_OF_BODY_PARAMS, **stream_params)

    sent: Final = upstream.only_request
    body: Final = json.loads(sent.content)
    assert body["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert keys_at_every_depth(body).isdisjoint({*_OUT_OF_BODY_PARAMS, *stream_params}), body
    assert not any(str(value) in sent.content.decode() for value in _OUT_OF_BODY_PARAMS.values()), body
    assert sent.url.host == "bedrock-runtime.out-of-body.example"
    assert "Credential=AKIAOUTOFBODY/" in sent.headers["Authorization"]
    assert "/eu-central-1/bedrock/aws4_request" in sent.headers["Authorization"]


_INFERENCE_PROFILE_ARN: Final = "arn:aws:bedrock:us-east-1:12345678910:application-inference-profile/ujdtmcirjhevpi"
_ENCODED_INFERENCE_PROFILE_ARN: Final = (
    "arn%3Aaws%3Abedrock%3Aus-east-1%3A12345678910%3Aapplication-inference-profile%2Fujdtmcirjhevpi"
)


@pytest.mark.asyncio
@SYNC_AND_ASYNC
@pytest.mark.parametrize(
    ("model", "request_params", "expected_region", "expected_path"),
    [
        (
            HAIKU,
            {"aws_region_name": "us-east-1"},
            "us-east-1",
            "/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
        ),
        (
            HAIKU,
            {"aws_region_name": "us-east-1", "stream": True},
            "us-east-1",
            "/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse-stream",
        ),
        ("bedrock/ap-northeast-1/moonshotai.kimi-k2.5", {}, "ap-northeast-1", "/model/moonshotai.kimi-k2.5/converse"),
        (
            "bedrock/us-west-2/anthropic.claude-haiku-4-5-20251001-v1:0",
            {},
            "us-west-2",
            "/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
        ),
        (
            "bedrock/ap-northeast-1/moonshotai.kimi-k2.5",
            {"aws_region_name": "eu-west-1"},
            "eu-west-1",
            "/model/moonshotai.kimi-k2.5/converse",
        ),
        (
            "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            {"aws_region_name": "us-east-1"},
            "us-east-1",
            "/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/converse",
        ),
        (
            HAIKU,
            {"aws_region_name": "us-east-1", "model_id": _INFERENCE_PROFILE_ARN},
            "us-east-1",
            f"/model/{_ENCODED_INFERENCE_PROFILE_ARN}/converse",
        ),
        (
            "bedrock/converse/nova-2/arn:aws:bedrock:us-east-1:123456789012:imported-model/abc123",
            {"aws_region_name": "us-east-1"},
            "us-east-1",
            "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Aimported-model%2Fabc123/converse",
        ),
    ],
    ids=[
        "converse",
        "converse_stream",
        "region_in_model_path",
        "region_in_model_path_with_colon_model_id",
        "explicit_region_beats_model_path",
        "cross_region_inference_prefix_is_not_a_region",
        "model_id_param",
        "nova_spec_prefix",
    ],
)
async def test_request_goes_to_the_regional_runtime_path_for_the_model_and_is_signed_for_that_region(
    sync: bool, model: str, request_params: Mapping[str, object], expected_region: str, expected_path: str
) -> None:
    upstream: Final = RecordingUpstream(body=b"" if request_params.get("stream") else CONVERSE_ANSWER)

    await complete(upstream, sync=sync, model=model, messages=list(HI), **STATIC_AWS_CREDENTIALS, **request_params)

    sent: Final = upstream.only_request
    assert sent.url.host == f"bedrock-runtime.{expected_region}.amazonaws.com"
    assert sent.url.raw_path.decode() == expected_path
    assert f"/{expected_region}/bedrock/aws4_request" in sent.headers["Authorization"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("local_beta_headers_config")
@SYNC_AND_ASYNC
async def test_unsupported_anthropic_beta_is_dropped_from_the_header_and_the_body(sync: bool) -> None:
    upstream: Final = RecordingUpstream()

    await complete(
        upstream,
        sync=sync,
        model="bedrock/converse/anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=list(HI),
        extra_headers={"anthropic-beta": "context-1m-2025-08-07,files-api-2025-04-14"},
        aws_region_name="us-east-1",
        **STATIC_AWS_CREDENTIALS,
    )

    sent: Final = upstream.only_request
    assert sent.headers["anthropic-beta"] == "context-1m-2025-08-07"
    body: Final = json.loads(sent.content)
    assert body["additionalModelRequestFields"]["anthropic_beta"] == ["context-1m-2025-08-07"]


@pytest.mark.asyncio
@SYNC_AND_ASYNC
async def test_caller_cannot_forge_the_request_metadata_header_when_forwarding_is_on(
    monkeypatch: pytest.MonkeyPatch, sync: bool
) -> None:
    monkeypatch.setattr(litellm, "bedrock_request_metadata_fields", ["user_api_key_alias"])
    upstream: Final = RecordingUpstream()

    await complete(
        upstream,
        sync=sync,
        model=HAIKU,
        messages=list(HI),
        metadata={"user_api_key_alias": "real-alias"},
        extra_headers={"X-Amzn-Bedrock-Request-Metadata": json.dumps({"user_api_key_alias": "forged"})},
        aws_region_name="us-east-1",
        **STATIC_AWS_CREDENTIALS,
    )

    sent: Final = upstream.only_request
    assert sent.headers.get_list("x-amzn-bedrock-request-metadata") == ['{"user_api_key_alias":"real-alias"}']
    assert json.loads(sent.content)["requestMetadata"] == {"user_api_key_alias": "real-alias"}


@pytest.mark.asyncio
@SYNC_AND_ASYNC
async def test_converse_like_endpoint_gets_the_request_as_is_and_unsigned(sync: bool) -> None:
    upstream: Final = RecordingUpstream()

    response: Final = await complete(
        upstream,
        sync=sync,
        model="bedrock/converse_like/my-gateway-model",
        messages=list(HI),
        api_base="https://converse-gateway.example/v1/converse",
    )

    assert response.choices[0].message.content == "hi"
    sent: Final = upstream.only_request
    assert str(sent.url) == "https://converse-gateway.example/v1/converse"
    assert "authorization" not in sent.headers
    assert json.loads(sent.content)["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]


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
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": {"type": "object"}},
                }
            },
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
        chunk.choices[0].delta.content or ""
        async for chunk in response
        if isinstance(chunk, litellm.ModelResponseStream)
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
            model="bedrock",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=slow_upstream_sync_client(),
        )


@pytest.mark.asyncio
async def test_async_live_stream_through_the_router_fails_at_the_stream_timeout() -> None:
    router: Final = _router_with_a_stream_timeout_shorter_than_the_upstreams_first_byte(
        "bedrock/anthropic.claude-sonnet-4-5-v1:0"
    )
    with pytest.raises(litellm.Timeout) as excinfo:
        await router.acompletion(
            model="bedrock",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=slow_upstream_async_client(),
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


# Bedrock's 403 for a role without the streaming action, seen live on us.anthropic.claude-sonnet-5 on 2026-11-06
_STREAM_PERMISSION_DENIAL: Final = json.dumps(
    {
        "Message": "User: arn:aws:sts::123456789012:assumed-role/invoke-only/qa is not authorized to perform: "
        "bedrock:InvokeModelWithResponseStream on resource: arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
        "us.anthropic.claude-opus-5-5 because no identity-based policy allows the "
        "bedrock:InvokeModelWithResponseStream action"
    }
).encode()


def _denies_streaming_then_answers(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/converse-stream"):
        return httpx.Response(403, content=_STREAM_PERMISSION_DENIAL, request=request)
    return httpx.Response(200, content=CONVERSE_ANSWER, headers={"content-type": "application/json"}, request=request)


@pytest.mark.asyncio
async def test_a_denied_native_output_stream_is_sent_again_buffered_with_native_output(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=_BEDROCK_RUNTIME_URL).mock(side_effect=_denies_streaming_then_answers)
    router: Final = _bedrock_router(
        "bedrock/converse/us.anthropic.claude-opus-5-5", timeout=60, stream_timeout=STREAM_TIMEOUT_SECONDS
    )

    response: Final = await router.acompletion(
        model="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        thinking={"type": "disabled"},
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": _ANSWER_SCHEMA}},
    )

    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == "hi"
    assert _sent_paths(respx_mock) == ["converse-stream", "converse"]
    retry_body: Final = json.loads(respx_mock.calls[1].request.content)
    assert "textFormat" in retry_body.get("outputConfig", {}) and "toolConfig" not in retry_body


_INVOKE_ANSWER: Final = json.dumps(
    {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": '{"answer": "hi"}'}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
).encode()


def _denies_invoke_streaming_then_answers(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/invoke-with-response-stream"):
        return httpx.Response(403, content=_STREAM_PERMISSION_DENIAL, request=request)
    return httpx.Response(200, content=_INVOKE_ANSWER, headers={"content-type": "application/json"}, request=request)


@pytest.mark.asyncio
async def test_a_denied_native_output_stream_on_the_invoke_route_is_sent_again_buffered_to_invoke(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=_BEDROCK_RUNTIME_URL).mock(side_effect=_denies_invoke_streaming_then_answers)
    router: Final = _bedrock_router(
        "bedrock/invoke/us.anthropic.claude-opus-5-5", timeout=60, stream_timeout=STREAM_TIMEOUT_SECONDS
    )

    response: Final = await router.acompletion(
        model="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        thinking={"type": "disabled"},
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": _ANSWER_SCHEMA}},
    )

    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == '{"answer": "hi"}'
    assert _sent_paths(respx_mock) == ["invoke-with-response-stream", "invoke"]
    retry_body: Final = json.loads(respx_mock.calls[1].request.content)
    assert "format" in retry_body.get("output_config", {}) and "tools" not in retry_body


_INVOKE_JSON_TOOL_ANSWER: Final = json.dumps(
    {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "json_tool_call", "input": {"answer": "hi"}}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
).encode()


def _denies_invoke_streaming_and_rejects_native_output(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/invoke-with-response-stream"):
        return httpx.Response(403, content=_STREAM_PERMISSION_DENIAL, request=request)
    if "format" in json.loads(request.content).get("output_config", {}):
        return httpx.Response(400, content=_NATIVE_OUTPUT_REJECTION, request=request)
    return httpx.Response(
        200, content=_INVOKE_JSON_TOOL_ANSWER, headers={"content-type": "application/json"}, request=request
    )


@pytest.mark.asyncio
async def test_a_rejected_schema_after_a_denied_invoke_stream_falls_back_to_the_json_tool_still_buffered(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(url__regex=_BEDROCK_RUNTIME_URL).mock(
        side_effect=_denies_invoke_streaming_and_rejects_native_output
    )
    router: Final = _bedrock_router(
        "bedrock/invoke/us.anthropic.claude-opus-5-5", timeout=60, stream_timeout=STREAM_TIMEOUT_SECONDS
    )

    response: Final = await router.acompletion(
        model="bedrock",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        thinking={"type": "disabled"},
        response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": _ANSWER_SCHEMA}},
    )

    assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == '{"answer": "hi"}'
    assert _sent_paths(respx_mock) == ["invoke-with-response-stream", "invoke", "invoke"]
    json_tool_body: Final = json.loads(respx_mock.calls[2].request.content)
    assert "format" not in json_tool_body.get("output_config", {}) and "tools" in json_tool_body


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
    router: Final = _bedrock_router(
        model, timeout=UPSTREAM_FIRST_BYTE_SECONDS / 2, stream_timeout=STREAM_TIMEOUT_SECONDS
    )

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


_REQUEST_ID_HEADERS: Final = MappingProxyType({"content-type": "application/json", "x-amzn-requestid": "req-123"})


@pytest.mark.asyncio
@SYNC_AND_ASYNC
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
async def test_bedrock_response_headers_are_forwarded_as_llm_provider_headers(sync: bool, stream: bool) -> None:
    upstream: Final = RecordingUpstream(body=b"" if stream else CONVERSE_ANSWER, headers=_REQUEST_ID_HEADERS)

    response: Final = await complete(
        upstream,
        sync=sync,
        model=HAIKU,
        messages=list(HI),
        stream=stream,
        aws_region_name="us-east-1",
        **STATIC_AWS_CREDENTIALS,
    )

    assert response._hidden_params["additional_headers"]["llm_provider-x-amzn-requestid"] == "req-123"


@pytest.mark.asyncio
@SYNC_AND_ASYNC
async def test_bedrock_error_keeps_the_bedrock_response_headers(sync: bool) -> None:
    upstream: Final = RecordingUpstream(
        status_code=500,
        body=json.dumps({"message": "Amazon Bedrock is unable to process your request."}).encode(),
        headers={"x-amzn-RequestId": "req-err-123", "x-amzn-ErrorType": "InternalServerException"},
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        await complete(
            upstream, sync=sync, model=HAIKU, messages=list(HI), aws_region_name="us-east-1", **STATIC_AWS_CREDENTIALS
        )

    assert exc_info.value.response.headers["x-amzn-requestid"] == "req-err-123"


def _stream_converse_completion_with_spied_client(model: str = HAIKU, **request: object) -> tuple[MagicMock, MagicMock]:
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client: Final = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)

    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        **request,
    )
    return mock_response.iter_bytes, client.post


def test_completion_plumbs_stream_chunk_size_through_converse() -> None:
    iter_bytes_spy, _ = _stream_converse_completion_with_spied_client(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", stream_chunk_size=2048
    )
    iter_bytes_spy.assert_called_once_with(chunk_size=2048)


def test_completion_stream_chunk_size_reaches_iter_bytes_but_not_converse_body() -> None:
    iter_bytes_spy, post_spy = _stream_converse_completion_with_spied_client(stream_chunk_size=64)

    iter_bytes_spy.assert_called_once_with(chunk_size=64)
    data: Final = post_spy.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data


@pytest.mark.parametrize("request_kwargs", DEFAULT_CHUNKING_REQUESTS)
def test_completion_uses_default_chunking_unless_a_valid_size_is_requested(
    request_kwargs: Mapping[str, object],
) -> None:
    iter_bytes_spy, _ = _stream_converse_completion_with_spied_client(**request_kwargs)

    iter_bytes_spy.assert_called_once_with(chunk_size=None)


async def _astream_converse_completion_with_spied_client(**request: object) -> tuple[MagicMock, AsyncMock]:
    async def _no_bytes(chunk_size: int | None = None) -> AsyncIterator[bytes]:
        return
        yield b""

    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    mock_response.aiter_bytes = MagicMock(return_value=_no_bytes())
    aiter_bytes_spy: Final = mock_response.aiter_bytes
    client: Final = AsyncHTTPHandler()
    client.post = AsyncMock(return_value=mock_response)

    await litellm.acompletion(
        model=HAIKU,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        client=client,
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
        aws_region_name="us-east-1",
        **request,
    )
    return aiter_bytes_spy, client.post


@pytest.mark.asyncio
async def test_acompletion_stream_chunk_size_reaches_aiter_bytes_but_not_converse_body() -> None:
    aiter_bytes_spy, post_spy = await _astream_converse_completion_with_spied_client(stream_chunk_size=64)

    aiter_bytes_spy.assert_called_once_with(chunk_size=64)
    data: Final = post_spy.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kwargs", DEFAULT_CHUNKING_REQUESTS)
async def test_acompletion_uses_default_chunking_unless_a_valid_size_is_requested(
    request_kwargs: Mapping[str, object],
) -> None:
    aiter_bytes_spy, _ = await _astream_converse_completion_with_spied_client(**request_kwargs)

    aiter_bytes_spy.assert_called_once_with(chunk_size=None)


CONVERSE_DEPLOYMENT: Final = MappingProxyType(
    {
        "model": HAIKU,
        "aws_access_key_id": "fake",
        "aws_secret_access_key": "fake",
        "aws_region_name": "us-east-1",
    }
)


@pytest.mark.parametrize("deployment_extras,expected_chunk_size", ROUTER_CHUNK_SIZE_CASES)
def test_router_deployment_stream_chunk_size_reaches_iter_bytes(
    deployment_extras: Mapping[str, object], expected_chunk_size: int | None
) -> None:
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_bytes = MagicMock(return_value=iter([]))
    client: Final = HTTPHandler()
    client.post = MagicMock(return_value=mock_response)
    router: Final = litellm.Router(
        model_list=[{"model_name": "converse-chunked", "litellm_params": {**CONVERSE_DEPLOYMENT, **deployment_extras}}],
        num_retries=0,
    )

    # the empty body is a stream that ends before its first chunk, which the Router now reports from the call
    with pytest.raises(litellm.BadGatewayError):
        router.completion(
            model="converse-chunked",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            client=client,
        )

    mock_response.iter_bytes.assert_called_once_with(chunk_size=expected_chunk_size)
    data: Final = client.post.call_args.kwargs["data"]
    assert "stream_chunk_size" not in keys_at_every_depth(json.loads(data)), data


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "non_stream"])
def test_converse_rejects_non_int_stream_chunk_size_before_calling_bedrock(stream: bool) -> None:
    upstream: Final = RecordingUpstream()

    with pytest.raises(litellm.BadRequestError):
        litellm.completion(
            model=HAIKU,
            messages=[{"role": "user", "content": "hi"}],
            stream=stream,
            client=upstream.sync_client(),
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-east-1",
            stream_chunk_size="sixty-four",
        )

    assert upstream.requests == ()
