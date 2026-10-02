from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Final, Literal

import httpx
import pytest
from httpx._client import USE_CLIENT_DEFAULT, UseClientDefault
from httpx._types import (
    CookieTypes,
    HeaderTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestFiles,
    TimeoutTypes,
)
from openai import AsyncOpenAI, AzureOpenAI, OpenAI

from litellm.litellm_core_utils.upstream_response_capture import UpstreamResponseCapture, upstream_attempt
from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.http_handler import blocked_cookie_jar
from litellm.llms.custom_httpx.upstream_response import (
    install_openai_capture_hook,
)
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.openai import OpenAIChatCompletion

BODY: Final = b'{"id":"chatcmpl-probe","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}'


def test_openai_client_acquisition_is_idempotent_and_keeps_hook_order() -> None:
    capture: Final = UpstreamResponseCapture()
    hooks_observed_capture: list[bool] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-request-id": "visible"}, content=BODY)

    def response_hook(response: httpx.Response) -> None:
        hooks_observed_capture.append(bool(capture.responses))

    with httpx.Client(
        transport=httpx.MockTransport(upstream),
        event_hooks={"response": [response_hook]},
    ) as borrowed:
        client: Final = OpenAI(api_key="synthetic", http_client=borrowed, max_retries=0)
        getter: Final = OpenAIChatCompletion()
        first: Final = getter._get_openai_client(is_async=False, api_key="synthetic", client=client, max_retries=0)
        second: Final = getter._get_openai_client(is_async=False, api_key="synthetic", client=client, max_retries=0)
        assert first is client
        assert second is client

        assert client.chat.completions.create(model="probe", messages=[]).choices[0].message.content == "ok"
        assert capture.responses == ()
        with upstream_attempt(capture, "attempt"):
            assert client.chat.completions.create(model="probe", messages=[]).choices[0].message.content == "ok"

        assert tuple(hooks_observed_capture) == (False, True)
        assert tuple(response.status_code for response in capture.responses) == (200,)
        assert tuple(dict(response.headers)["x-request-id"] for response in capture.responses) == ("visible",)
        assert not borrowed.is_closed


def test_azure_client_acquisition_keeps_identity_and_captures_headers() -> None:
    capture: Final = UpstreamResponseCapture()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-request-id": "azure-visible"}, content=BODY)

    with httpx.Client(transport=httpx.MockTransport(upstream)) as borrowed:
        client: Final = AzureOpenAI(
            api_key="synthetic",
            azure_endpoint="https://upstream.invalid",
            api_version="2024-02-01-preview",
            http_client=borrowed,
            max_retries=0,
        )
        returned: Final = BaseAzureLLM().get_azure_openai_client(
            api_key="synthetic",
            api_base="https://upstream.invalid",
            api_version="2024-02-01-preview",
            client=client,
            model="probe",
        )
        assert returned is client
        with upstream_attempt(capture, "attempt"):
            assert client.chat.completions.create(model="probe", messages=[]).choices[0].message.content == "ok"
        assert tuple(response.status_code for response in capture.responses) == (200,)
        assert tuple(dict(response.headers)["x-request-id"] for response in capture.responses) == ("azure-visible",)
        assert not borrowed.is_closed


@pytest.mark.asyncio
async def test_async_openai_client_acquisition_is_idempotent_and_keeps_hook_order() -> None:
    capture: Final = UpstreamResponseCapture()
    hooks_observed_capture: list[bool] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-request-id": "visible"}, content=BODY)

    async def response_hook(response: httpx.Response) -> None:
        hooks_observed_capture.append(bool(capture.responses))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
        event_hooks={"response": [response_hook]},
    ) as borrowed:
        client: Final = AsyncOpenAI(api_key="synthetic", http_client=borrowed, max_retries=0)
        getter: Final = OpenAIChatCompletion()
        first: Final = getter._get_openai_client(is_async=True, api_key="synthetic", client=client, max_retries=0)
        second: Final = getter._get_openai_client(is_async=True, api_key="synthetic", client=client, max_retries=0)
        assert first is client
        assert second is client

        assert (await client.chat.completions.create(model="probe", messages=[])).choices[0].message.content == "ok"
        assert capture.responses == ()
        with upstream_attempt(capture, "attempt"):
            assert (
                await client.chat.completions.create(model="probe", messages=[])
            ).choices[0].message.content == "ok"

        assert tuple(hooks_observed_capture) == (False, True)
        assert tuple(response.status_code for response in capture.responses) == (200,)
        assert tuple(dict(response.headers)["x-request-id"] for response in capture.responses) == ("visible",)
        assert not borrowed.is_closed


def test_base_llm_does_not_close_a_borrowed_session(monkeypatch: pytest.MonkeyPatch) -> None:
    with httpx.Client() as borrowed:
        monkeypatch.setattr("litellm.client_session", borrowed)
        llm: Final = BaseLLM()
        session: Final = llm.create_client_session()
        assert session is borrowed
        llm._client_session = session
        llm.__exit__()
        assert not borrowed.is_closed


class Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, capture: UpstreamResponseCapture) -> None:
        self.capture: Final = capture
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        assert self.capture.responses
        yield BODY

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for value in self:
            yield value

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


def test_sdk_capture_preserves_request_settings_and_borrowed_client_ownership() -> None:
    capture: Final = UpstreamResponseCapture()
    body: Final = Body(capture)

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["x-custom"] == "kept"
        assert request.url.params["custom"] == "kept"
        assert request.headers["authorization"] == "Bearer synthetic"
        assert request.headers["cookie"] == "session=synthetic"
        assert request.extensions["timeout"]["read"] == 11
        return httpx.Response(200, headers={"x-request-id": "visible"}, stream=body)

    with httpx.Client(
        transport=httpx.MockTransport(upstream),
        headers={"x-custom": "kept"},
        params={"custom": "kept"},
        cookies={"session": "synthetic"},
        timeout=11,
    ) as borrowed:
        client: Final = OpenAI(api_key="synthetic", base_url="https://upstream.invalid/v1", http_client=borrowed)
        install_openai_capture_hook(client)
        with upstream_attempt(capture, "attempt"):
            response: Final = client.chat.completions.create(model="probe", messages=[])
        assert response.choices[0].message.content == "ok"
        assert not borrowed.is_closed
        assert client.chat.completions.create(model="probe", messages=[]).choices[0].message.content == "ok"
        assert len(capture.responses) == 1
    assert body.closed
    assert capture.snapshot() == (
        {
            "attempt_id": "attempt",
            "status_code": 200,
            "headers": (("x-request-id", "visible"),),
            "truncated": False,
        },
    )


@pytest.mark.asyncio
async def test_sdk_async_retries_are_captured_without_closing_borrowed_client() -> None:
    capture: Final = UpstreamResponseCapture()
    statuses: Final = iter((429, 200, 200))
    body: Final = Body(capture)

    def upstream(request: httpx.Request) -> httpx.Response:
        status: Final = next(statuses)
        if status == 429:
            return httpx.Response(
                429, headers={"x-request-id": "retry", "retry-after-ms": "1"}, json={"error": {"message": "retry"}}
            )
        return httpx.Response(200, headers={"x-request-id": "success"}, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as borrowed:
        client: Final = AsyncOpenAI(api_key="synthetic", http_client=borrowed, max_retries=1)
        install_openai_capture_hook(client)
        with upstream_attempt(capture, "attempt"):
            response: Final = await client.chat.completions.create(model="probe", messages=[])
        assert response.choices[0].message.content == "ok"
        assert not borrowed.is_closed
        assert (await client.chat.completions.create(model="probe", messages=[])).choices[0].message.content == "ok"
        assert len(capture.responses) == 2
    assert body.closed
    assert tuple(item.status_code for item in capture.responses) == (429, 200)
    assert tuple(dict(item.headers)["x-request-id"] for item in capture.responses) == ("retry", "success")


class SigningRequests:
    def build_request(
        self,
        method: str,
        url: httpx.URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        json: object = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        cookies: CookieTypes | None = None,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        extensions: Mapping[str, object] | None = None,
    ) -> httpx.Request:
        request: Final = httpx.Request(
            method,
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            extensions=extensions,
        )
        request.headers["x-custom-signature"] = "signed-by-original-client"
        return request


class SigningClient(SigningRequests, httpx.Client):
    pass


class AsyncSigningClient(SigningRequests, httpx.AsyncClient):
    pass


@dataclass(frozen=True)
class WireRequest:
    method: str
    url: str
    body: bytes
    cookie: str | None
    signature: str | None
    authorization: str | None
    headers: tuple[tuple[str, str], ...]


class CompatibilityUpstream:
    def __init__(self, signing: bool, redirects: bool) -> None:
        self.signing: Final = signing
        self.redirects: Final = redirects
        self.hooks: tuple[str, ...] = ()
        self.requests: tuple[WireRequest, ...] = ()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests = (
            *self.requests,
            WireRequest(
                request.method,
                str(request.url),
                request.content,
                request.headers.get("cookie"),
                request.headers.get("x-custom-signature"),
                request.headers.get("authorization"),
                tuple(sorted(request.headers.multi_items())),
            ),
        )
        if self.signing and request.headers.get("x-custom-signature") != "signed-by-original-client":
            return httpx.Response(401, json={"error": {"message": "signature missing"}})
        if self.redirects and request.url.path != "/redirect":
            return httpx.Response(307, headers={"location": "/redirect", "set-cookie": "route=synthetic; Path=/"})
        if len(self.requests) == (2 if self.redirects else 1):
            return httpx.Response(
                429,
                headers={
                    "set-cookie": "session=synthetic; Path=/",
                    "retry-after-ms": "0.001",
                    "x-request-id": "retry",
                },
                json={"error": {"message": "retry"}},
            )
        return httpx.Response(200, headers={"x-request-id": "success"}, content=BODY)

    def request_hook(self, request: httpx.Request) -> None:
        self.hooks = (*self.hooks, "request:" + request.url.path)

    def response_hook(self, response: httpx.Response) -> None:
        self.hooks = (*self.hooks, "response:" + str(response.status_code))

    async def async_request_hook(self, request: httpx.Request) -> None:
        self.request_hook(request)

    async def async_response_hook(self, response: httpx.Response) -> None:
        self.response_hook(response)


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    requests: tuple[WireRequest, ...]
    hooks: tuple[str, ...]
    cookies: tuple[tuple[str, str], ...]


def run_sync_contract(enabled: bool, signing: bool, redirects: bool) -> CompatibilityResult:
    capture: Final = UpstreamResponseCapture()
    upstream: Final = CompatibilityUpstream(signing, redirects)
    client_type: Final = SigningClient if signing else httpx.Client
    with client_type(
        transport=httpx.MockTransport(upstream),
        cookies={} if redirects else blocked_cookie_jar(),
        follow_redirects=redirects,
        auth=httpx.BasicAuth("probe", "synthetic"),
        event_hooks={"request": [upstream.request_hook], "response": [upstream.response_hook]},
    ) as borrowed:
        original: Final = OpenAI(api_key="synthetic", http_client=borrowed, max_retries=1)
        if enabled:
            install_openai_capture_hook(original)
        used: Final = original
        if enabled:
            with upstream_attempt(capture, "attempt"):
                assert used.chat.completions.create(model="probe", messages=[]).choices[0].message.content == "ok"
            assert not borrowed.is_closed
        else:
            assert used.chat.completions.create(model="probe", messages=[]).choices[0].message.content == "ok"
        statuses: Final = (307, 429, 307, 200) if redirects else (429, 200)
        assert tuple(item.status_code for item in capture.responses) == (statuses if enabled else ())
        return CompatibilityResult(upstream.requests, upstream.hooks, tuple(borrowed.cookies.items()))


@pytest.mark.parametrize("signing", (False, True), ids=("cookie-policy", "request-signing"))
@pytest.mark.parametrize("redirects", (False, True))
def test_capture_matches_original_sync_client(signing: bool, redirects: bool) -> None:
    expected: Final = run_sync_contract(False, signing, redirects)
    assert run_sync_contract(True, signing, redirects) == expected


async def run_async_contract(enabled: bool, signing: bool, redirects: bool) -> CompatibilityResult:
    capture: Final = UpstreamResponseCapture()
    upstream: Final = CompatibilityUpstream(signing, redirects)
    client_type: Final = AsyncSigningClient if signing else httpx.AsyncClient
    async with client_type(
        transport=httpx.MockTransport(upstream),
        cookies={} if redirects else blocked_cookie_jar(),
        follow_redirects=redirects,
        auth=httpx.BasicAuth("probe", "synthetic"),
        event_hooks={"request": [upstream.async_request_hook], "response": [upstream.async_response_hook]},
    ) as borrowed:
        original: Final = AsyncOpenAI(api_key="synthetic", http_client=borrowed, max_retries=1)
        if enabled:
            install_openai_capture_hook(original)
        used: Final = original
        if enabled:
            with upstream_attempt(capture, "attempt"):
                assert (
                    await used.chat.completions.create(model="probe", messages=[])
                ).choices[0].message.content == "ok"
            assert not borrowed.is_closed
        else:
            assert (
                await used.chat.completions.create(model="probe", messages=[])
            ).choices[0].message.content == "ok"
        statuses: Final = (307, 429, 307, 200) if redirects else (429, 200)
        assert tuple(item.status_code for item in capture.responses) == (statuses if enabled else ())
        return CompatibilityResult(upstream.requests, upstream.hooks, tuple(borrowed.cookies.items()))


@pytest.mark.asyncio
@pytest.mark.parametrize("signing", (False, True), ids=("cookie-policy", "request-signing"))
@pytest.mark.parametrize("redirects", (False, True))
async def test_capture_matches_original_async_client(signing: bool, redirects: bool) -> None:
    expected: Final = await run_async_contract(False, signing, redirects)
    assert await run_async_contract(True, signing, redirects) == expected


class FailingBody(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        raise httpx.ReadError("interrupted response")
        yield b""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self:
            yield chunk

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("failure", ("status", "read", "connect"))
def test_sdk_failure_contract(enabled: bool, failure: Literal["status", "read", "connect"]) -> None:
    from openai import APIConnectionError, AuthenticationError

    capture: Final = UpstreamResponseCapture()
    body: Final = FailingBody()

    def upstream(request: httpx.Request) -> httpx.Response:
        if failure == "connect":
            raise httpx.ConnectError("connection refused", request=request)
        if failure == "status":
            return httpx.Response(401, headers={"x-request-id": "failed"}, json={"error": {"message": "denied"}})
        return httpx.Response(200, headers={"x-request-id": "failed"}, stream=body)

    with httpx.Client(transport=httpx.MockTransport(upstream)) as borrowed:
        original: Final = OpenAI(api_key="synthetic", http_client=borrowed, max_retries=0)
        if enabled:
            install_openai_capture_hook(original)
        used: Final = original
        if enabled:
            with upstream_attempt(capture, "attempt"):
                with pytest.raises(AuthenticationError if failure == "status" else APIConnectionError):
                    used.chat.completions.create(model="probe", messages=[])
        else:
            with pytest.raises(AuthenticationError if failure == "status" else APIConnectionError):
                used.chat.completions.create(model="probe", messages=[])
        assert not borrowed.is_closed
    assert body.closed == (failure == "read")
    expected_status: Final = (401 if failure == "status" else 200,) if enabled and failure != "connect" else ()
    assert tuple(item.status_code for item in capture.responses) == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("failure", ("status", "read", "connect"))
async def test_async_sdk_failure_contract(enabled: bool, failure: Literal["status", "read", "connect"]) -> None:
    from openai import APIConnectionError, AuthenticationError

    capture: Final = UpstreamResponseCapture()
    body: Final = FailingBody()

    def upstream(request: httpx.Request) -> httpx.Response:
        if failure == "connect":
            raise httpx.ConnectError("connection refused", request=request)
        if failure == "status":
            return httpx.Response(401, headers={"x-request-id": "failed"}, json={"error": {"message": "denied"}})
        return httpx.Response(200, headers={"x-request-id": "failed"}, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as borrowed:
        original: Final = AsyncOpenAI(api_key="synthetic", http_client=borrowed, max_retries=0)
        if enabled:
            install_openai_capture_hook(original)
        used: Final = original
        if enabled:
            with upstream_attempt(capture, "attempt"):
                with pytest.raises(AuthenticationError if failure == "status" else APIConnectionError):
                    await used.chat.completions.create(model="probe", messages=[])
        else:
            with pytest.raises(AuthenticationError if failure == "status" else APIConnectionError):
                await used.chat.completions.create(model="probe", messages=[])
        assert not borrowed.is_closed
    assert body.closed == (failure == "read")
    expected_status: Final = (401 if failure == "status" else 200,) if enabled and failure != "connect" else ()
    assert tuple(item.status_code for item in capture.responses) == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("first_chunk", (False, True))
async def test_async_sdk_cancellation_closes_stream_without_buffering(enabled: bool, first_chunk: bool) -> None:
    import asyncio

    capture: Final = UpstreamResponseCapture()
    waiting: Final = asyncio.Event()
    blocked: Final = asyncio.Event()
    closed: Final = asyncio.Event()

    class StreamingBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            if first_chunk:
                yield b'data: {"id":"probe","choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
            waiting.set()
            await blocked.wait()
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            closed.set()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-request-id": "stream"}, stream=StreamingBody())

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as borrowed:
        original: Final = AsyncOpenAI(api_key="synthetic", http_client=borrowed, max_retries=0)
        if enabled:
            install_openai_capture_hook(original)
        used: Final = original
        attempt_context: Final = upstream_attempt(capture, "attempt") if enabled else nullcontext()
        with attempt_context:
            response: Final = await used.chat.completions.create(model="probe", messages=[], stream=True)
        assert not waiting.is_set()
        assert bool(capture.responses) == enabled

        async def consume() -> tuple[str | None, ...]:
            return tuple([chunk.choices[0].delta.content async for chunk in response])

        task: Final = asyncio.create_task(consume())
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert not borrowed.is_closed
    assert tuple(item.status_code for item in capture.responses) == ((200,) if enabled else ())


@pytest.mark.parametrize("enabled", (False, True))
def test_sync_sdk_stream_is_lazy_and_closes_without_closing_pool(enabled: bool) -> None:
    capture: Final = UpstreamResponseCapture()

    class StreamingBody(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.reads = 0
            self.closed = False

        def __iter__(self) -> Iterator[bytes]:
            self.reads += 1
            yield BODY

        def close(self) -> None:
            self.closed = True

    body: Final = StreamingBody()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-request-id": "stream"}, stream=body)

    with httpx.Client(transport=httpx.MockTransport(upstream)) as borrowed:
        original: Final = OpenAI(api_key="synthetic", http_client=borrowed)
        if enabled:
            install_openai_capture_hook(original)
        used: Final = original
        scope: Final = upstream_attempt(capture, "attempt") if enabled else nullcontext()
        with scope:
            with used.chat.completions.with_streaming_response.create(model="probe", messages=[]) as response:
                assert body.reads == 0
                assert bool(capture.responses) == enabled
                assert response.parse().choices[0].message.content == "ok"
                assert body.reads == 1
        assert body.closed
        assert not borrowed.is_closed
