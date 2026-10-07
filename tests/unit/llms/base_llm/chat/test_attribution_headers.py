"""
Provider attribution headers (`BaseConfig.get_attribution_headers`) must reach
the outbound request on every OpenAI-compatible chat path, and a caller header
with the same name must win.

Requests go through a real `litellm.completion` into an in-process httpx
transport that records what would have been sent, because the default path
(OpenAI SDK) never calls `validate_environment`.
"""

import json
from collections.abc import AsyncIterable, Iterable
from typing import Final, cast

import httpx
import openai
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import with_attribution_headers
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

_API_BASE: Final = "https://provider.invalid/v1"
_COMPLETION_BODY: Final = json.dumps(
    {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
).encode()
_STREAM_BODY: Final = (
    "data: "
    + json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        }
    )
    + "\n\ndata: [DONE]\n\n"
).encode()


class _HeaderCapturingTransport(httpx.BaseTransport, httpx.AsyncBaseTransport):
    """Records each outbound request's headers and answers like a chat completions server."""

    def __init__(self) -> None:
        self.sent: tuple[httpx.Headers, ...] = ()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._respond(request, request.read())

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._respond(request, await request.aread())

    def _respond(self, request: httpx.Request, body: bytes) -> httpx.Response:
        self.sent = (*self.sent, request.headers)
        if json.loads(body).get("stream"):
            return httpx.Response(200, content=_STREAM_BODY, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, content=_COMPLETION_BODY, headers={"content-type": "application/json"})

    def last(self, header: str) -> list[str]:
        return self.sent[-1].get_list(header)


def _client(transport: _HeaderCapturingTransport, path: str, is_async: bool) -> object:
    if path == "sdk":
        if is_async:
            return openai.AsyncOpenAI(
                api_key="k", base_url=_API_BASE, http_client=httpx.AsyncClient(transport=transport)
            )
        return openai.OpenAI(api_key="k", base_url=_API_BASE, http_client=httpx.Client(transport=transport))
    if is_async:
        return AsyncHTTPHandler(transport=transport)
    return HTTPHandler(client=httpx.Client(transport=transport))


@pytest.fixture
def transport() -> _HeaderCapturingTransport:
    return _HeaderCapturingTransport()


@pytest.fixture(params=["sdk", "http_handler"])
def handler_path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "http_handler":
        monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "true")
    else:
        monkeypatch.delenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", raising=False)
    return request.param


_NOVITA_MODEL: Final = "novita/meta-llama/llama-3.3-70b-instruct"

_ATTRIBUTED: Final = [
    pytest.param(_NOVITA_MODEL, "x-novita-source", id="novita"),
    pytest.param("perplexity/sonar", "x-pplx-integration", id="perplexity"),
]


def _drain(response: object) -> None:
    for _ in cast(Iterable[object], response):
        pass


async def _adrain(response: object) -> None:
    async for _ in cast(AsyncIterable[object], response):
        pass


@pytest.mark.parametrize(("model", "header"), _ATTRIBUTED)
@pytest.mark.parametrize("stream", [False, True])
def test_attribution_header_sent_sync(
    transport: _HeaderCapturingTransport, handler_path: str, model: str, header: str, stream: bool
) -> None:
    response: Final = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        api_base=_API_BASE,
        api_key="k",
        stream=stream,
        client=_client(transport, handler_path, is_async=False),
    )
    if stream:
        _drain(response)

    assert transport.last(header) == ["litellm"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "header"), _ATTRIBUTED)
@pytest.mark.parametrize("stream", [False, True])
async def test_attribution_header_sent_async(
    transport: _HeaderCapturingTransport, handler_path: str, model: str, header: str, stream: bool
) -> None:
    response: Final = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        api_base=_API_BASE,
        api_key="k",
        stream=stream,
        client=_client(transport, handler_path, is_async=True),
    )
    if stream:
        await _adrain(response)

    assert transport.last(header) == ["litellm"]


@pytest.mark.parametrize(("model", "header"), _ATTRIBUTED)
@pytest.mark.parametrize("header_kwarg", ["headers", "extra_headers"])
def test_caller_header_overrides_attribution_any_casing(
    transport: _HeaderCapturingTransport, handler_path: str, model: str, header: str, header_kwarg: str
) -> None:
    caller_headers: Final = {header.upper(): "my-app"}

    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        api_base=_API_BASE,
        api_key="k",
        client=_client(transport, handler_path, is_async=False),
        **{header_kwarg: caller_headers},
    )

    assert transport.last(header) == ["my-app"]
    assert caller_headers == {header.upper(): "my-app"}


def test_provider_without_attribution_sends_none(transport: _HeaderCapturingTransport, handler_path: str) -> None:
    litellm.completion(
        model="deepinfra/meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "hi"}],
        api_base=_API_BASE,
        api_key="k",
        client=_client(transport, handler_path, is_async=False),
    )

    assert transport.last("x-novita-source") == []
    assert transport.last("x-pplx-integration") == []


def test_global_litellm_headers_still_apply_and_are_not_mutated(
    transport: _HeaderCapturingTransport, handler_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_headers: Final = {"X-Global": "1"}
    monkeypatch.setattr(litellm, "headers", global_headers)

    litellm.completion(
        model=_NOVITA_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        api_base=_API_BASE,
        api_key="k",
        client=_client(transport, handler_path, is_async=False),
    )

    assert transport.last("x-global") == ["1"]
    assert transport.last("x-novita-source") == ["litellm"]
    assert global_headers == {"X-Global": "1"}


def test_with_attribution_headers_returns_headers_unchanged_when_nothing_to_add() -> None:
    headers: Final = {"A": "1"}

    assert with_attribution_headers({}, headers) is headers
    assert with_attribution_headers({}, None) is None
