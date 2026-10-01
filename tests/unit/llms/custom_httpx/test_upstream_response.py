from collections.abc import AsyncIterator, Iterator
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from litellm.litellm_core_utils.upstream_response_capture import UpstreamResponseCapture
from litellm.llms.custom_httpx.upstream_response import capture_async_openai_client, capture_openai_client

BODY: Final = b'{"id":"chatcmpl-probe","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}'


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
        captured: Final = capture_openai_client(client, capture, "attempt")
        response: Final = captured.chat.completions.create(model="probe", messages=[])
        assert response.choices[0].message.content == "ok"
        captured.close()
        assert not borrowed.is_closed
        assert client.chat.completions.create(model="probe", messages=[]).choices[0].message.content == "ok"
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
        captured: Final = capture_async_openai_client(client, capture, "attempt")
        response: Final = await captured.chat.completions.create(model="probe", messages=[])
        assert response.choices[0].message.content == "ok"
        await captured.close()
        assert not borrowed.is_closed
        assert (await client.chat.completions.create(model="probe", messages=[])).choices[0].message.content == "ok"
    assert body.closed
    assert tuple(item.status_code for item in capture.responses) == (429, 200)
    assert tuple(dict(item.headers)["x-request-id"] for item in capture.responses) == ("retry", "success")
