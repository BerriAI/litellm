from typing import Final

import httpx
from openai import AsyncOpenAI, OpenAI

from litellm.litellm_core_utils.upstream_response_capture import (
    UpstreamResponseCapture,
    async_send_with_capture,
    send_with_capture,
)


class _CaptureTransport(httpx.BaseTransport):
    def __init__(self, client: httpx.Client, capture: UpstreamResponseCapture, attempt_id: str) -> None:
        self._client: Final = client
        self._capture: Final = capture
        self._attempt_id: Final = attempt_id

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return send_with_capture(self._client, request, self._capture, self._attempt_id, stream=True)


class _AsyncCaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self, client: httpx.AsyncClient, capture: UpstreamResponseCapture, attempt_id: str) -> None:
        self._client: Final = client
        self._capture: Final = capture
        self._attempt_id: Final = attempt_id

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await async_send_with_capture(self._client, request, self._capture, self._attempt_id, stream=True)


def capture_openai_client(client: OpenAI, capture: UpstreamResponseCapture, attempt_id: str) -> OpenAI:
    borrowed: Final = client._client  # pyright: ignore[reportPrivateUsage]  # SDK exposes no HTTP client accessor
    return client.with_options(
        http_client=httpx.Client(
            transport=_CaptureTransport(borrowed, capture, attempt_id),
            headers=borrowed.headers,
            params=borrowed.params,
            cookies=borrowed.cookies,
            timeout=borrowed.timeout,
            base_url=borrowed.base_url,
        )
    )


def capture_async_openai_client(client: AsyncOpenAI, capture: UpstreamResponseCapture, attempt_id: str) -> AsyncOpenAI:
    borrowed: Final = client._client  # pyright: ignore[reportPrivateUsage]  # SDK exposes no HTTP client accessor
    return client.with_options(
        http_client=httpx.AsyncClient(
            transport=_AsyncCaptureTransport(borrowed, capture, attempt_id),
            headers=borrowed.headers,
            params=borrowed.params,
            cookies=borrowed.cookies,
            timeout=borrowed.timeout,
            base_url=borrowed.base_url,
        )
    )
