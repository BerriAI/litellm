from typing import Final, Generic, TypeVar

import httpx
from httpx._client import USE_CLIENT_DEFAULT, UseClientDefault
from httpx._types import AuthTypes, TimeoutTypes
from openai import AsyncOpenAI, OpenAI

from litellm.litellm_core_utils.upstream_response_capture import (
    UpstreamResponseCapture,
    async_send_with_capture,
    send_with_capture,
)

_Client = TypeVar("_Client", httpx.Client, httpx.AsyncClient)


class _BorrowedClient(Generic[_Client]):
    def __init__(self, client: _Client, capture: UpstreamResponseCapture, attempt_id: str) -> None:
        self._borrowed: Final = client
        self.build_request: Final = client.build_request
        self._capture: Final = capture
        self._attempt_id: Final = attempt_id
        self._closed = False

    @property
    def timeout(self) -> httpx.Timeout:
        return self._borrowed.timeout

    @timeout.setter
    def timeout(self, timeout: TimeoutTypes) -> None:
        self._borrowed.timeout = timeout

    @property
    def is_closed(self) -> bool:
        return self._closed or self._borrowed.is_closed


class _CaptureClient(_BorrowedClient[httpx.Client], httpx.Client):
    def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> httpx.Response:
        if self.is_closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return send_with_capture(
            self._borrowed,
            request,
            self._capture,
            self._attempt_id,
            stream,
            auth=auth,
            follow_redirects=follow_redirects,
        )

    def close(self) -> None:
        self._closed = True


class _AsyncCaptureClient(_BorrowedClient[httpx.AsyncClient], httpx.AsyncClient):
    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
        follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> httpx.Response:
        if self.is_closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return await async_send_with_capture(
            self._borrowed,
            request,
            self._capture,
            self._attempt_id,
            stream,
            auth=auth,
            follow_redirects=follow_redirects,
        )

    async def aclose(self) -> None:
        self._closed = True


def capture_openai_client(client: OpenAI, capture: UpstreamResponseCapture, attempt_id: str) -> OpenAI:
    borrowed: Final = client._client  # pyright: ignore[reportPrivateUsage]  # SDK exposes no HTTP client accessor
    return client.with_options(http_client=_CaptureClient(borrowed, capture, attempt_id))


def capture_async_openai_client(client: AsyncOpenAI, capture: UpstreamResponseCapture, attempt_id: str) -> AsyncOpenAI:
    borrowed: Final = client._client  # pyright: ignore[reportPrivateUsage]  # SDK exposes no HTTP client accessor
    return client.with_options(http_client=_AsyncCaptureClient(borrowed, capture, attempt_id))
