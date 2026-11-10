import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from functools import wraps
from typing import Final

import httpx
import httpx2
from pytest_httpx2._plugin import HTTPCore2Mocker
from respx import MockResponse as MockResponse
from respx import MockRouter as MockRouter
from respx import Route as Route
from respx import Router as Router
from respx import models as models
from respx.mocks import HTTPCoreMocker


class HTTPCore2AndLegacyMocker(HTTPCore2Mocker):
    name = "httpcore2_and_legacy"
    targets = HTTPCore2Mocker.targets + HTTPCoreMocker.targets


mock: Final = MockRouter(using=HTTPCore2AndLegacyMocker.name, assert_all_called=False)
post: Final = mock.post
get: Final = mock.get
delete: Final = mock.delete
put: Final = mock.put
patch: Final = mock.patch
head: Final = mock.head
options: Final = mock.options
request: Final = mock.request
route: Final = mock.route
add: Final = mock.add
calls: Final = mock.calls


def request_for_respx(value: httpx2.Request) -> httpx.Request:
    return httpx.Request(value.method, str(value.url), headers=value.headers.raw, content=value.content)


class _SyncResponseStream(httpx.SyncByteStream):
    def __init__(self, response: httpx2.Response) -> None:
        self._response: Final = response

    def __iter__(self) -> Iterator[bytes]:
        yield from self._response.iter_raw()

    def close(self) -> None:
        self._response.close()


class _AsyncResponseStream(httpx.AsyncByteStream):
    def __init__(self, response: httpx2.Response) -> None:
        self._response: Final = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_raw():
            yield chunk

    async def aclose(self) -> None:
        await self._response.aclose()


def response_for_respx(value: object) -> object:
    if not isinstance(value, httpx2.Response):
        return value
    if value.is_stream_consumed:
        headers: Final = tuple(
            (name, data)
            for name, data in value.headers.raw
            if name.lower() not in (b"content-encoding", b"content-length")
        )
        return httpx.Response(value.status_code, headers=headers, content=value.content, extensions=value.extensions)
    stream: Final = (
        _SyncResponseStream(value) if isinstance(value.stream, httpx2.SyncByteStream) else _AsyncResponseStream(value)
    )
    return httpx.Response(value.status_code, headers=value.headers.raw, stream=stream, extensions=value.extensions)


async def _response_after_await(result: Awaitable[object]) -> object:
    return response_for_respx(await result)


def _callback_for_respx(callback: Callable[..., object]) -> Callable[..., object]:
    @wraps(callback)
    def call(*args: object, **kwargs: object) -> object:
        result: Final = callback(*args, **kwargs)
        return _response_after_await(result) if inspect.isawaitable(result) else response_for_respx(result)

    return call


def side_effect_for_respx(value: object) -> object:
    if isinstance(value, type) and issubclass(value, BaseException):
        return value
    if callable(value):
        return _callback_for_respx(value)
    if isinstance(value, (tuple, list, Iterator)):
        return (response_for_respx(item) for item in value)
    return value
