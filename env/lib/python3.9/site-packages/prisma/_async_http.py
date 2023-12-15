import json
from typing import Any

import httpx

from ._types import Method
from .http_abstract import AbstractResponse, AbstractHTTP


__all__ = ('HTTP', 'Response', 'client')


class HTTP(AbstractHTTP[httpx.AsyncClient, httpx.Response]):
    session: httpx.AsyncClient

    __slots__ = ()

    async def download(self, url: str, dest: str) -> None:
        async with self.session.stream('GET', url, timeout=None) as resp:
            resp.raise_for_status()
            with open(dest, 'wb') as fd:
                async for chunk in resp.aiter_bytes():
                    fd.write(chunk)

    async def request(
        self, method: Method, url: str, **kwargs: Any
    ) -> 'Response':
        return Response(await self.session.request(method, url, **kwargs))

    def open(self) -> None:
        self.session = httpx.AsyncClient(**self.session_kwargs)

    async def close(self) -> None:
        if self.should_close():
            await self.session.aclose()

            # mypy doesn't like us assigning None as the type of
            # session is not optional, however the argument that
            # the setter takes is optional, so this is fine
            self.session = None  # type: ignore[assignment]


client: HTTP = HTTP()


class Response(AbstractResponse[httpx.Response]):
    __slots__ = ()

    @property
    def status(self) -> int:
        return self.original.status_code

    @property
    def headers(self) -> httpx.Headers:
        return self.original.headers

    async def json(self, **kwargs: Any) -> Any:
        return json.loads(await self.original.aread(), **kwargs)

    async def text(self, **kwargs: Any) -> str:
        return ''.join(
            [part async for part in self.original.aiter_text(**kwargs)]
        )
