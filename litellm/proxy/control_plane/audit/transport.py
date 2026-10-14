"""The one HTTP call the HTTP sinks make, as a protocol so tests script the answer."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


class Poster(Protocol):
    """Raises only httpx errors and never retries on its own, the worker owns retries."""

    async def post(self, url: str, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response: ...


@dataclass(frozen=True, slots=True)
class HttpxPoster:
    """Holds the handler, not its client: the shared cache may close and replace the client between batches. Never
    follows a redirect, a sink URL that bounces to plain http would otherwise carry the credential and the records there."""

    handler: AsyncHTTPHandler

    async def post(self, url: str, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        return await self.handler.client.post(
            url, content=content, headers=dict(headers), timeout=timeout, follow_redirects=False
        )
