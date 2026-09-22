from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import BaseModel, TypeAdapter

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])


class CapturingTransport(httpx.AsyncBaseTransport, httpx.BaseTransport):
    def __init__(self, response: BaseModel) -> None:
        self._response: Final = response
        self.request_bodies: tuple[Mapping[str, object], ...] = ()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._respond(request.read())

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return self._respond(await request.aread())

    def _respond(self, body: bytes) -> httpx.Response:
        self.request_bodies = (*self.request_bodies, _JSON_OBJECT.validate_json(body))
        return httpx.Response(200, json=self._response.model_dump(mode="json"))
