"""Transport: the typed request primitives clients use, behind a Protocol.

`Transport` is what each client depends on (composition + DI); `HttpTransport` is
the concrete frozen-slots dataclass that fulfils it via the e2e_http wrapper. No
client touches requests.* or builds raw dicts; they pass pydantic models here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

import e2e_http
from e2e_http import URL, AuthHeaders, ProbeResult, Result, StreamingResponse


class Transport(Protocol):
    def post[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]: ...

    def stream(
        self, path: str, *, headers: BaseModel, json: BaseModel
    ) -> StreamingResponse: ...

    def get[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        params: BaseModel,
        response_type: type[R],
    ) -> Result[R]: ...

    def delete[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]: ...

    def probe(self, path: str, *, params: BaseModel) -> ProbeResult: ...

    def bearer(self, key: str) -> AuthHeaders: ...

    @property
    def master(self) -> AuthHeaders: ...


@dataclass(frozen=True, slots=True)
class HttpTransport:
    base_url: str
    master_key: str
    request_timeout: float = 60.0

    def _url(self, path: str) -> URL:
        return URL(f"{self.base_url.rstrip('/')}{path}")

    def bearer(self, key: str) -> AuthHeaders:
        return AuthHeaders(authorization=f"Bearer {key}")

    @property
    def master(self) -> AuthHeaders:
        return self.bearer(self.master_key)

    def post[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        return e2e_http.post(
            self._url(path),
            headers=headers,
            json=json,
            response_type=response_type,
            timeout=self.request_timeout,
        )

    def get[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        params: BaseModel,
        response_type: type[R],
    ) -> Result[R]:
        return e2e_http.get(
            self._url(path),
            headers=headers,
            params=params,
            response_type=response_type,
            timeout=self.request_timeout,
        )

    def delete[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        return e2e_http.delete(
            self._url(path),
            headers=headers,
            json=json,
            response_type=response_type,
            timeout=self.request_timeout,
        )

    def stream(
        self, path: str, *, headers: BaseModel, json: BaseModel
    ) -> StreamingResponse:
        return e2e_http.stream(
            self._url(path), headers=headers, json=json, timeout=self.request_timeout
        )

    def probe(self, path: str, *, params: BaseModel) -> ProbeResult:
        return e2e_http.probe(
            self._url(path),
            headers=self.master,
            params=params,
            timeout=self.request_timeout,
        )
