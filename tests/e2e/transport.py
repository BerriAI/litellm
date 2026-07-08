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
from e2e_http import (
    URL,
    AuthHeaders,
    FileUploadForm,
    ProbeResult,
    Result,
    StreamingResponse,
)


class Transport(Protocol):
    def post[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]: ...

    def stream(
        self, path: str, *, headers: BaseModel, json: BaseModel
    ) -> StreamingResponse: ...

    def send(
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        params: BaseModel | None = None,
        stream: bool = False,
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

    def upload[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        form: FileUploadForm,
        filename: str,
        content: bytes,
        params: BaseModel | None = None,
        response_type: type[R],
    ) -> Result[R]: ...

    def download(self, path: str, *, headers: BaseModel) -> StreamingResponse: ...

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

    def send(
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        params: BaseModel | None = None,
        stream: bool = False,
    ) -> StreamingResponse:
        return e2e_http.send(
            self._url(path),
            headers=headers,
            json=json,
            params=params,
            stream=stream,
            timeout=self.request_timeout,
        )

    def probe(self, path: str, *, params: BaseModel) -> ProbeResult:
        return e2e_http.probe(
            self._url(path),
            headers=self.master,
            params=params,
            timeout=self.request_timeout,
        )

    def upload[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        form: FileUploadForm,
        filename: str,
        content: bytes,
        params: BaseModel | None = None,
        response_type: type[R],
    ) -> Result[R]:
        return e2e_http.upload(
            self._url(path),
            headers=headers,
            form=form,
            filename=filename,
            content=content,
            params=params,
            response_type=response_type,
            timeout=self.request_timeout,
        )

    def download(self, path: str, *, headers: BaseModel) -> StreamingResponse:
        return e2e_http.download(
            self._url(path), headers=headers, timeout=self.request_timeout
        )


# Top-level management/admin route groups. In a split deployment these are served
# by the control plane (a different service from the LLM data plane). LLM routes
# (/chat, /embeddings, and native passthrough like /gemini, /anthropic) are NOT
# here and fall through to the data plane. Matched as path prefixes.
CONTROL_PLANE_PREFIXES: tuple[str, ...] = (
    "/key",
    "/user",
    "/team",
    "/organization",
    "/customer",
    "/end_user",
    "/tag",
    "/budget",
    "/model/",
    "/spend",
    "/global",
    "/openapi.json",
)


def is_control_plane_path(path: str) -> bool:
    """True if `path` is a management/admin route (served by the control plane in a
    split deployment), false for LLM data-plane routes."""
    return path.startswith(CONTROL_PLANE_PREFIXES)


@dataclass(frozen=True, slots=True)
class SplitTransport:
    """A Transport that dispatches each call by path to one of two backends: the
    management/admin control plane or the LLM data plane.

    Litellm can run as a split control-plane/data-plane deployment where the two
    surfaces live on different services. Clients here stay plane-agnostic — they
    keep calling ``transport.post("/budget/new", ...)`` or
    ``transport.send("/chat/completions", ...)`` — and routing happens in one place
    by path (see ``CONTROL_PLANE_PREFIXES``). When ``control`` and ``data`` share a
    base URL (the monolithic default), routing is a no-op. ``bearer``/``master``
    are plane-agnostic (same master key both planes), so they come from ``data``.
    """

    data: HttpTransport
    control: HttpTransport

    def _route(self, path: str) -> HttpTransport:
        return self.control if is_control_plane_path(path) else self.data

    def bearer(self, key: str) -> AuthHeaders:
        return self.data.bearer(key)

    @property
    def master(self) -> AuthHeaders:
        return self.data.master

    def post[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        return self._route(path).post(
            path, headers=headers, json=json, response_type=response_type
        )

    def get[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        params: BaseModel,
        response_type: type[R],
    ) -> Result[R]:
        return self._route(path).get(
            path, headers=headers, params=params, response_type=response_type
        )

    def delete[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        return self._route(path).delete(
            path, headers=headers, json=json, response_type=response_type
        )

    def stream(
        self, path: str, *, headers: BaseModel, json: BaseModel
    ) -> StreamingResponse:
        return self._route(path).stream(path, headers=headers, json=json)

    def send(
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        params: BaseModel | None = None,
        stream: bool = False,
    ) -> StreamingResponse:
        return self._route(path).send(
            path, headers=headers, json=json, params=params, stream=stream
        )

    def probe(self, path: str, *, params: BaseModel) -> ProbeResult:
        return self._route(path).probe(path, params=params)

    def upload[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        form: FileUploadForm,
        filename: str,
        content: bytes,
        params: BaseModel | None = None,
        response_type: type[R],
    ) -> Result[R]:
        return self._route(path).upload(
            path,
            headers=headers,
            form=form,
            filename=filename,
            content=content,
            params=params,
            response_type=response_type,
        )

    def download(self, path: str, *, headers: BaseModel) -> StreamingResponse:
        return self._route(path).download(path, headers=headers)
