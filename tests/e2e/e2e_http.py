"""The ONLY module permitted to call ``requests.*``.

Enforced by tests/code_coverage_tests/check_e2e_no_raw_requests.py. Every request
body / query / header / response is a pydantic model; outcomes are a tagged union
(``Result[R]``) so callers ``match`` on them instead of catching exceptions.

Named e2e_http (not http) so it does not shadow the stdlib ``http`` package that
requests itself imports.
"""

from __future__ import annotations

from typing import Generic, Iterator, Literal, NewType, TypeVar, cast

import pytest
import requests
from pydantic import BaseModel, ConfigDict, Field

URL = NewType("URL", str)


class Headers(BaseModel):
    """Base for header models. Subclasses may alias to hyphenated header names
    (e.g. ``x-litellm-api-key``); serialization uses by_alias."""

    model_config = ConfigDict(populate_by_name=True)


class AuthHeaders(Headers):
    # litellm accepts either; set whichever the call needs, leave the other None.
    authorization: str | None = None
    x_litellm_api_key: str | None = Field(default=None, alias="x-litellm-api-key")


class AnthropicHeaders(AuthHeaders):
    """Auth plus the ``anthropic-version`` header the Anthropic-native
    /v1/messages and /v1/messages/count_tokens routes expect. It is harmless on
    the other providers the proxy routes to, and matches what Claude Code sends
    on its own internal calls."""

    anthropic_version: str = Field(default="2023-06-01", alias="anthropic-version")


class NoBody(BaseModel):
    """Empty body/query for routes that take none."""


class FileUploadForm(BaseModel):
    """Multipart form fields for POST /v1/files. The file bytes are passed
    separately; `model` is not here because the proxy reads it from the query
    (?model=) not the form."""

    purpose: str = "batch"
    target_model_names: str | None = None
    custom_llm_provider: str | None = None


# ---------- Result types ----------

R = TypeVar("R", bound=BaseModel)


class Success(BaseModel, Generic[R]):
    kind: Literal["success"] = "success"
    data: R


class NetworkError(BaseModel):
    kind: Literal["network"] = "network"
    message: str


class UnauthorizedError(BaseModel):
    kind: Literal["unauthorized"] = "unauthorized"


class RateLimitedError(BaseModel):
    kind: Literal["rate_limited"] = "rate_limited"
    retry_after_seconds: int | None = None
    # litellm overloads 429 for budget_exceeded too, so keep the body to tell them apart.
    body: str = ""


class ValidationError(BaseModel):
    kind: Literal["validation"] = "validation"
    message: str


class UnknownApiError(BaseModel):
    kind: Literal["unknown"] = "unknown"
    status_code: int
    body: str


type Result[R: BaseModel] = (
    Success[R]
    | NetworkError
    | UnauthorizedError
    | RateLimitedError
    | ValidationError
    | UnknownApiError
)


class ProbeResult(BaseModel):
    """A route's reachability: status + body, no schema validation. Healthy ==
    route exists (not 404) and the handler did not crash (not 5xx)."""

    status_code: int
    body: str

    @property
    def healthy(self) -> bool:
        return 200 <= self.status_code < 500 and self.status_code != 404


class StreamingResponse(BaseModel):
    """Raw outcome for calls whose body is provider-native or streamed: status, the
    x-litellm-call-id header, the x-litellm-response-cost header (StandardLogging
    response_cost), the content-type (which tells streaming `text/event-stream` from
    non-streaming `application/json`), the response headers (lowercased names, e.g.
    the x-ratelimit-* pacing headers and retry-after on a 429), and the body.
    SpendLogs.request_id is the completion body id, not call_id. Used by passthrough
    and streaming, where one validated JSON model does not fit."""

    status_code: int
    call_id: str | None = None  # x-litellm-call-id header
    response_cost: float | None = None  # x-litellm-response-cost header
    content_type: str | None = None
    headers: dict[str, str] = {}
    body: str
    chunks: int = 0  # streamed events (0 for non-streaming)
    stream_events: list[str] = []
    # First in-stream error event, if any. A streamed call commits its HTTP 200
    # before the upstream completes, so upstream failures (e.g. insufficient
    # quota) arrive as SSE error events inside an otherwise-successful response;
    # the consumed body is elided, so this is the only place they surface.
    stream_error: str | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def is_streaming(self) -> bool:
        return "text/event-stream" in (self.content_type or "")


class BinaryStream(BaseModel):
    """Outcome of consuming a binary chunked response (e.g. TTS audio) as a stream.

    Unlike StreamingResponse, which line-splits an SSE text body, this iterates the
    raw bytes with iter_content and reports how many non-empty chunks arrived and
    the total byte count, so a caller can assert customer-observable streaming
    (multiple chunks, real bytes) without decoding the payload."""

    status_code: int
    content_type: str | None = None
    call_id: str | None = None
    transfer_encoding: str | None = None
    content_length: str | None = None
    error_body: str | None = None
    chunk_count: int = 0
    total_bytes: int = 0

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def chunked(self) -> bool:
        return "chunked" in (self.transfer_encoding or "")


def _hdr(resp: requests.Response, name: str) -> str | None:
    value = resp.headers.get(name)
    return value if isinstance(value, str) else None


def unwrap[R: BaseModel](result: Result[R]) -> R:
    match result:
        case Success(data=data):
            return data
        case _:
            raise AssertionError(result)


def is_ok[R: BaseModel](result: Result[R]) -> bool:
    match result:
        case Success():
            return True
        case _:
            return False


def require_successful_call(result: StreamingResponse) -> None:
    """A call that should have succeeded but didn't is a hard failure, never a skip:
    if the proxy can't make a call it's expected to, the test must fail."""
    if result.ok:
        return
    pytest.fail(
        f"upstream call failed (status {result.status_code}); body={result.body[:300]}"
    )


def _headers(headers: BaseModel) -> dict[str, str]:
    dumped: dict[str, object] = headers.model_dump(by_alias=True, exclude_none=True)
    return {key: str(value) for key, value in dumped.items()}


def _params(params: BaseModel | None) -> dict[str, str]:
    if params is None:
        return {}
    dumped: dict[str, object] = params.model_dump(by_alias=True, exclude_none=True)
    return {key: str(value) for key, value in dumped.items()}


def _classify[R: BaseModel](
    resp: requests.Response, response_type: type[R]
) -> Result[R]:
    if resp.status_code == 401:
        return UnauthorizedError()
    if resp.status_code == 429:
        return RateLimitedError(body=resp.text)
    if not resp.ok:
        return UnknownApiError(status_code=resp.status_code, body=resp.text)
    try:
        return Success(data=response_type.model_validate(resp.json()))
    except Exception as exc:  # noqa: BLE001 - any parse/validation failure is a value
        return ValidationError(message=str(exc))


def post[R: BaseModel](
    url: URL,
    *,
    headers: BaseModel,
    json: BaseModel,
    response_type: type[R],
    timeout: float = 30.0,
) -> Result[R]:
    try:
        resp = requests.post(
            str(url),
            headers=_headers(headers),
            json=json.model_dump(by_alias=True, exclude_none=True),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return NetworkError(message=str(exc))
    return _classify(resp, response_type)


def get[R: BaseModel](
    url: URL,
    *,
    headers: BaseModel,
    params: BaseModel,
    response_type: type[R],
    timeout: float = 30.0,
) -> Result[R]:
    try:
        resp = requests.get(
            str(url),
            headers=_headers(headers),
            params=params.model_dump(by_alias=True, exclude_none=True),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return NetworkError(message=str(exc))
    return _classify(resp, response_type)


def delete[R: BaseModel](
    url: URL,
    *,
    headers: BaseModel,
    json: BaseModel,
    response_type: type[R],
    params: BaseModel | None = None,
    timeout: float = 30.0,
) -> Result[R]:
    try:
        resp = requests.delete(
            str(url),
            headers=_headers(headers),
            json=json.model_dump(by_alias=True, exclude_none=True),
            params=_params(params),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return NetworkError(message=str(exc))
    return _classify(resp, response_type)


def patch[R: BaseModel](
    url: URL,
    *,
    headers: BaseModel,
    json: BaseModel,
    response_type: type[R],
    timeout: float = 30.0,
) -> Result[R]:
    try:
        resp = requests.patch(
            str(url),
            headers=_headers(headers),
            json=json.model_dump(by_alias=True, exclude_none=True),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return NetworkError(message=str(exc))
    return _classify(resp, response_type)


def probe(
    url: URL, *, headers: BaseModel, params: BaseModel, timeout: float = 30.0
) -> ProbeResult:
    try:
        resp = requests.get(
            str(url),
            headers=_headers(headers),
            params=params.model_dump(by_alias=True, exclude_none=True),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return ProbeResult(status_code=-1, body=str(exc))
    return ProbeResult(status_code=resp.status_code, body=resp.text)


def _parse_response_cost(resp: requests.Response) -> float | None:
    raw = _hdr(resp, "x-litellm-response-cost")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _streaming_outcome(resp: requests.Response, stream: bool) -> StreamingResponse:
    call_id = _hdr(resp, "x-litellm-call-id")
    response_cost = _parse_response_cost(resp)
    content_type = _hdr(resp, "content-type")
    headers = {name.lower(): value for name, value in resp.headers.items()}
    if not stream or not (200 <= resp.status_code < 300):
        return StreamingResponse(
            status_code=resp.status_code,
            call_id=call_id,
            response_cost=response_cost,
            content_type=content_type,
            headers=headers,
            body=resp.text,
        )
    lines = cast("Iterator[bytes]", resp.iter_lines())
    chunks = 0
    stream_error: str | None = None
    stream_events: list[str] = []
    for line in lines:
        if not line:
            continue
        chunks += 1
        decoded_line = line.decode(errors="replace")
        if decoded_line.startswith("data: "):
            payload = decoded_line.removeprefix("data: ")
            if payload != "[DONE]":
                stream_events.append(payload)
        if stream_error is None and (
            line.startswith(b"event: error")
            or b'"type":"error"' in line
            or b'"type": "error"' in line
            or line.startswith(b'data: {"error"')
        ):
            stream_error = line.decode(errors="replace")[:300]
    return StreamingResponse(
        status_code=resp.status_code,
        call_id=call_id,
        response_cost=response_cost,
        content_type=content_type,
        headers=headers,
        body="<streamed>",
        chunks=chunks,
        stream_events=stream_events,
        stream_error=stream_error,
    )


def send(
    url: URL,
    *,
    headers: BaseModel,
    json: BaseModel,
    params: BaseModel | None = None,
    stream: bool = False,
    timeout: float = 60.0,
) -> StreamingResponse:
    """Raw POST returning the unparsed HTTP outcome: status, full body, and the
    x-litellm-call-id header. For native/passthrough bodies and for calls judged by
    status rather than a typed JSON model (e.g. a budget block is a non-2xx). With
    ``stream=True`` the SSE body is consumed and its events counted instead."""
    try:
        resp = requests.post(
            str(url),
            headers=_headers(headers),
            params=_params(params),
            json=json.model_dump(by_alias=True, exclude_none=True),
            stream=stream,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return StreamingResponse(status_code=-1, body=str(exc))
    return _streaming_outcome(resp, stream)


def stream(
    url: URL, *, headers: BaseModel, json: BaseModel, timeout: float = 60.0
) -> StreamingResponse:
    """Streaming (SSE) call: consumes the stream counting events, and captures the
    x-litellm-call-id + content-type headers. Body is elided."""
    return send(url, headers=headers, json=json, stream=True, timeout=timeout)


def upload[R: BaseModel](
    url: URL,
    *,
    headers: BaseModel,
    form: BaseModel,
    filename: str,
    content: bytes,
    file_content_type: str = "application/jsonl",
    params: BaseModel | None = None,
    response_type: type[R],
    timeout: float = 60.0,
) -> Result[R]:
    """Multipart POST for file-bearing routes (/v1/files, /v1/audio/transcriptions).
    Form fields come from `form`, the file bytes are sent as the `file` part with
    `file_content_type`, and `params` carries any query routing (e.g. ?model=).
    requests sets the multipart Content-Type itself."""
    dumped: dict[str, object] = form.model_dump(by_alias=True, exclude_none=True)
    data = {key: str(value) for key, value in dumped.items()}
    try:
        resp = requests.post(
            str(url),
            headers=_headers(headers),
            params=_params(params),
            data=data,
            files={"file": (filename, content, file_content_type)},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return NetworkError(message=str(exc))
    return _classify(resp, response_type)


def stream_binary(
    url: URL,
    *,
    headers: BaseModel,
    json: BaseModel,
    chunk_size: int = 8192,
    timeout: float = 60.0,
) -> BinaryStream:
    """POST that consumes a binary chunked response (e.g. TTS audio) as a stream,
    counting non-empty chunks and total bytes with iter_content. A non-2xx status
    short-circuits with the counts left at zero so the caller can fail loudly."""
    try:
        resp = requests.post(
            str(url),
            headers=_headers(headers),
            json=json.model_dump(by_alias=True, exclude_none=True),
            stream=True,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return BinaryStream(status_code=-1, error_body=str(exc)[:300])
    with resp:
        content_type = _hdr(resp, "content-type")
        call_id = _hdr(resp, "x-litellm-call-id")
        transfer_encoding = _hdr(resp, "transfer-encoding")
        content_length = _hdr(resp, "content-length")
        if not (200 <= resp.status_code < 300):
            return BinaryStream(
                status_code=resp.status_code,
                content_type=content_type,
                call_id=call_id,
                transfer_encoding=transfer_encoding,
                content_length=content_length,
                error_body=resp.text[:300],
            )
        raw_chunks = cast("Iterator[bytes]", resp.iter_content(chunk_size=chunk_size))
        chunks = tuple(chunk for chunk in raw_chunks if chunk)
        return BinaryStream(
            status_code=resp.status_code,
            content_type=content_type,
            call_id=call_id,
            transfer_encoding=transfer_encoding,
            content_length=content_length,
            chunk_count=len(chunks),
            total_bytes=sum(len(chunk) for chunk in chunks),
        )


def download(
    url: URL, *, headers: BaseModel, timeout: float = 60.0
) -> StreamingResponse:
    """Raw GET for file content (/v1/files/{id}/content): provider-native bytes, no
    schema. Returns the decoded body and the x-litellm-call-id header."""
    try:
        resp = requests.get(str(url), headers=_headers(headers), timeout=timeout)
    except requests.RequestException as exc:
        return StreamingResponse(status_code=-1, body=str(exc))
    return StreamingResponse(
        status_code=resp.status_code,
        call_id=_hdr(resp, "x-litellm-call-id"),
        content_type=_hdr(resp, "content-type"),
        body=resp.text,
    )
