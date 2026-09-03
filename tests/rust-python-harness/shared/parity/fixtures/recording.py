from __future__ import annotations

import queue
from collections.abc import Callable, Generator, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Final, TypeVar, cast

import httpx
from vcr.filters import remove_query_parameters
from vcr.request import Request

from ..http import (
    PARITY_PROVIDER_HOST,
    dropped_request_headers,
    dropped_response_headers,
    is_streaming_response,
    local_response_header,
    normalized_response_header,
)
from ..local_server import LocalHttpHandler, LocalHttpServer, serve_in_thread
from ..recorded_http import (
    HttpHeader,
    RecordedHttpResponse,
    RecordedHttpStreamResponse,
    RecordedResponse,
    RecordedStreamChunk,
)

_SECRET_HEADERS: Final = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "x-api-key",
        "api-key",
        "anthropic-api-key",
        "openai-api-key",
        "azure-api-key",
        "x-goog-api-key",
        "ocp-apim-subscription-key",
        "x-amz-security-token",
    }
)

InputT = TypeVar("InputT")


@dataclass(frozen=True, slots=True)
class UpstreamEndpoint:
    base_url: str


@dataclass(frozen=True, slots=True)
class RecordedInteraction:
    request: Request
    response: RecordedResponse


def _end_to_end_headers(headers: httpx.Headers) -> tuple[HttpHeader, ...]:
    decoded: Final = tuple((name.decode("ascii"), value.decode("latin-1")) for name, value in headers.raw)
    excluded: Final = dropped_response_headers(decoded)
    return tuple(
        HttpHeader(name=name, value=normalized_response_header(name, value))
        for name, value in decoded
        if name.lower() not in excluded
    )


class _RecordingProvider(LocalHttpServer):
    def __init__(self, spec: UpstreamEndpoint) -> None:
        super().__init__(("127.0.0.1", 0), _RecordingHandler)
        self.spec: Final = spec
        self.interactions: queue.Queue[RecordedInteraction] = queue.Queue()

    def take_interactions(self) -> tuple[RecordedInteraction, ...]:
        try:
            first: Final = self.interactions.get(timeout=5)
        except queue.Empty as error:
            raise RuntimeError("successful SDK call did not produce a recorded response") from error
        remaining: Final = tuple(self.interactions.get_nowait() for _ in range(self.interactions.qsize()))
        return (first, *remaining)


class _RecordingHandler(LocalHttpHandler):
    def do_POST(self) -> None:
        self._forward()

    def do_GET(self) -> None:
        self._forward()

    def do_PUT(self) -> None:
        self._forward()

    def do_PATCH(self) -> None:
        self._forward()

    def do_DELETE(self) -> None:
        self._forward()

    def _forward(self) -> None:
        provider: Final = self.server
        assert isinstance(provider, _RecordingProvider)
        length: Final = int(self.headers.get("content-length") or "0")
        request_body: Final = self.rfile.read(length) if length else b""
        raw_headers: Final = tuple(self.headers.raw_items())
        excluded: Final = dropped_request_headers(raw_headers)
        forwarded_headers: Final = tuple((name, value) for name, value in raw_headers if name.lower() not in excluded)
        upstream_url: Final = f"{provider.spec.base_url.rstrip('/')}{self.path}"

        try:
            with httpx.stream(
                self.command,
                upstream_url,
                headers=forwarded_headers,
                content=request_body,
                timeout=120,
            ) as upstream:
                headers: Final = _end_to_end_headers(upstream.headers)
                recorded_response: Final = self._record_upstream_response(upstream, headers)
        except httpx.HTTPError as error:
            self._send_response(502, (), str(error).encode("utf-8"))
            return

        recorded_request: Final = remove_query_parameters(
            Request(
                self.command,
                f"http://{PARITY_PROVIDER_HOST}{self.path}",
                request_body,
                {name: value for name, value in forwarded_headers if name.lower() not in _SECRET_HEADERS},
            ),
            ("api_key", "api-key", "key", "access_token", "subscription-key"),
        )
        provider.interactions.put(RecordedInteraction(recorded_request, recorded_response))
        if isinstance(recorded_response, RecordedHttpResponse):
            self._send_response(
                recorded_response.status_code, recorded_response.headers, recorded_response.body_bytes()
            )

    def _record_upstream_response(
        self,
        upstream: httpx.Response,
        headers: tuple[HttpHeader, ...],
    ) -> RecordedResponse:
        content_type: Final = cast(str, upstream.headers.get("content-type", ""))
        if is_streaming_response(content_type):
            return self._record_stream(upstream, headers)
        response_body: Final = b"".join(upstream.iter_bytes())
        return RecordedHttpResponse.from_bytes(
            status_code=upstream.status_code,
            headers=headers,
            body=response_body,
        )

    def _record_stream(
        self,
        upstream: httpx.Response,
        headers: tuple[HttpHeader, ...],
    ) -> RecordedHttpStreamResponse:
        self.send_response_only(upstream.status_code)
        provider: Final = self.server
        assert isinstance(provider, _RecordingProvider)
        for header in headers:
            self.send_header(header.name, local_response_header(header.name, header.value, provider.url))
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        chunks: Final = tuple(self._relay_chunks(upstream.iter_bytes()))
        try:
            self.finish_chunked()
        except (BrokenPipeError, ConnectionResetError):
            pass
        return RecordedHttpStreamResponse(
            kind="http_stream",
            status_code=upstream.status_code,
            headers=headers,
            chunks=chunks,
        )

    def _relay_chunks(self, chunks: Iterable[bytes]) -> Generator[RecordedStreamChunk, None, None]:
        for chunk in chunks:
            self.write_chunk(chunk)
            yield RecordedStreamChunk.from_bytes(chunk)

    def _send_response(self, status_code: int, headers: tuple[HttpHeader, ...], body: bytes) -> None:
        self.send_response_only(status_code)
        provider: Final = self.server
        assert isinstance(provider, _RecordingProvider)
        for header in headers:
            self.send_header(header.name, local_response_header(header.name, header.value, provider.url))
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def _recording_provider(spec: UpstreamEndpoint) -> AbstractContextManager[_RecordingProvider]:
    return serve_in_thread(_RecordingProvider(spec))


def _invoke_and_take_interactions(
    recorder: _RecordingProvider,
    case_input: InputT,
    sdk_call: Callable[[str, InputT], object],
) -> tuple[RecordedInteraction, ...]:
    try:
        sdk_call(recorder.url, case_input)
    except Exception as invocation_error:
        try:
            return recorder.take_interactions()
        except RuntimeError:
            raise invocation_error
    return recorder.take_interactions()


def record_upstream_interactions(
    spec: UpstreamEndpoint,
    case_input: InputT,
    sdk_call: Callable[[str, InputT], object],
) -> tuple[RecordedInteraction, ...]:
    with _recording_provider(spec) as recorder:
        return _invoke_and_take_interactions(recorder, case_input, sdk_call)


def record_upstream_responses(
    spec: UpstreamEndpoint,
    case_input: InputT,
    sdk_call: Callable[[str, InputT], object],
) -> tuple[RecordedResponse, ...]:
    return tuple(item.response for item in record_upstream_interactions(spec, case_input, sdk_call))
