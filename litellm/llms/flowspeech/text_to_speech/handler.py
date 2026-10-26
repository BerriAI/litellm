import asyncio
import io
import json
import math
import struct
import time
import wave
from collections.abc import Awaitable, Callable, Generator, Mapping
from datetime import timezone
from email.utils import parsedate_to_datetime
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx

from litellm.types.llms.openai import HttpxBinaryResponseContent

from .transformation import (
    FlowSpeechErrorResponse,
    FlowSpeechException,
    FlowSpeechJob,
    validate_url,
    validate_uuid,
)

Flow: TypeAlias = Generator[
    httpx.Request | float, httpx.Response | httpx.TransportError | None, HttpxBinaryResponseContent
]
Exchange: TypeAlias = Generator[httpx.Request | float, httpx.Response | httpx.TransportError | None, httpx.Response]


def validate_wav_structure(content: bytes) -> None:
    if (
        len(content) < 12
        or content[:4] != b"RIFF"
        or content[8:12] != b"WAVE"
        or int.from_bytes(content[4:8], "little") != len(content) - 8
    ):
        raise ValueError("Invalid RIFF container length")
    stream: Final = io.BytesIO(content)
    stream.seek(12)
    for header in iter(lambda: stream.read(8), b""):
        if len(header) != 8:
            raise ValueError("Truncated WAV chunk header")
        size = int.from_bytes(header[4:], "little")
        end = stream.tell() + size + size % 2
        if end > len(content):
            raise ValueError("Truncated WAV chunk")
        if header[:4] == b"fmt " and stream.read(min(size, 16))[2:] != struct.pack("<HIIHH", 1, 24000, 48000, 2, 16):
            raise ValueError("Invalid PCM format fields")
        if header[:4] == b"data" and size % 2:
            raise ValueError("Unaligned PCM data")
        stream.seek(end)


class FlowSpeechTTS:
    MAX_AUDIO_BYTES = 100 * 1024 * 1024
    MAX_JSON_BYTES = 1024 * 1024
    POLL_INTERVAL = 5.0

    def __init__(
        self,
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, object],
        timeout: float | httpx.Timeout,
        max_retries: object = 2,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_completed: Callable[[FlowSpeechJob], None] | None = None,
    ) -> None:
        seconds: Final = timeout.read if isinstance(timeout, httpx.Timeout) else timeout
        if isinstance(seconds, bool) or seconds is None or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("FlowSpeech timeout must be finite and positive (httpx.Timeout uses read as total)")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 10:
            raise ValueError("FlowSpeech max_retries must be an integer between 0 and 10")
        self.url = str(validate_url(url))
        self.headers = httpx.Headers(headers)
        self.request_id = validate_uuid(self.headers["Idempotency-Key"])
        self.job_id: str | None = None
        serialized: Final = json.dumps(
            dict(body),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.body = serialized.encode("utf-8")
        if len(self.body) > self.MAX_JSON_BYTES:
            raise ValueError("FlowSpeech request exceeds 1 MiB")
        self.timeout = timeout
        self.deadline = clock() + seconds
        self.clock = clock
        self.wall_clock = wall_clock
        self.sleep = sleep
        self.async_sleep = async_sleep
        self.max_retries = max_retries
        self.on_completed = on_completed

    def error(self, status: int, message: str, headers: httpx.Headers | None = None) -> FlowSpeechException:
        return FlowSpeechException(
            status,
            f"{message} (request_id={self.request_id}, job_id={self.job_id or 'unknown'})",
            headers,
        )

    def remaining(self) -> float:
        remaining: Final = self.deadline - self.clock()
        if remaining <= 0:
            raise self.error(408, "FlowSpeech total deadline exceeded; recover using the same request_id")
        return remaining

    def delay(self, response: httpx.Response | None = None) -> float:
        value: Final = (
            response.headers["retry-after"] if response is not None and "retry-after" in response.headers else None
        )
        if not value:
            return self.POLL_INTERVAL
        try:
            seconds: Final = float(value)
        except ValueError:
            try:
                date: Final = parsedate_to_datetime(value)
                return max(
                    self.POLL_INTERVAL, date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp() - self.wall_clock()
                )
            except (ValueError, TypeError, OverflowError):
                return self.POLL_INTERVAL
        return max(self.POLL_INTERVAL, seconds) if math.isfinite(seconds) else self.POLL_INTERVAL

    def wait_budget(self, delay: float) -> None:
        if delay >= self.remaining():
            raise self.error(408, "FlowSpeech total deadline cannot accommodate Retry-After/poll delay")

    def request(self, method: str, url: str, download: bool) -> httpx.Request:
        remaining: Final = self.remaining()
        timeouts: Final[Mapping[str, float | None]] = (
            self.timeout.as_dict() if isinstance(self.timeout, httpx.Timeout) else MappingProxyType({})
        )
        return httpx.Request(
            method,
            url,
            headers=MappingProxyType({"Accept-Encoding": "identity"})
            if download
            else MappingProxyType({**self.headers, "Accept-Encoding": "identity"}),
            content=self.body if method == "POST" else None,
            extensions=MappingProxyType(
                {
                    "timeout": MappingProxyType(
                        {
                            phase: min(limit, remaining) if (limit := timeouts.get(phase)) is not None else remaining
                            for phase in ("connect", "read", "write", "pool")
                        }
                    ),
                    "flowspeech_download": download,
                }
            ),
        )

    def exchange(self, method: str, url: str, *, download: bool = False) -> Exchange:
        for attempt in range(self.max_retries + 1):
            response = yield self.request(method, url, download)
            self.remaining()
            if isinstance(response, httpx.TransportError):
                if attempt == self.max_retries:
                    raise self.error(
                        502, "FlowSpeech download transport failed" if download else "FlowSpeech API transport failed"
                    )
                yield self.delay()
                continue
            if not isinstance(response, httpx.Response):
                raise self.error(502, "FlowSpeech transport did not return a response")
            if response.status_code in ((200,) if download or method == "GET" else (200, 202)):
                return response
            if response.is_redirect:
                raise self.error(502, "FlowSpeech unexpected redirect rejected", response.headers)
            try:
                payload = FlowSpeechErrorResponse.model_validate_json(response.content)
            except ValueError:
                payload = None
            retryable = response.status_code == 429 or (
                response.status_code >= 500 and (download or (payload is not None and payload.error.retryable))
            )
            if retryable and attempt < self.max_retries:
                yield self.delay(response)
                continue
            detail = (
                f"{payload.error.code}: {payload.error.message}; traceId={payload.traceId or 'unknown'}"
                if payload
                else f"unexpected HTTP {response.status_code}"
            )
            raise self.error(
                response.status_code if response.status_code >= 400 else 502,
                f"FlowSpeech {'download' if download else 'API'}: {detail}",
                response.headers,
            )
        raise self.error(502, "FlowSpeech retries exhausted")

    def flow(self) -> Flow:
        response = yield from self.exchange("POST", self.url)  # rebind-ok: advances through job status responses
        while True:
            try:
                job = FlowSpeechJob.model_validate_json(response.content)
                job_id = job.id
            except ValueError as exc:
                raise self.error(502, "FlowSpeech returned an invalid job, ID, status or result") from exc
            if job.requestId != self.request_id:
                raise self.error(502, "FlowSpeech returned a different request ID")
            if self.job_id is not None and self.job_id != job_id:
                raise self.error(502, "FlowSpeech returned a different job ID")
            self.job_id = job_id
            if job.status in ("failed", "cancelled"):
                detail = f"{job.error.code}: {job.error.message}" if job.error else job.status
                raise self.error(422, f"FlowSpeech job {job.status}: {detail}")
            if job.status == "success":
                if job.resultDeleted:
                    raise self.error(410, "FlowSpeech recording was deleted (resultDeleted=true)")
                if job.result is None:
                    raise self.error(502, "FlowSpeech successful job has no result")
                try:
                    audio_url = str(validate_url(job.result.url))
                except ValueError as exc:
                    raise self.error(502, "FlowSpeech returned an unsafe audio URL") from exc
                audio = yield from self.exchange("GET", audio_url, download=True)
                return self.completed_response(job, audio)
            yield self.delay(response)
            response = yield from self.exchange("GET", f"{self.url}/{job_id}")

    def completed_response(self, job: FlowSpeechJob, response: httpx.Response) -> HttpxBinaryResponseContent:
        result: Final = self.audio_response(response)
        if self.on_completed is not None:
            self.on_completed(job)
        return result

    def audio_response(self, response: httpx.Response) -> HttpxBinaryResponseContent:
        try:
            validate_wav_structure(response.content)
            with wave.open(io.BytesIO(response.content), "rb") as audio:
                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (
                    24000,
                    1,
                    2,
                    "NONE",
                ):
                    raise ValueError("Unexpected WAV encoding")
                if audio.getnframes() == 0 or len(audio.readframes(audio.getnframes() + 1)) != audio.getnframes() * 2:
                    raise ValueError("Empty or truncated WAV")
        except (wave.Error, EOFError, ValueError) as exc:
            raise self.error(502, "FlowSpeech download is not complete 24 kHz mono 16-bit WAV audio") from exc
        return HttpxBinaryResponseContent(
            httpx.Response(
                200,
                content=response.content,
                headers=MappingProxyType(
                    {
                        "content-type": "audio/wav",
                        "x-flowspeech-request-id": self.request_id,
                        "x-flowspeech-job-id": self.job_id or "",
                    }
                ),
                request=response.request,
            )
        )

    def limit(self, request: httpx.Request, response: httpx.Response) -> int:
        maximum: Final = self.MAX_AUDIO_BYTES if request.extensions["flowspeech_download"] else self.MAX_JSON_BYTES
        if "content-encoding" in response.headers and response.headers["content-encoding"].lower() != "identity":
            raise self.error(502, "FlowSpeech encoded response rejected")
        try:
            length: Final = int(response.headers["content-length"]) if "content-length" in response.headers else 0
        except ValueError as exc:
            raise self.error(502, "FlowSpeech invalid Content-Length") from exc
        if not 0 <= length <= maximum:
            raise self.error(502, "FlowSpeech response exceeds size limit")
        return maximum

    def read(self, client: httpx.Client, request: httpx.Request) -> httpx.Response:
        response: Final = client.send(request, stream=True, auth=None, follow_redirects=False)
        try:
            maximum: Final = self.limit(request, response)
            with io.BytesIO() as content:
                for chunk in response.iter_bytes():
                    self.remaining()
                    if content.tell() + len(chunk) > maximum:
                        raise self.error(502, "FlowSpeech response exceeds size limit")
                    content.write(chunk)
                self.remaining()
                return httpx.Response(
                    response.status_code, headers=response.headers, content=content.getvalue(), request=request
                )
        finally:
            response.close()

    async def aread(self, client: httpx.AsyncClient, request: httpx.Request) -> httpx.Response:
        response: Final = await client.send(request, stream=True, auth=None, follow_redirects=False)
        try:
            maximum: Final = self.limit(request, response)
            with io.BytesIO() as content:
                async for chunk in response.aiter_bytes():
                    self.remaining()
                    if content.tell() + len(chunk) > maximum:
                        raise self.error(502, "FlowSpeech response exceeds size limit")
                    content.write(chunk)
                self.remaining()
                return httpx.Response(
                    response.status_code, headers=response.headers, content=content.getvalue(), request=request
                )
        finally:
            await response.aclose()

    def run(self, client: httpx.Client | None = None) -> HttpxBinaryResponseContent:
        if client is None:
            with httpx.Client(trust_env=False) as owned_client:
                return self._run(owned_client)
        return self._run(client)

    def _run(self, client: httpx.Client) -> HttpxBinaryResponseContent:
        flow: Final = self.flow()
        action = next(flow)  # rebind-ok: workflow driver advances actions
        try:
            while True:
                result: httpx.Response | httpx.TransportError | None = None
                if not isinstance(action, httpx.Request):
                    self.wait_budget(action)
                    self.sleep(action)
                else:
                    try:
                        result = self.read(client, action)
                    except httpx.TransportError as exc:
                        result = exc
                action = flow.send(result)
        except StopIteration as done:
            value: Final[object] = done.value  # pyright: ignore[reportAny]  # StopIteration erases the typed generator return; checked below
            if not isinstance(value, HttpxBinaryResponseContent):
                raise self.error(502, "FlowSpeech workflow did not return audio")
            return value  # pyright: ignore[reportUnknownVariableType]  # StopIteration erases the response generic
        finally:
            flow.close()

    async def arun(self, client: httpx.AsyncClient | None = None) -> HttpxBinaryResponseContent:
        if client is None:
            async with httpx.AsyncClient(trust_env=False) as owned_client:
                return await self._arun_with_deadline(owned_client)
        return await self._arun_with_deadline(client)

    async def _arun_with_deadline(self, client: httpx.AsyncClient) -> HttpxBinaryResponseContent:
        try:
            return await asyncio.wait_for(self._arun(client), timeout=self.remaining())
        except asyncio.TimeoutError as exc:
            raise self.error(408, "FlowSpeech total deadline exceeded; recover using the same request_id") from exc

    async def _arun(self, client: httpx.AsyncClient) -> HttpxBinaryResponseContent:
        flow: Final = self.flow()
        action = next(flow)  # rebind-ok: workflow driver advances actions
        try:
            while True:
                result: httpx.Response | httpx.TransportError | None = None
                if not isinstance(action, httpx.Request):
                    self.wait_budget(action)
                    await self.async_sleep(action)
                else:
                    try:
                        result = await self.aread(client, action)
                    except httpx.TransportError as exc:
                        result = exc
                action = flow.send(result)
        except StopIteration as done:
            value: Final[object] = done.value  # pyright: ignore[reportAny]  # StopIteration erases the typed generator return; checked below
            if not isinstance(value, HttpxBinaryResponseContent):
                raise self.error(502, "FlowSpeech workflow did not return audio")
            return value  # pyright: ignore[reportUnknownVariableType]  # StopIteration erases the response generic
        finally:
            flow.close()
