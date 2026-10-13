"""One HTTP call to one dataplane: credential injection, timeout, bounded retry with jitter, failure as a value.

The transport is a plain ``httpx.AsyncClient`` (the pool behind ``get_async_httpx_client``), not ``AsyncHTTPHandler``:
the handler retries writes on connection errors, replays PATCH as POST and raises ``litellm.Timeout``, none of which
a management passthrough can tolerate. Retrying is this module's job and only ever happens for reads."""

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias

import httpx
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy.control_plane.dataplanes.credentials import Caller, CredentialKind, CredentialUnavailable
from litellm.proxy.control_plane.dataplanes.registry import Dataplane

Method: TypeAlias = Literal["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]
WRITE_METHODS: Final = frozenset({"POST", "PUT", "PATCH", "DELETE"})
Sleeper: TypeAlias = Callable[[float], Awaitable[None]]
RETRYABLE_STATUSES: Final = frozenset({502, 503, 504})
DEFAULT_MAX_RESPONSE_BYTES: Final = 64 * 1024 * 1024
_STREAM_CHUNK_BYTES: Final = 1024 * 1024
_JSON: Final = TypeAdapter(object)


class HttpTransport(Protocol):
    """The slice of ``httpx.AsyncClient.request`` the client needs, so tests inject a fake. Raises only httpx errors
    and never retries on its own."""

    def stream(
        self,
        method: str,
        url: str,
        *,
        params: Sequence[tuple[str, str]] | None = None,
        json: dict[str, object] | None = None,  # mutable-ok: httpx takes dict
        headers: dict[str, str] | None = None,  # mutable-ok: httpx takes dict
        timeout: float,
    ) -> AbstractAsyncContextManager[httpx.Response]: ...


@dataclass(frozen=True, slots=True)
class DataplaneReply:
    status: int
    body: object
    credential: CredentialKind
    raw_content: bytes | None = None
    content_type: str | None = None
    content_disposition: str | None = None
    cache_control: str | None = None


@dataclass(frozen=True, slots=True)
class DataplaneUnreachable:
    detail: str
    attempts: int


@dataclass(frozen=True, slots=True)
class DataplaneResponseTooLarge:
    limit_bytes: int


CallResult: TypeAlias = DataplaneReply | DataplaneUnreachable | DataplaneResponseTooLarge | CredentialUnavailable


@dataclass(frozen=True, slots=True)
class _RawReply:
    status: int
    body: object
    raw_content: bytes | None
    content_type: str | None
    content_disposition: str | None
    cache_control: str | None


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 3
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 2.0

    def delay(self, attempt: int, jitter: float) -> float:
        capped: Final = min(self.max_delay_seconds, self.base_delay_seconds * (2.0**attempt))
        return capped * (0.5 + jitter / 2)


def _parse_body(content: bytes, response: httpx.Response) -> tuple[object, bool]:
    try:
        return _JSON.validate_json(content), True
    except ValueError:
        try:
            return content.decode(response.charset_encoding or "utf-8", errors="replace"), False
        except LookupError:
            return content, False


async def _read_chunks(
    chunks: AsyncIterator[bytes],
    limit_bytes: int,
    collected: tuple[bytes, ...],
    total_bytes: int,
) -> bytes | None:
    try:
        chunk: Final = await anext(chunks)
    except StopAsyncIteration:
        return b"".join(collected)
    next_total: Final = total_bytes + len(chunk)
    if next_total > limit_bytes:
        return None
    return await _read_chunks(chunks, limit_bytes, (*collected, chunk), next_total)


async def _read_bounded(response: httpx.Response, limit_bytes: int) -> bytes | None:
    chunks: Final = response.aiter_bytes(chunk_size=_STREAM_CHUNK_BYTES).__aiter__()
    return await _read_chunks(chunks, limit_bytes, (), 0)


async def _close_quietly(stack: AsyncExitStack) -> None:
    try:
        await stack.aclose()
    except httpx.HTTPError as error:
        verbose_proxy_logger.debug("Control-plane dataplane stream teardown failed: %s", error)


@dataclass(frozen=True, slots=True)
class DataplaneClient:
    transport: HttpTransport
    timeout_seconds: float
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    retry: RetryPolicy = RetryPolicy()
    sleep: Sleeper = asyncio.sleep
    jitter: Callable[[], float] = random.random

    async def get(
        self,
        dataplane: Dataplane,
        path: str,
        caller: Caller,
        params: Sequence[tuple[str, str]] | None = None,
    ) -> CallResult:
        return await self._call("GET", dataplane, path, caller, params=params, json=None)

    async def post(self, dataplane: Dataplane, path: str, caller: Caller, json: Mapping[str, object]) -> CallResult:
        return await self._call("POST", dataplane, path, caller, params=None, json=json)

    async def request(
        self,
        method: Method,
        dataplane: Dataplane,
        path: str,
        caller: Caller,
        params: Sequence[tuple[str, str]] | None = None,
        json: Mapping[str, object] | None = None,
    ) -> CallResult:
        """Any management verb. Reads may retry, every write goes out exactly once."""
        return await self._call(method, dataplane, path, caller, params=params, json=json)

    async def _call(
        self,
        method: Method,
        dataplane: Dataplane,
        path: str,
        caller: Caller,
        params: Sequence[tuple[str, str]] | None,
        json: Mapping[str, object] | None,
    ) -> CallResult:
        credential: Final = await dataplane.credential.headers(caller)
        if isinstance(credential, CredentialUnavailable):
            return credential
        url: Final = f"{dataplane.url}/{path.lstrip('/')}"
        attempts: Final = 1 if method in WRITE_METHODS else self.retry.attempts
        outcome: Final = await self._attempt(
            method, url, credential.headers, params, json, attempt=0, attempts=attempts
        )
        if isinstance(outcome, DataplaneResponseTooLarge):
            return outcome
        return (
            outcome
            if isinstance(outcome, DataplaneUnreachable)
            else DataplaneReply(
                outcome.status,
                outcome.body,
                credential.kind,
                outcome.raw_content,
                outcome.content_type,
                outcome.content_disposition,
                outcome.cache_control,
            )
        )

    async def _attempt(
        self,
        method: Method,
        url: str,
        headers: Mapping[str, str],
        params: Sequence[tuple[str, str]] | None,
        json: Mapping[str, object] | None,
        attempt: int,
        attempts: int,
    ) -> _RawReply | DataplaneUnreachable | DataplaneResponseTooLarge:
        outcome: Final = await self._send(method, url, headers, params, json)
        if isinstance(outcome, DataplaneResponseTooLarge):
            return outcome
        last: Final = attempt + 1 >= attempts
        if last or not _retryable(outcome):
            return outcome if isinstance(outcome, _RawReply) else DataplaneUnreachable(outcome.detail, attempt + 1)
        await self.sleep(self.retry.delay(attempt, self.jitter()))
        return await self._attempt(method, url, headers, params, json, attempt + 1, attempts)

    async def _send(
        self,
        method: Method,
        url: str,
        headers: Mapping[str, str],
        params: Sequence[tuple[str, str]] | None,
        json: Mapping[str, object] | None,
    ) -> _RawReply | DataplaneUnreachable | DataplaneResponseTooLarge:
        async with AsyncExitStack() as stack:
            try:
                response: Final = await stack.enter_async_context(
                    self.transport.stream(
                        method,
                        url,
                        params=params,
                        json=None if json is None else dict(json),
                        headers=dict(headers),
                        timeout=self.timeout_seconds,
                    )
                )
                content: Final = await _read_bounded(response, self.max_response_bytes)
            except httpx.HTTPError as error:
                verbose_proxy_logger.debug("Control-plane dataplane transport failed: %s", error)
                await _close_quietly(stack)
                return DataplaneUnreachable(_describe(error), attempts=1)
            content_type: Final = response.headers.get("content-type")
            is_json: Final = content_type is not None and "json" in content_type.lower()
            body, parsed_as_json = _parse_body(content, response) if content is not None else ("", False)
            parsed_json_response: Final = is_json and parsed_as_json
            outcome: Final = (
                DataplaneResponseTooLarge(self.max_response_bytes)
                if content is None
                else _RawReply(
                    response.status_code,
                    body,
                    None if parsed_json_response else content,
                    None if parsed_json_response else content_type,
                    response.headers.get("content-disposition"),
                    response.headers.get("cache-control"),
                )
            )
            deferred_exit: Final = stack.pop_all()
        await _close_quietly(deferred_exit)
        return outcome


def _describe(error: httpx.HTTPError) -> str:
    return type(error).__name__


def _retryable(outcome: _RawReply | DataplaneUnreachable | DataplaneResponseTooLarge) -> bool:
    return isinstance(outcome, DataplaneUnreachable) or (
        isinstance(outcome, _RawReply) and outcome.status in RETRYABLE_STATUSES
    )
