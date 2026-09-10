"""Session registry and IPC data structures for paired ext_proc and HTTP proxy requests."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TypeVar

logger = logging.getLogger("ext_proc_proxy.session")

T = TypeVar("T")


@dataclass(frozen=True)
class UpstreamRequestHeaders:
    """Request headers and metadata forwarded from HTTP proxy to Envoy."""

    method: str
    path: str
    headers: list[tuple[str, str]]
    has_body: bool
    scheme: str


@dataclass(frozen=True)
class UpstreamRequestBodyChunk:
    """Request body chunk forwarded from HTTP proxy to Envoy."""

    data: bytes
    is_last: bool


@dataclass(frozen=True)
class EnvoyResponseHeaders:
    """Response headers received from Envoy forwarded to HTTP proxy."""

    status: int
    headers: list[tuple[str, str]]
    is_empty_body: bool


@dataclass(frozen=True)
class EnvoyResponseBodyChunk:
    """Response body chunk received from Envoy forwarded to HTTP proxy."""

    data: bytes
    is_last: bool


@dataclass(frozen=True)
class SessionAbort:
    """Sentinel indicating connection cancellation or abort."""

    reason: str


RequestToEnvoyItem = UpstreamRequestHeaders | UpstreamRequestBodyChunk | SessionAbort
ResponseFromEnvoyItem = EnvoyResponseHeaders | EnvoyResponseBodyChunk | SessionAbort


DEFAULT_QUEUE_MAXSIZE: int = 32


@dataclass
class ExtProcSession:
    """State and queues for an active paired ext_proc session."""

    request_id: str
    request_to_envoy_queue: asyncio.Queue[RequestToEnvoyItem] = field(
        default_factory=lambda: asyncio.Queue(maxsize=DEFAULT_QUEUE_MAXSIZE)
    )
    response_from_envoy_queue: asyncio.Queue[ResponseFromEnvoyItem] = field(
        default_factory=lambda: asyncio.Queue(maxsize=DEFAULT_QUEUE_MAXSIZE)
    )
    is_paired: bool = False
    abort_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def is_aborted(self) -> bool:
        return self.abort_event.is_set()

    @staticmethod
    def _put_abort(
        q: asyncio.Queue[RequestToEnvoyItem] | asyncio.Queue[ResponseFromEnvoyItem],
        msg: SessionAbort,
    ) -> None:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except (asyncio.QueueEmpty, ValueError):
                pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass

    def abort(self, reason: str = "Session aborted") -> None:
        """Abort session and signal both queues."""
        if not self.abort_event.is_set():
            self.abort_event.set()
            abort_msg = SessionAbort(reason=reason)
            self._put_abort(self.request_to_envoy_queue, abort_msg)
            self._put_abort(self.response_from_envoy_queue, abort_msg)

    async def put_request(self, item: RequestToEnvoyItem) -> bool:
        """Put item into request_to_envoy_queue or return False if aborted."""
        return await self._put_with_abort(self.request_to_envoy_queue, item)

    async def put_response(self, item: ResponseFromEnvoyItem) -> bool:
        """Put item into response_from_envoy_queue or return False if aborted."""
        return await self._put_with_abort(self.response_from_envoy_queue, item)

    async def _put_with_abort(
        self,
        q: asyncio.Queue[T],
        item: T,
    ) -> bool:
        if self.abort_event.is_set():
            return False

        try:
            q.put_nowait(item)
            return True
        except asyncio.QueueFull:
            pass

        put_task = asyncio.create_task(q.put(item))
        abort_task = asyncio.create_task(self.abort_event.wait())

        try:
            done, _ = await asyncio.wait(
                [put_task, abort_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            return put_task in done
        finally:
            if not put_task.done():
                put_task.cancel()
            if not abort_task.done():
                abort_task.cancel()
            await asyncio.gather(put_task, abort_task, return_exceptions=True)


class SessionRegistry:
    """Thread-safe / asyncio registry of active ExtProcSession instances."""

    def __init__(self) -> None:
        self._sessions: dict[str, ExtProcSession] = {}

    def register(self, session: ExtProcSession) -> None:
        """Register a new ExtProcSession."""
        self._sessions[session.request_id] = session
        logger.debug("Registered session for request ID %s", session.request_id)

    def get(self, request_id: str) -> ExtProcSession | None:
        """Look up an active session by request ID."""
        return self._sessions.get(request_id)

    def unregister(self, request_id: str) -> ExtProcSession | None:
        """Unregister a session when the ext_proc stream completes."""
        session = self._sessions.pop(request_id, None)
        if session:
            logger.debug("Unregistered session for request ID %s", request_id)
        return session

    def __len__(self) -> int:
        return len(self._sessions)

    def __bool__(self) -> bool:
        """Ensure SessionRegistry instances are always truthy regardless of item count."""
        return True
