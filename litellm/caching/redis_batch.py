"""The current request's native Redis batches: a pre-call batch for the reads it needs before answering and a
post-call batch for the writes nobody waits on, both owned by the native ``NativeRequestRedis``."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextvars import ContextVar, Token
from enum import Enum, auto
from types import TracebackType
from typing import TYPE_CHECKING, Final, Generic, Literal, Protocol, TypeAlias, TypeVar

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeRequestRedis, NativeSend

_T = TypeVar("_T")
_T_co: Final = TypeVar("_T_co", covariant=True)
POST_CALL_FLUSH_DEADLINE_SECONDS: Final = 1.0


class NativePending(Protocol[_T_co]):
    def resolve(self) -> Awaitable[_T_co]: ...


class PostCallSettlement(Protocol):
    """Post-call writes declared together: settled once every reply is back, whoever sent their batch."""

    @property
    def settled(self) -> bool: ...


class NativeBatchResult(Generic[_T]):
    """Awaitable handle for a read declared on the request's native pre-call batch; awaiting it runs the
    native flush hooks, flushes that batch, then answers from what it read."""

    __slots__ = ("_pending", "_request")

    def __init__(self, request: RequestRedisBatches, pending: NativePending[_T]) -> None:
        self._request: Final = request
        self._pending: Final = pending

    def __await__(self) -> Generator[object, None, _T]:
        return self._wait().__await__()

    async def _wait(self) -> _T:
        await self._request.flush_native_pre()
        return await self._pending.resolve()


NativeRequestFactory: TypeAlias = Callable[[float], "NativeRequestRedis"]


def _native_request(post_call_deadline: float) -> NativeRequestRedis:
    from litellm.rust_bridge._native import NativeRequestRedis

    return NativeRequestRedis(post_call_deadline)


class Batch(Enum):
    """Which of the request's batches a store op rides: none, so it goes out on its own when awaited, the pre-call
    batch of reads the request needs before answering, or the post-call batch of writes nobody waits on. Outside a
    request every op goes out on its own."""

    NONE = auto()
    PRE_CALL = auto()
    POST_CALL = auto()


RequestPhase: TypeAlias = Literal[Batch.PRE_CALL, Batch.POST_CALL]
NativeSendFactory: TypeAlias = Callable[[RequestPhase, "NativeRequestRedis"], "NativeSend"]


def _native_send(phase: RequestPhase, request: NativeRequestRedis) -> NativeSend:
    from litellm.rust_bridge._native import NativeSend

    match phase:
        case Batch.PRE_CALL:
            return NativeSend.PreCall(request)
        case Batch.POST_CALL:
            return NativeSend.PostCall(request)


_open_post_call: Final[weakref.WeakSet[RequestRedisBatches]] = weakref.WeakSet()
"""Requests whose post-call batch may still hold declared ops, so a shutdown can send them before Redis goes away."""


class RequestRedisBatches:
    """The request's native batches. The post-call batch goes out once, when the success or failure callbacks have
    all run, or ``post_call_deadline`` seconds after its first declaration when no callback phase closes it."""

    __slots__ = (
        "__weakref__",
        "_native",
        "_native_factory",
        "_native_flush_hooks",
        "_native_misses",
        "_native_send_factory",
        "_post_call_settlements",
        "_post_call_unsent",
        "post_call_deadline",
        "prefetched",
    )

    def __init__(
        self,
        post_call_deadline: float = POST_CALL_FLUSH_DEADLINE_SECONDS,
        native_factory: NativeRequestFactory = _native_request,
        native_send_factory: NativeSendFactory = _native_send,
    ) -> None:
        self._post_call_settlements: tuple[PostCallSettlement, ...] = ()
        self._post_call_unsent = False
        self.post_call_deadline: Final = post_call_deadline
        self._native_factory: Final = native_factory
        self._native_send_factory: Final = native_send_factory
        self._native: NativeRequestRedis | None = None
        self._native_flush_hooks: Final[list[Callable[[], None]]] = []  # mutable-ok: append-only registry
        self._native_misses: frozenset[str] = frozenset()
        # Reads declared early for a consumer that runs later in the request, keyed by consumer name.
        self.prefetched: Final[dict[str, object]] = {}  # mutable-ok: armed pre-admission, taken at use

    def native(self) -> NativeRequestRedis:
        """The request's native batches, built on first use so requests that touch no native store pay nothing."""
        if self._native is None:
            self._native = self._native_factory(self.post_call_deadline)
        return self._native

    def native_post_call(self) -> NativeRequestRedis:
        """``native`` for a post-call declaration, so a shutdown drain also flushes it."""
        _open_post_call.add(self)
        self._post_call_unsent = True
        return self.native()

    def native_send(self, phase: RequestPhase) -> NativeSend:
        """The ``NativeSend`` for ``phase``; a post-call one registers the request with the shutdown drain."""
        match phase:
            case Batch.PRE_CALL:
                return self._native_send_factory(phase, self.native())
            case Batch.POST_CALL:
                return self._native_send_factory(phase, self.native_post_call())

    def native_result(self, pending: NativePending[_T]) -> NativeBatchResult[_T]:
        return NativeBatchResult(self, pending)

    def add_pre_call_flush_hook(self, hook: Callable[[], None]) -> None:
        """Called before every pre-call flush so late readers can declare into the same trip. The hook runs with this
        request active, so a ``Batch.PRE_CALL`` op it declares rides this request whoever triggered the flush."""
        self._native_flush_hooks.append(hook)

    def note_native_misses(self, keys: Sequence[str]) -> None:
        """Keys a native pre-call read found absent, or a native DEL removed, so a later GET can skip Redis."""
        self._native_misses = self._native_misses | frozenset(keys)

    def forget_native_miss(self, key: str) -> None:
        self._native_misses = self._native_misses - frozenset((key,))

    def native_read_as_missing(self, key: str) -> bool:
        return key in self._native_misses

    async def flush_native_pre(self) -> None:
        token: Final = _active_request_batches.set(self)
        try:
            for hook in self._native_flush_hooks:
                hook()
        finally:
            _active_request_batches.reset(token)
        await self.flush_pre_call()

    async def flush_pre_call(self) -> None:
        """Send the pre-call ops declared so far, such as write-backs nobody awaits before the request scope closes.
        The pre-call flush hooks are skipped: they declare reads for a reader about to await, and none is waiting."""
        if self._native is not None:
            await self._native.flush_pre()

    async def flush_post_call(self) -> None:
        """Send the post-call writes. The request stays registered with the shutdown drain until they are back, so a
        drain that overlaps this flush waits for it, and an op declared while it runs keeps the request registered."""
        self._post_call_unsent = False
        if self._native is not None:
            await self._native.flush_post()
        if not self._post_call_unsent:
            _open_post_call.discard(self)

    def track_post_call_settlement(self, settlement: PostCallSettlement) -> None:
        """Post-call writes a later reader in this request must know have landed, such as its committed spend."""
        self._post_call_settlements = (*self._post_call_settlements, settlement)

    @property
    def post_call_settlements(self) -> tuple[PostCallSettlement, ...]:
        return self._post_call_settlements


_active_request_batches: Final[ContextVar[RequestRedisBatches | None]] = ContextVar(
    "request_redis_batches", default=None
)


def active_request_redis_batches() -> RequestRedisBatches | None:
    return _active_request_batches.get()


async def flush_post_call_redis_batches() -> None:
    """Called where the success and failure callbacks of a request have all run."""
    batches: Final = _active_request_batches.get()
    if batches is not None:
        await batches.flush_post_call()


async def drain_post_call_redis_batches() -> None:
    """Sends every post-call batch still waiting on its callbacks or deadline; for the shutdown path."""
    await asyncio.gather(*(batches.flush_post_call() for batches in tuple(_open_post_call)))


class request_redis_batch_scope:
    """Redis ops declared inside share the request's native batches; nested scopes join the outer one."""

    __slots__ = ("_native_factory", "_native_send_factory", "_post_call_deadline", "_token")

    def __init__(
        self,
        post_call_deadline: float = POST_CALL_FLUSH_DEADLINE_SECONDS,
        native_factory: NativeRequestFactory = _native_request,
        native_send_factory: NativeSendFactory = _native_send,
    ) -> None:
        self._token: Token[RequestRedisBatches | None] | None = None
        self._post_call_deadline: Final = post_call_deadline
        self._native_factory: Final = native_factory
        self._native_send_factory: Final = native_send_factory

    def __enter__(self) -> RequestRedisBatches:
        outer: Final = _active_request_batches.get()
        if outer is not None:
            return outer
        batches: Final = RequestRedisBatches(
            post_call_deadline=self._post_call_deadline,
            native_factory=self._native_factory,
            native_send_factory=self._native_send_factory,
        )
        self._token = _active_request_batches.set(batches)
        return batches

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._token is not None:
            _active_request_batches.reset(self._token)
