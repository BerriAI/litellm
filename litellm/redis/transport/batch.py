import asyncio
import time
from collections.abc import Awaitable, Callable, Generator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Generic, Protocol, TypeVar

from litellm._internal_context import (
    REDIS_FAMILIES_METADATA_KEY,
    current_service_target,
    service_caller,
    service_target,
)
from litellm._logging import verbose_logger
from litellm.caching.redis_cache import (
    RedisCircuitBreaker,
    _get_call_stack_info,  # pyright: ignore[reportPrivateUsage]  # same caller chain every RedisCache method reports
    _run_under_circuit_breaker,  # pyright: ignore[reportPrivateUsage]  # same health signal as every RedisCache method
)
from litellm.redis.transport.client import PipelineClient, RedisPipeline
from litellm.redis.transport.contract import Admit

if TYPE_CHECKING:
    from litellm._service_logger import ServiceLogging

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
SettledHook = Callable[[asyncio.Future[_T]], Awaitable[None] | None]
MIXED_PIPELINE_TARGET: Final = "mixed"


class OpSpec(Protocol[_T_co]):
    """How one operation rides a pipeline: the commands it queues, how its replies become its result, what a
    failed pipeline or command makes of it (raise to fail the op, return to settle it), and how it runs alone."""

    def enqueue(self, pipe: RedisPipeline) -> int: ...

    def resolve(self, replies: Sequence[object]) -> _T_co: ...

    def recover(self, error: Exception) -> _T_co: ...

    def run_alone(self) -> Awaitable[_T_co]: ...


@dataclass(frozen=True, slots=True)
class PipelineBackend:
    client: Callable[[], PipelineClient]
    breaker: RedisCircuitBreaker
    admit: Admit
    service_logging: "ServiceLogging"
    runs_alone: bool


def _mark_retrieved(future: asyncio.Future[object]) -> None:
    if not future.cancelled():
        future.exception()


def _is_missing_script(failure: Exception) -> bool:
    from litellm.redis.transport.redis_py import is_missing_script

    return is_missing_script(failure)


class _Op(Generic[_T]):
    __slots__ = ("caller", "future", "retry_missing_script", "settled_hooks", "spec", "target")

    def __init__(self, spec: OpSpec[_T], *, retry_missing_script: bool) -> None:
        self.spec: Final = spec
        self.retry_missing_script: Final = retry_missing_script
        self.future: Final[asyncio.Future[_T]] = asyncio.get_running_loop().create_future()
        self.future.add_done_callback(_mark_retrieved)
        self.settled_hooks: Final[list[SettledHook[_T]]] = []  # mutable-ok: append-only registry
        self.target: Final = current_service_target()
        self.caller: Final = _get_call_stack_info()

    async def run_settled_hooks(self) -> None:
        for hook in self.settled_hooks:
            await self._run_settled_hook(hook)

    async def _run_settled_hook(self, hook: SettledHook[_T]) -> None:
        try:
            follow_up: Final = hook(self.future)
            if follow_up is not None:
                await follow_up
        except Exception as e:  # noqa: BLE001  # one owner's follow-up must not stop the others
            verbose_logger.warning("redis batch settled hook failed: %s", e)

    def settle(self, replies: Sequence[object]) -> Awaitable[None] | None:
        failure: Final = next((reply for reply in replies if isinstance(reply, Exception)), None)
        if failure is None:
            try:
                self.future.set_result(self.spec.resolve(replies))
            except Exception as e:  # noqa: BLE001  # a reply this op cannot decode fails this op alone
                self.future.set_exception(e)
            return None
        if self.retry_missing_script and _is_missing_script(failure):
            return self.settle_alone()
        self.recover(failure)
        return None

    def recover(self, error: Exception) -> None:
        try:
            self.future.set_result(self.spec.recover(error))
        except Exception as e:  # noqa: BLE001  # the declaring caller owns the failure of its own operation
            self.future.set_exception(e)

    async def settle_alone(self) -> None:
        try:
            with service_target(self.target), service_caller(self.caller):
                self.future.set_result(await self.spec.run_alone())
        except Exception as e:  # noqa: BLE001  # the declaring caller owns the failure of its own operation
            self.future.set_exception(e)


class BatchResult(Generic[_T]):
    """Awaitable handle for one declared operation; awaiting it flushes the batch it belongs to."""

    __slots__ = ("_batch", "_op")

    def __init__(self, batch: "PipelineBatch", op: _Op[_T]) -> None:
        self._batch: Final = batch
        self._op: Final = op

    def __await__(self) -> Generator[object, None, _T]:
        return self._wait().__await__()

    async def _wait(self) -> _T:
        if not self._op.future.done():
            await self._batch.flush()
        return self._op.future.result()

    @property
    def done(self) -> bool:
        return self._op.future.done()

    def on_settled(self, hook: SettledHook[_T]) -> None:
        """For an owner that does not await: runs inside the flush once this operation has its result or
        failure (or was cancelled with the pipeline), so the flush completes with the follow-up done."""
        self._op.settled_hooks.append(hook)


@dataclass(slots=True)
class PipelineBatch:
    """Operations declared here go out in one pipeline the next time any of them is awaited or ``flush`` runs.
    A backend that cannot pipeline them (a cluster client) runs each on its own, unless the batch is a
    transaction, which always goes out as one MULTI/EXEC."""

    backend: PipelineBackend
    name: str = "redis_batch"
    transaction: bool = False
    raise_on_error: bool = False
    _pending: list[_Op[object]] = field(default_factory=list)  # mutable-ok: drained by flush
    _flush_hooks: list[Callable[[], None]] = field(default_factory=list)  # mutable-ok: append-only registry
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _misses: set[str] = field(default_factory=set)  # mutable-ok: keys a read of this batch found absent
    flushes: int = 0

    def declare(self, spec: OpSpec[_T], *, retry_missing_script: bool = True) -> BatchResult[_T]:
        op: Final = _Op(spec, retry_missing_script=retry_missing_script)
        self._pending.append(op)  # pyright: ignore[reportArgumentType]  # heterogeneous ops share the flush loop
        return BatchResult(self, op)

    def note_missing(self, keys: Iterable[str]) -> None:
        self._misses.update(keys)

    def note_present(self, key: str) -> None:
        self._misses.discard(key)

    def read_as_missing(self, key: str) -> bool:
        """True when a read on this batch already found no value under ``key`` and nothing has set it since,
        so a per-key GET later in the same request can be answered without another round trip."""
        return key in self._misses

    def add_flush_hook(self, hook: Callable[[], None]) -> None:
        """Called at the start of every flush so lazily bound readers can declare their keys into the same trip."""
        self._flush_hooks.append(hook)

    @property
    def pending(self) -> int:
        return len(self._pending)

    async def flush(self) -> None:
        async with self._lock:
            for hook in self._flush_hooks:
                hook()
            ops: Final = tuple(self._pending)
            self._pending.clear()
            if not ops:
                return
            self.flushes += 1
            try:
                if self.backend.runs_alone and not self.transaction:
                    await asyncio.gather(*(op.settle_alone() for op in ops))
                else:
                    await self._flush_pipeline(ops)
            finally:
                for op in ops:
                    if not op.future.done():
                        op.future.cancel()
                await asyncio.gather(*(op.run_settled_hooks() for op in ops))

    async def _flush_pipeline(self, ops: Sequence[_Op[object]]) -> None:
        from litellm.redis.transport.events import report_pipeline

        start_time: Final = time.time()
        target, metadata = _pipeline_service_event(ops)
        widths: list[int] = []  # mutable-ok: filled while enqueuing

        async def run() -> list[object]:
            async with self.backend.client().pipeline(transaction=self.transaction) as pipe:
                widths.extend(op.spec.enqueue(pipe) for op in ops)
                return await pipe.execute(raise_on_error=self.raise_on_error)

        try:
            replies: Final = await _run_under_circuit_breaker(
                self.backend.breaker, self.name, lambda: self.backend.admit(run)
            )
        except Exception as e:  # noqa: BLE001  # each declaring caller applies its own Redis fallback
            report_pipeline(
                self.backend.service_logging,
                self.name,
                op_count=len(ops),
                target=target,
                metadata=metadata,
                start_time=start_time,
                error=e,
            )
            for op in ops:
                op.recover(e)
            return
        report_pipeline(
            self.backend.service_logging,
            self.name,
            op_count=len(ops),
            target=target,
            metadata=metadata,
            start_time=start_time,
            error=None,
        )
        retries: list[Awaitable[None]] = []  # mutable-ok: collected while slicing replies
        offset = 0
        for op, width in zip(ops, widths):
            retry = op.settle(replies[offset : offset + width])
            offset += width
            if retry is not None:
                retries.append(retry)
        if retries:
            await asyncio.gather(*retries)


def _pipeline_service_event(ops: Sequence[_Op[object]]) -> tuple[str | None, dict[str, int | str]]:
    """The target and metadata of one pipeline flush: the one key family every op was declared under, or
    ``"mixed"`` plus the sorted families when owners of several families share the trip."""
    families: Final = sorted({op.target for op in ops if op.target is not None})
    if len(families) > 1:
        return MIXED_PIPELINE_TARGET, {"op_count": len(ops), REDIS_FAMILIES_METADATA_KEY: ",".join(families)}
    return next(iter(families), None), {"op_count": len(ops)}
