import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from math import isfinite
from typing import Final
from uuid import uuid4

import anyio
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.spend_tracking.charge_receipts import (
    ChargeReceipt,
    ChargeReceiptSettings,
    ChargeReceiptStore,
    ReceiptCommit,
    ReceiptFailure,
)
from litellm.proxy.spend_tracking.spend_event import SpendEvent, SpendEventBuildError, project_spend_event
from litellm.types.charge_receipts import CommittedCharge, FailedCharge
from litellm.types.llms.base import LiteLLMBaseModel


class BillingContext(LiteLLMBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str = Field(min_length=1, max_length=200)


@dataclass(frozen=True, slots=True)
class AuthenticatedBillingContext:
    context: BillingContext
    key_hash: str


class _Usage(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


_USAGE: Final = TypeAdapter(_Usage)


class ChargeReceiptRuntime:
    def __init__(
        self,
        settings: ChargeReceiptSettings,
        store: ChargeReceiptStore,
        before_commit: Callable[[Mapping[str, object], SpendEvent, str], Awaitable[None]] | None = None,
    ) -> None:
        self.settings: Final = settings
        self.store: Final = store
        self.available = False
        self.closing = False
        self._health_lock: Final = asyncio.Lock()
        self.before_commit: Final = before_commit
        self._captures: frozenset[ChargeCapture] = frozenset()
        self._tasks: frozenset[asyncio.Task[CommittedCharge | FailedCharge]] = frozenset()

    async def ready(self) -> bool:
        if self.closing:
            return False
        async with self._health_lock:
            if self.available:
                return True
            try:
                with anyio.fail_after(self.settings.commit_timeout_seconds):
                    self.available = await self.store.ready()
            except TimeoutError:
                self.available = False
            return self.available

    def bind(self, context: BillingContext, key_hash: str) -> "ChargeCapture":
        capture: Final = ChargeCapture(self, context, key_hash)
        self.hold(capture)
        return capture

    def hold(self, capture: "ChargeCapture") -> None:
        self._captures = self._captures.union((capture,))

    def release(self, capture: "ChargeCapture") -> None:
        self._captures = self._captures.difference((capture,))

    def track(self, task: asyncio.Task[CommittedCharge | FailedCharge]) -> None:
        self._tasks = self._tasks.union((task,))
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[CommittedCharge | FailedCharge]) -> None:
        self._tasks = self._tasks.difference((task,))
        if not task.cancelled() and task.exception() is not None:
            verbose_proxy_logger.error("Required charge capture failed before completion")

    async def close(self) -> None:
        self.closing = True
        try:
            with anyio.fail_after(self.settings.commit_timeout_seconds + 2):
                tasks: Final = tuple(self._tasks)
                if tasks:
                    await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)
                pending: Final = tuple(self._captures)
                if pending:
                    completed: Final = await asyncio.gather(*(capture.wait() for capture in pending))
                    if not all(completed):
                        verbose_proxy_logger.warning("Some requests ended without a confirmed durable charge receipt")
        except TimeoutError:
            verbose_proxy_logger.warning("Charge capture shutdown reached its drain deadline")


class ChargeCapture:
    def __init__(self, runtime: ChargeReceiptRuntime, context: BillingContext, key_hash: str) -> None:
        self.runtime: Final = runtime
        self.context: Final = context
        self.key_hash: Final = key_hash
        self.event_id: Final = str(uuid4())
        self._lock: Final = asyncio.Lock()
        self._done: Final = asyncio.Event()
        self.receipt: ChargeReceipt | None = None
        self.result: ReceiptCommit | ReceiptFailure | None = None
        self.outcome: CommittedCharge | FailedCharge | None = None

    def fork(self) -> "ChargeCapture":
        return self.runtime.bind(self.context, self.key_hash)

    async def capture_success(
        self, details: Mapping[str, object], response: object, start_time: datetime, end_time: datetime
    ) -> CommittedCharge | FailedCharge:
        event: Final = project_spend_event(details, response, start_time, end_time, store_bodies=False)
        if isinstance(event, SpendEventBuildError):
            return await self._record_failure()
        return await self.capture_event(event, details)

    async def capture_failure(
        self,
        details: Mapping[str, object],
        cost: float,
        start_time: datetime,
        end_time: datetime,
    ) -> CommittedCharge | FailedCharge:
        if cost <= 0:
            return await self._record_failure()
        event: Final = project_spend_event(details, None, start_time, end_time, store_bodies=False)
        if isinstance(event, SpendEventBuildError):
            return await self._record_failure()
        return await self.capture_event(event.model_copy(update={"response_cost": cost}), details)

    async def capture_event(
        self, event: SpendEvent, details: Mapping[str, object] | None = None
    ) -> CommittedCharge | FailedCharge:
        task: Final = asyncio.create_task(self._capture_event(event, details))
        self.runtime.track(task)
        return await asyncio.shield(task)

    async def _capture_event(
        self,
        event: SpendEvent,
        details: Mapping[str, object] | None,
    ) -> CommittedCharge | FailedCharge:
        async with self._lock:
            if isinstance(self.result, ReceiptCommit) or self.receipt is not None:
                return self._outcome()
            cost: Final = event.response_cost
            if cost is None or not isfinite(cost) or cost < 0 or not event.model:
                self.result = ReceiptFailure("invalid_charge")
                outcome: Final = self._outcome()
                self._complete()
                return outcome
            try:
                usage: Final = _USAGE.validate_python(event.combined_usage or event.standard_logging_object or {})
                receipt: Final = ChargeReceipt(
                    event_id=self.event_id,
                    request_id=self.context.request_id,
                    gateway_scope=self.runtime.settings.gateway_scope,
                    key_hash=self.key_hash,
                    amount_usd="0" if cost == 0 else format(Decimal(str(cost)), "f"),
                    model=event.model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
            except ValidationError:
                self.result = ReceiptFailure("invalid_charge")
                self._complete()
                return self._outcome()
            self.result = None
            self.outcome = None
            self._done.clear()
            self.runtime.hold(self)
            self.receipt = receipt
            if self.runtime.before_commit is not None and details is not None:
                try:
                    with anyio.fail_after(min(1, self.runtime.settings.commit_timeout_seconds)):
                        await self.runtime.before_commit(details, event, self.key_hash)
                except Exception:  # noqa: BLE001  # cache failure must not prevent durable financial capture
                    verbose_proxy_logger.warning(
                        "Spend counter preparation unavailable; attempting durable charge capture"
                    )
            try:
                with anyio.fail_after(self.runtime.settings.commit_timeout_seconds):
                    self.result = await self._append_with_retry(receipt)
            except TimeoutError:
                self.result = ReceiptFailure("database_unavailable")
            finally:
                if not isinstance(self.result, ReceiptCommit):
                    self.runtime.available = False
                self._complete()
            return self._outcome()

    async def _append_with_retry(
        self,
        receipt: ChargeReceipt,
        remaining: int = 3,
    ) -> ReceiptCommit | ReceiptFailure:
        result: Final = await self.runtime.store.append(receipt)
        if isinstance(result, ReceiptCommit) or result.reason != "database_unavailable" or remaining <= 1:
            return result
        await asyncio.sleep(0.05)
        return await self._append_with_retry(receipt, remaining - 1)

    async def _record_failure(self) -> CommittedCharge | FailedCharge:
        async with self._lock:
            if self.result is None:
                self.result = ReceiptFailure("invalid_charge")
                self._complete()
            return self._outcome()

    def _complete(self) -> None:
        self._done.set()
        self.runtime.release(self)

    async def wait(self) -> bool:
        try:
            with anyio.fail_after(self.runtime.settings.commit_timeout_seconds + 2):
                await self._done.wait()
        except TimeoutError:
            return False
        return isinstance(self.result, ReceiptCommit)

    def _outcome(self) -> CommittedCharge | FailedCharge:
        if self.outcome is None:
            self.outcome = (
                CommittedCharge(self.event_id, self.key_hash, self.receipt.amount_usd)
                if isinstance(self.result, ReceiptCommit) and self.receipt is not None
                else FailedCharge(
                    self.key_hash,
                    self.result.reason if isinstance(self.result, ReceiptFailure) else "capture_interrupted",
                )
            )
        return self.outcome
