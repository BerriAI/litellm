import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from decimal import Decimal
from typing import Final
from uuid import uuid4

import anyio
from pydantic import ConfigDict, Field, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceipt, ChargeReceiptSettings, ChargeReceiptStore
from litellm.proxy.spend_tracking.spend_event import SpendEvent, SpendEventBuildError, project_spend_event
from litellm.types.charge_receipts import CommittedCharge, FailedCharge
from litellm.types.llms.base import LiteLLMBaseModel


class BillingContext(LiteLLMBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request_id: str = Field(min_length=1, max_length=200)
    key_hash: str


class _Usage(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ChargeReceiptRuntime:
    def __init__(
        self,
        settings: ChargeReceiptSettings,
        store: ChargeReceiptStore,
        finalize: Callable[[Mapping[str, object], SpendEvent, "ChargeCapture"], Awaitable[None]] | None = None,
    ) -> None:
        self.settings: Final = settings
        self.store: Final = store
        self.finalize: Final = finalize
        self.available = False
        self.closing = False
        self.captures: frozenset[ChargeCapture] = frozenset()
        self._health_lock: Final = asyncio.Lock()

    async def ready(self) -> bool:
        if self.closing:
            return False
        async with self._health_lock:
            if not self.available:
                with anyio.move_on_after(self.settings.commit_timeout_seconds):
                    self.available = await self.store.ready()
            return self.available

    def bind(self, context: BillingContext) -> "ChargeCapture":
        capture: Final = ChargeCapture(self, context)
        self.captures = self.captures.union((capture,))
        return capture

    async def close(self) -> None:
        self.closing = True
        if not all(await asyncio.gather(*(capture.wait() for capture in self.captures))):
            verbose_proxy_logger.warning("Some requests ended without a confirmed durable charge receipt")


class ChargeCapture:
    def __init__(self, runtime: ChargeReceiptRuntime, context: BillingContext) -> None:
        self.runtime: Final = runtime
        self.context: Final = context
        self.event_id: Final = str(uuid4())
        self.done: Final = asyncio.Event()
        self.receipt: ChargeReceipt | None = None
        self.result: CommittedCharge | FailedCharge | None = None
        self.task: asyncio.Task[CommittedCharge | FailedCharge] | None = None

    def fork(self) -> "ChargeCapture":
        return self.runtime.bind(self.context)

    async def capture_success(
        self, details: Mapping[str, object], response: object, start_time: datetime, end_time: datetime
    ) -> CommittedCharge | FailedCharge:
        event: Final = project_spend_event(details, response, start_time, end_time, store_bodies=False)
        return await self.capture_event(
            SpendEventBuildError("unpriced_charge")
            if isinstance(event, SpendEvent) and event.response_cost == 0 and details.get("response_cost") is None
            else event,
            details,
        )

    async def capture_failure(
        self, details: Mapping[str, object], cost: float, start_time: datetime, end_time: datetime
    ) -> CommittedCharge | FailedCharge:
        event: Final = project_spend_event(details, None, start_time, end_time, store_bodies=False)
        return await self.capture_event(
            event.model_copy(update={"response_cost": cost})
            if isinstance(event, SpendEvent) and cost > 0
            else SpendEventBuildError("unpriced_failure"),
            details,
        )

    async def capture_event(
        self, event: SpendEvent | SpendEventBuildError, details: Mapping[str, object] | None = None
    ) -> CommittedCharge | FailedCharge:
        if self.task is None or self.task.done() and self.receipt is None:
            self.done.clear()
            self.result = None
            self.runtime.captures = self.runtime.captures.union((self,))
            self.task = asyncio.create_task(self._capture(event, details))
        return await asyncio.shield(self.task)

    async def _capture(
        self, event: SpendEvent | SpendEventBuildError, details: Mapping[str, object] | None
    ) -> CommittedCharge | FailedCharge:
        try:
            self.result = await self._commit(event, details)
            return self.result
        finally:
            self.done.set()
            self.runtime.captures = self.runtime.captures.difference((self,))

    async def _commit(
        self, event: SpendEvent | SpendEventBuildError, details: Mapping[str, object] | None
    ) -> CommittedCharge | FailedCharge:
        if isinstance(event, SpendEventBuildError) or event.response_cost is None or not event.model:
            return FailedCharge(self.context.key_hash, "invalid_charge")
        try:
            usage: Final = _Usage.model_validate(event.combined_usage or event.standard_logging_object or {})
            receipt: Final = ChargeReceipt(
                event_id=self.event_id,
                request_id=self.context.request_id,
                gateway_scope=self.runtime.settings.gateway_scope,
                key_hash=self.context.key_hash,
                amount_usd="0" if event.response_cost == 0 else format(Decimal(str(event.response_cost)), "f"),
                model=event.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )
        except ValidationError:
            return FailedCharge(self.context.key_hash, "invalid_charge")
        self.receipt = receipt
        if self.runtime.finalize is not None and details is not None:
            try:
                with anyio.fail_after(self.runtime.settings.commit_timeout_seconds):
                    await self.runtime.finalize(details, event, self)
            except Exception:  # noqa: BLE001  # optional accounting cannot discard the durable charge
                verbose_proxy_logger.warning("Accounting finalization failed; retaining the durable charge")
        return self.result if self.result is not None else await self.persist()

    async def persist(self) -> CommittedCharge | FailedCharge:
        assert self.receipt is not None
        if self.result is not None:
            return self.result
        with anyio.move_on_after(self.runtime.settings.commit_timeout_seconds):
            result: Final = await self._append_with_retry()
            self.result = result
            self.runtime.available = isinstance(result, CommittedCharge)
            self.done.set()
            return result
        self.runtime.available = False
        return FailedCharge(self.context.key_hash, "database_unavailable")

    async def _append_with_retry(self, remaining: int = 3) -> CommittedCharge | FailedCharge:
        assert self.receipt is not None
        result: Final = await self.runtime.store.append(self.receipt)
        if isinstance(result, CommittedCharge) or result.reason != "database_unavailable" or remaining == 1:
            return result
        await asyncio.sleep(0.05)
        return await self._append_with_retry(remaining - 1)

    async def wait(self) -> bool:
        with anyio.move_on_after(2 * self.runtime.settings.commit_timeout_seconds + 2):
            await self.done.wait()
        return isinstance(self.result, CommittedCharge)
