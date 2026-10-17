import asyncio
import os
import signal
from typing import Final
from uuid import uuid4

import httpx
from pydantic import ConfigDict, Field, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy.spend_tracking.charge_receipt_request import writer_query
from litellm.proxy.spend_tracking.charge_receipts import ChargeReceiptSettings, ChargeReceiptStore
from litellm.types.llms.base import LiteLLMBaseModel

MAX_ACKNOWLEDGEMENT_BYTES: Final = 256 * 1024


class _Acknowledgement(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    acknowledged: tuple[str, ...] = Field(max_length=32)


class ChargeReceiptPublisher:
    def __init__(
        self,
        store: ChargeReceiptStore,
        settings: ChargeReceiptSettings,
        client: httpx.AsyncClient,
    ) -> None:
        self.store: Final = store
        self.settings: Final = settings
        self.client: Final = client
        self.owner: Final = str(uuid4())

    async def deliver_once(self) -> int:
        receipts: Final = await self.store.claim(self.owner, self.settings.batch_size)
        if not receipts:
            return 0
        identities: Final = tuple(receipt.event_id for receipt in receipts)
        try:
            async with self.client.stream(
                "POST",
                self.settings.destination_url,
                json={"events": tuple(receipt.model_dump() for receipt in receipts)},
                headers={
                    "Authorization": "Bearer " + self.settings.destination_secret.get_secret_value(),
                    "Cache-Control": "no-store",
                    "Accept-Encoding": "identity",
                },
                follow_redirects=False,
                timeout=self.settings.delivery_timeout_seconds,
            ) as response:
                if response.status_code != 200:
                    await self.store.retry(self.owner, identities, "receiver_unavailable")
                    return 0
                if response.headers.get("content-encoding", "identity") != "identity":
                    await self.store.retry(self.owner, identities, "invalid_acknowledgement")
                    return 0
                body: Final = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > MAX_ACKNOWLEDGEMENT_BYTES:
                        await self.store.retry(self.owner, identities, "invalid_acknowledgement")
                        return 0
                    body.extend(chunk)
            acknowledgement: Final = _Acknowledgement.model_validate_json(body)
            acknowledged: Final = frozenset(acknowledgement.acknowledged)
            if not acknowledged.issubset(identities):
                await self.store.retry(self.owner, identities, "invalid_acknowledgement")
                return 0
            await self.store.acknowledge(self.owner, tuple(acknowledged))
            retained: Final = tuple(identity for identity in identities if identity not in acknowledged)
            await self.store.retry(self.owner, retained, "receipt_rejected")
            if retained:
                verbose_proxy_logger.warning("Charge receipt delivery retained %d rejected events", len(retained))
            return len(acknowledged)
        except httpx.HTTPError:
            await self.store.retry(self.owner, identities, "receiver_unavailable")
            return 0
        except ValidationError:
            await self.store.retry(self.owner, identities, "invalid_acknowledgement")
            return 0

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                await self.deliver_once()
            except Exception:  # noqa: BLE001  # a failed acknowledgement leaves the durable lease retryable
                verbose_proxy_logger.warning("Charge receipt delivery paused; pending events remain durable")
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.settings.poll_seconds)
            except TimeoutError:
                pass


async def run_charge_receipt_publisher() -> None:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging

    settings: Final = ChargeReceiptSettings()
    error: Final = settings.configuration_error()
    if error is not None or not settings.enabled:
        raise SystemExit(error or "Charge receipts are not enabled.")
    database_url: Final = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("Charge receipt delivery requires DATABASE_URL.")
    client: Final = PrismaClient(database_url=database_url, proxy_logging_obj=ProxyLogging(UserApiKeyCache()))
    stop: Final = asyncio.Event()
    loop: Final = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    try:
        await client.connect()
        store: Final = ChargeReceiptStore(writer_query(client))
        if not await store.ready():
            raise SystemExit("Charge receipt delivery requires the durable receipt schema.")
        async with httpx.AsyncClient() as http:
            await ChargeReceiptPublisher(store, settings, http).run(stop)
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signum)
        await client.disconnect()
