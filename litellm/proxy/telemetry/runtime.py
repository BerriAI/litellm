import asyncio
import contextlib
import hashlib
import hmac
import secrets
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from typing import Final, TypeAlias

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # legacy params: dict signature
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import get_salt_key
from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.proxy.telemetry.settings import FLUSH_INTERVAL_SECONDS, EnvPolicy, TelemetrySettings, env_policy
from litellm.telemetry.aggregate import AggregatingSink
from litellm.telemetry.consent import OFF, ConsentGatedSink, TelemetryConsent
from litellm.telemetry.http_exporter import HttpExporter
from litellm.telemetry.records import InstanceInfo, TelemetryGroup
from litellm.telemetry.sink import Exporter
from litellm.types.llms.custom_http import httpxSpecialProvider


def shared_http_client() -> httpx.AsyncClient:
    return get_async_httpx_client(httpxSpecialProvider.LoggingCallback, params={"timeout": 10.0}).client


StoredConsent: TypeAlias = Callable[[], Awaitable[TelemetryConsent | None]]


async def nothing_stored() -> TelemetryConsent | None:
    return None


def proxy_hash_secret() -> bytes:
    salt_key: Final = get_salt_key()
    return salt_key.encode() if salt_key is not None else secrets.token_bytes(32)


def deployment_hasher(secret: bytes) -> Callable[[str], str]:
    key: Final = hashlib.sha256(b"litellm-telemetry-deployment:" + secret).digest()
    return lambda model_id: hmac.new(key, model_id.encode(), hashlib.sha256).hexdigest()[:16]


class TelemetryRuntime:
    """Owns the proxy's telemetry sink, re-reads which groups are on before every window, and flushes each window"""

    def __init__(self) -> None:
        self.sink: ConsentGatedSink | None = None
        self.settings: TelemetrySettings = TelemetrySettings.model_construct()
        self.policy: EnvPolicy | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._flushing: asyncio.Future[None] | None = None
        self._pending: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs keep finalizers alive

    def spawn(self, coroutine: Coroutine[None, None, None]) -> None:
        task: Final = asyncio.create_task(coroutine)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def start(
        self,
        *,
        litellm_version: str,
        settings: TelemetrySettings,
        register: Callable[[TelemetryAttemptLogger], None],
        stored: StoredConsent = nothing_stored,
        http_client: Callable[[], httpx.AsyncClient] = shared_http_client,
        hash_secret: Callable[[], bytes] = proxy_hash_secret,
        flush_interval_s: float = FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self.settings = settings
        policy: Final = env_policy(settings)
        self.policy = policy
        if policy.locked_off:
            return
        if settings.endpoint is None:
            return
        exporter: Final = HttpExporter(http_client(), str(settings.endpoint))
        instance: Final = InstanceInfo(instance_id=uuid.uuid4().hex, litellm_version=litellm_version)
        register(
            TelemetryAttemptLogger(
                lambda: self.sink, deployment_hasher(hash_secret()), blocks_enabled=self._counts_blocks
            )
        )
        await self._refresh(policy, stored, exporter, instance)
        self._flush_task = asyncio.create_task(self._flush_every(policy, stored, exporter, instance, flush_interval_s))

    def _counts_blocks(self) -> bool:
        sink: Final = self.sink
        return sink is not None and sink.consent.allows(TelemetryGroup.EVENT_DETAILS)

    async def _refresh(
        self, policy: EnvPolicy, stored: StoredConsent, exporter: Exporter, instance: InstanceInfo
    ) -> None:
        current: Final = self.sink
        try:
            consent: Final = policy.effective(await stored())
        except Exception:  # noqa: BLE001 -- a failed settings read keeps the current groups rather than flapping off
            verbose_proxy_logger.debug("telemetry: could not read stored settings", exc_info=True)
            return
        if current is not None and current.consent == consent:
            return
        if consent == OFF:
            self.sink = None
            return
        gated: Final = ConsentGatedSink(AggregatingSink(exporter), consent)
        gated.set_instance(instance)
        self.sink = gated

    async def _flush_every(
        self, policy: EnvPolicy, stored: StoredConsent, exporter: Exporter, instance: InstanceInfo, interval_s: float
    ) -> None:
        while True:
            await asyncio.sleep(interval_s)
            await self._flush_window()
            await self._refresh(policy, stored, exporter, instance)

    async def _flush_window(self) -> None:
        flushing: Final = asyncio.ensure_future(self._flush_current())
        self._flushing = flushing
        await asyncio.shield(flushing)

    async def _flush_current(self) -> None:
        sink: Final = self.sink
        if sink is not None:
            await sink.flush()

    async def stop(self) -> None:
        flush_task: Final = self._flush_task
        if flush_task is not None:
            flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await flush_task
        flushing: Final = self._flushing
        if flushing is not None:
            await asyncio.gather(flushing, return_exceptions=True)
        await asyncio.gather(*self._pending, return_exceptions=True)
        await self._flush_current()
        self.sink = None
