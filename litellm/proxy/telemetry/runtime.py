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
from litellm.proxy.telemetry.store import Database, LocalTableExporter, TelemetryStore, store_read_errors
from litellm.telemetry.aggregate import AggregatingSink
from litellm.telemetry.consent import OFF, ConsentGatedSink, TelemetryConsent
from litellm.telemetry.http_exporter import HttpExporter
from litellm.telemetry.records import InstanceInfo, TelemetryGroup
from litellm.telemetry.report import Report
from litellm.telemetry.sink import Exporter, ExportOutcome
from litellm.types.llms.custom_http import httpxSpecialProvider

StoredConsent: TypeAlias = Callable[[], Awaitable[TelemetryConsent | None]]


def proxy_hash_secret() -> bytes:
    salt_key: Final = get_salt_key()
    return salt_key.encode() if salt_key is not None else secrets.token_bytes(32)


async def deployment_hash_secret(store: TelemetryStore | None, fallback: Callable[[], bytes]) -> bytes:
    if store is None:
        return fallback()
    try:
        return await store.deployment_hash_secret()
    except store_read_errors() as e:
        verbose_proxy_logger.warning(
            "telemetry: hashing deployments with the salt key, stored secret unreadable: %s", e
        )
        return fallback()


def deployment_hasher(secret: bytes) -> Callable[[str], str]:
    key: Final = hashlib.sha256(b"litellm-telemetry-deployment:" + secret).digest()
    return lambda model_id: hmac.new(key, model_id.encode(), hashlib.sha256).hexdigest()[:16]


class _RememberingExporter:
    def __init__(self, inner: Exporter) -> None:
        self._inner: Final = inner
        self.last_report: Report | None = None

    async def export(self, report: Report) -> ExportOutcome:
        outcome: Final = await self._inner.export(report)
        self.last_report = report if outcome is ExportOutcome.SENT else self.last_report
        return outcome


def shared_http_client() -> httpx.AsyncClient:
    return get_async_httpx_client(httpxSpecialProvider.LoggingCallback, params={"timeout": 10.0}).client


def _exporter(
    endpoint: str | None, store: TelemetryStore | None, http_client: Callable[[], httpx.AsyncClient]
) -> Exporter:
    if endpoint is not None:
        return HttpExporter(http_client(), endpoint)
    assert store is not None, "start() returns early when there is neither an endpoint nor a database"
    return LocalTableExporter(store)


class TelemetryRuntime:
    """Owns the proxy's telemetry sink, re-reads which groups are on before every window, and flushes each window"""

    def __init__(self) -> None:
        self.sink: ConsentGatedSink | None = None
        self.store: TelemetryStore | None = None
        self.policy: EnvPolicy | None = None
        self.settings: TelemetrySettings = TelemetrySettings.model_construct()
        self.litellm_version: str = ""
        self._exporter: _RememberingExporter | None = None
        self._instance: InstanceInfo | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._flushing: asyncio.Future[None] | None = None
        self._pending: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs keep finalizers alive

    @property
    def last_report(self) -> Report | None:
        return self._exporter.last_report if self._exporter is not None else None

    def spawn(self, coroutine: Coroutine[None, None, None]) -> None:
        task: Final = asyncio.create_task(coroutine)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def start(
        self,
        *,
        litellm_version: str,
        settings: TelemetrySettings,
        db: Callable[[], Database | None],
        register: Callable[[TelemetryAttemptLogger], None],
        http_client: Callable[[], httpx.AsyncClient] = shared_http_client,
        hash_secret: Callable[[], bytes] = proxy_hash_secret,
        flush_interval_s: float = FLUSH_INTERVAL_SECONDS,
    ) -> None:
        self.settings, self.litellm_version = settings, litellm_version
        policy: Final = env_policy(settings)
        self.policy = policy
        if policy.locked_off:
            return
        database: Final = db()
        store: Final = TelemetryStore(database, settings.retention_days) if database is not None else None
        if settings.endpoint is None and store is None:
            return
        self.store = store
        endpoint: Final = str(settings.endpoint) if settings.endpoint is not None else None
        self._exporter = _RememberingExporter(_exporter(endpoint, store, http_client))
        register(
            TelemetryAttemptLogger(
                lambda: self.sink,
                deployment_hasher(await deployment_hash_secret(store, hash_secret)),
                blocks_enabled=self._counts_blocks,
            )
        )
        await self.refresh()
        self._flush_task = asyncio.create_task(self._flush_every(flush_interval_s))

    def _counts_blocks(self) -> bool:
        sink: Final = self.sink
        return sink is not None and sink.consent.allows(TelemetryGroup.EVENT_DETAILS)

    async def stored_consent(self) -> TelemetryConsent | None:
        store: Final = self.store
        stored: Final = await store.consent() if store is not None else None
        if stored is None or isinstance(stored, TelemetryConsent):
            return stored
        verbose_proxy_logger.warning("telemetry: ignoring stored settings, %s", stored.message())
        return None

    async def _instance_info(self) -> InstanceInfo:
        known: Final = self._instance
        if known is not None:
            return known
        store: Final = self.store
        instance: Final = InstanceInfo(
            instance_id=await store.instance_id() if store is not None else uuid.uuid4().hex,
            litellm_version=self.litellm_version,
        )
        self._instance = instance
        return instance

    async def refresh(self) -> None:
        """Apply the current groups; only called between windows so every report has a single set of groups"""
        policy: Final = self.policy
        exporter: Final = self._exporter
        if policy is None or exporter is None:
            return
        try:
            consent: Final = policy.effective(await self.stored_consent() if policy.pinned is None else None)
            current: Final = self.sink
            if current is not None and current.consent == consent:
                return
            if consent == OFF:
                self.sink = None
                return
            gated: Final = ConsentGatedSink(AggregatingSink(exporter), consent)
            gated.set_instance(await self._instance_info())
            self.sink = gated
        except store_read_errors() as e:
            verbose_proxy_logger.debug("telemetry: keeping the current groups, could not read stored settings: %s", e)

    async def _flush_every(self, interval_s: float) -> None:
        while True:
            await asyncio.sleep(interval_s)
            await self._flush_window()
            await self.refresh()

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
        self.store = None
