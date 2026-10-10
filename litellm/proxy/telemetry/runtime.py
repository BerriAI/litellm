import asyncio
import contextlib
import hashlib
import hmac
import secrets
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import MAX_CALLBACKS
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # legacy params: dict signature
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import get_salt_key
from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.proxy.telemetry.settings import (
    FLUSH_INTERVAL_SECONDS,
    EnvPolicy,
    TelemetryDestination,
    TelemetrySettings,
    env_policy,
)
from litellm.proxy.telemetry.store import Database, LocalTableExporter, TelemetryStore, store_read_errors
from litellm.telemetry.aggregate import AggregatingSink
from litellm.telemetry.consent import OFF, ConsentGatedSink, TelemetryConsent
from litellm.telemetry.http_exporter import HttpExporter
from litellm.telemetry.records import InstanceInfo, TelemetryGroup
from litellm.telemetry.report import Report
from litellm.telemetry.sink import Exporter, ExportOutcome
from litellm.types.llms.custom_http import httpxSpecialProvider

StoredConsent: TypeAlias = Callable[[], Awaitable[TelemetryConsent | None]]


@dataclass(frozen=True, slots=True)
class StoredChoice:
    consent: TelemetryConsent | None = None
    destination: TelemetryDestination | None = None


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


def register_attempt_logger(logger: TelemetryAttemptLogger) -> None:
    litellm.logging_callback_manager.add_litellm_callback(logger)
    if logger not in litellm.callbacks:
        verbose_proxy_logger.warning(
            "telemetry: litellm.callbacks already holds %s callbacks, provider attempts will not be recorded",
            MAX_CALLBACKS,
        )


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


def _available_exporters(
    endpoint: str | None, store: TelemetryStore | None, http_client: Callable[[], httpx.AsyncClient]
) -> Iterator[tuple[TelemetryDestination, _RememberingExporter]]:
    if endpoint is not None:
        yield TelemetryDestination.HTTP, _RememberingExporter(HttpExporter(http_client(), endpoint))
    if store is not None:
        yield TelemetryDestination.LOCAL_TABLE, _RememberingExporter(LocalTableExporter(store))


def _log_applied(consent: TelemetryConsent, destination: TelemetryDestination | None, *, recording: bool) -> None:
    groups: Final = sorted(group.value for group in consent.groups)
    if recording:
        verbose_proxy_logger.info(
            "telemetry: recording %s, reports go to %s", groups, destination and destination.value
        )
    elif consent != OFF:
        verbose_proxy_logger.warning(
            "telemetry: %s are on but no usable destination is chosen, recording nothing. Set "
            "LITELLM_TELEMETRY_DESTINATION or pick one on the Telemetry page",
            groups,
        )


class TelemetryRuntime:
    """Owns the proxy's telemetry sink, re-reads which groups are on before every window, and flushes each window"""

    def __init__(self) -> None:
        self.sink: ConsentGatedSink | None = None
        self.store: TelemetryStore | None = None
        self.policy: EnvPolicy | None = None
        self.settings: TelemetrySettings = TelemetrySettings.model_construct()
        self.litellm_version: str = ""
        self.destination: TelemetryDestination | None = None
        self._exporters: Mapping[TelemetryDestination, _RememberingExporter] = MappingProxyType({})
        self._applied: tuple[TelemetryConsent, TelemetryDestination | None] | None = None
        self._instance: InstanceInfo | None = None
        self._attach_logger: Callable[[], Awaitable[None]] | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._flushing: asyncio.Future[None] | None = None
        self._pending: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs keep finalizers alive

    @property
    def last_report(self) -> Report | None:
        exporter: Final = self._exporters.get(self.destination) if self.destination is not None else None
        return exporter.last_report if exporter is not None else None

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
        self._exporters = MappingProxyType(dict(_available_exporters(endpoint, store, http_client)))

        async def attach_logger() -> None:
            register(
                TelemetryAttemptLogger(
                    lambda: self.sink,
                    deployment_hasher(await deployment_hash_secret(store, hash_secret)),
                    blocks_enabled=self._counts_blocks,
                )
            )

        self._attach_logger = attach_logger
        await self.refresh()
        self._flush_task = asyncio.create_task(self._flush_every(flush_interval_s))

    async def _attach_logger_once(self) -> None:
        """Registers the attempt logger and creates its hash secret the first time a sink is built, so a proxy that never
        opts in carries no callback and writes nothing. It stays registered after groups are turned off, returning
        early, since removing it could skip in-flight calls"""
        attach: Final = self._attach_logger
        self._attach_logger = None
        if attach is not None:
            await attach()

    def _counts_blocks(self) -> bool:
        sink: Final = self.sink
        return sink is not None and sink.consent.allows(TelemetryGroup.EVENT_DETAILS)

    async def stored_choice(self) -> StoredChoice:
        store: Final = self.store
        stored: Final = await store.settings() if store is not None else None
        if stored is None:
            return StoredChoice()
        if isinstance(stored.consent, TelemetryConsent):
            return StoredChoice(stored.consent, stored.destination)
        verbose_proxy_logger.warning("telemetry: ignoring stored groups, %s", stored.consent.message())
        return StoredChoice(destination=stored.destination)

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
        """Apply the current groups and destination; only called between windows so every report has a single set"""
        policy: Final = self.policy
        if policy is None or not self._exporters:
            return
        try:
            pinned: Final = policy.pinned is not None and policy.destination is not None
            stored: Final = StoredChoice() if pinned else await self.stored_choice()
            consent: Final = policy.effective(stored.consent)
            destination: Final = policy.effective_destination(stored.destination)
            if self._applied == (consent, destination):
                return
            exporter: Final = self._exporters.get(destination) if destination is not None else None
            sink: Final = (
                ConsentGatedSink(AggregatingSink(exporter), consent)
                if consent != OFF and exporter is not None
                else None
            )
            if sink is not None:
                sink.set_instance(await self._instance_info())
                await self._attach_logger_once()
            self.sink, self.destination, self._applied = sink, destination, (consent, destination)
            _log_applied(consent, destination, recording=sink is not None)
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
