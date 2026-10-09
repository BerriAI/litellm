import asyncio
from typing import Final

import httpx
import pytest

from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.proxy.telemetry.runtime import TelemetryRuntime, deployment_hasher
from litellm.proxy.telemetry.settings import TelemetrySettings
from litellm.telemetry.consent import ConsentGatedSink, TelemetryConsent
from litellm.telemetry.records import AttemptRecord, InstanceInfo, RequestRecord, TelemetryGroup, UIEvent

_ENDPOINT: Final = "http://telemetry.invalid/v1/reports"


def _offline() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(204)))


_HEARTBEAT: Final = TelemetryConsent(frozenset({TelemetryGroup.HEARTBEAT}))


async def _stored_heartbeat() -> TelemetryConsent | None:
    return _HEARTBEAT


async def _broken_store() -> TelemetryConsent | None:
    raise ConnectionError("db down")


@pytest.mark.parametrize(
    "settings",
    [
        TelemetrySettings(),
        TelemetrySettings(groups="", endpoint=_ENDPOINT),
        TelemetrySettings(groups="heartbeat,verbose", endpoint=_ENDPOINT),
        TelemetrySettings(groups="heartbeat"),
        TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, disabled=True),
    ],
)
@pytest.mark.asyncio
async def test_telemetry_stays_off_unless_valid_groups_and_an_endpoint_are_set_and_not_vetoed(
    settings: TelemetrySettings,
) -> None:
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(litellm_version="1.0.0", settings=settings, register=registered.append, http_client=_offline)
    assert runtime.sink is None
    assert registered == []


@pytest.mark.asyncio
async def test_pinned_groups_register_the_attempt_logger_and_stop_cleanly() -> None:
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="HEARTBEAT", endpoint=_ENDPOINT, flush_interval_seconds=3600),
        register=registered.append,
        http_client=_offline,
    )
    assert runtime.sink is not None and runtime.sink.consent == _HEARTBEAT
    assert len(registered) == 1
    await runtime.stop()
    assert runtime.sink is None


@pytest.mark.asyncio
async def test_stored_groups_apply_when_no_groups_are_pinned_and_a_veto_ignores_them() -> None:
    enabled: Final = TelemetryRuntime()
    await enabled.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(endpoint=_ENDPOINT, flush_interval_seconds=3600),
        register=lambda _logger: None,
        http_client=_offline,
        stored=_stored_heartbeat,
    )
    vetoed: Final = TelemetryRuntime()
    await vetoed.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(endpoint=_ENDPOINT, disabled=True),
        register=lambda _logger: None,
        http_client=_offline,
        stored=_stored_heartbeat,
    )
    assert enabled.sink is not None and enabled.sink.consent == _HEARTBEAT
    assert vetoed.sink is None
    await enabled.stop()


@pytest.mark.asyncio
async def test_a_failing_settings_store_leaves_telemetry_off_instead_of_crashing_startup() -> None:
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(endpoint=_ENDPOINT, flush_interval_seconds=3600),
        register=lambda _logger: None,
        http_client=_offline,
        stored=_broken_store,
    )
    assert runtime.sink is None
    await runtime.stop()


def test_deployment_hashes_are_stable_per_install_and_differ_across_installs() -> None:
    assert deployment_hasher("install-a")("model-1") == deployment_hasher("install-a")("model-1")
    assert deployment_hasher("install-a")("model-1") != deployment_hasher("install-b")("model-1")
    assert "model-1" not in deployment_hasher("install-a")("model-1")


@pytest.mark.asyncio
async def test_stop_waits_for_in_flight_request_finalizers_before_the_last_flush() -> None:
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, flush_interval_seconds=3600),
        register=lambda _logger: None,
        http_client=_offline,
    )
    finished: Final[list[bool]] = []  # mutable-ok: records that the finalizer ran to completion

    async def _finalizer() -> None:
        await asyncio.sleep(0.01)
        finished.append(True)

    runtime.spawn(_finalizer())
    await runtime.stop()
    assert finished == [True]


@pytest.mark.asyncio
async def test_stopping_a_runtime_that_never_started_is_a_no_op() -> None:
    runtime: Final = TelemetryRuntime()
    await runtime.stop()
    assert runtime.sink is None


class _HeldFlushSink:
    def __init__(self) -> None:
        self.started: Final = asyncio.Event()
        self.release: Final = asyncio.Event()
        self.flushes: Final[list[bool]] = []  # mutable-ok: records each flush that ran to completion

    def set_instance(self, info: InstanceInfo) -> None: ...

    def record_request(self, record: RequestRecord) -> None: ...

    def record_attempt(self, record: AttemptRecord) -> None: ...

    def record_ui_event(self, event: UIEvent) -> None: ...

    async def flush(self) -> None:
        self.started.set()
        await self.release.wait()
        self.flushes.append(True)


@pytest.mark.asyncio
async def test_stop_during_an_export_lets_that_export_finish() -> None:
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, flush_interval_seconds=0.001),
        register=lambda _logger: None,
        http_client=_offline,
    )
    held: Final = _HeldFlushSink()
    runtime.sink = ConsentGatedSink(held, _HEARTBEAT)
    await held.started.wait()
    stopping: Final = asyncio.create_task(runtime.stop())
    await asyncio.sleep(0)
    held.release.set()
    await stopping
    assert len(held.flushes) == 2


async def test_settle_timeout_reads_the_configured_value_after_start_and_the_default_before() -> None:
    runtime: Final = TelemetryRuntime()
    assert runtime.settings.settle_timeout_seconds == 2.0
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, settle_timeout_seconds=7.5),
        register=lambda _logger: None,
        http_client=_offline,
    )
    assert runtime.settings.settle_timeout_seconds == 7.5
    await runtime.stop()
