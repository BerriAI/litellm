import asyncio
from typing import Final

import httpx
import pytest

import litellm
from pydantic import HttpUrl

from litellm.proxy.telemetry.attempt_logger import TelemetryAttemptLogger
from litellm.constants import MAX_CALLBACKS
from litellm.proxy.telemetry.runtime import TelemetryRuntime, deployment_hasher, register_attempt_logger
from litellm.proxy.telemetry.settings import TelemetrySettings
from litellm.telemetry.consent import ConsentGatedSink, TelemetryConsent
from litellm.telemetry.records import AttemptRecord, InstanceInfo, RequestRecord, TelemetryGroup, UIEvent
from tests.unit.proxy.telemetry.fake_database import SettingsDatabase

_ENDPOINT: Final = HttpUrl("http://telemetry.invalid/v1/reports")


def _offline() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(204)))


_HEARTBEAT: Final = TelemetryConsent(frozenset({TelemetryGroup.HEARTBEAT}))
_SUCCESS: Final = TelemetryConsent(frozenset({TelemetryGroup.HEARTBEAT, TelemetryGroup.REQUEST_SUCCESS}))


async def _started(settings: TelemetrySettings, db: SettingsDatabase | None) -> TelemetryRuntime:
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0", settings=settings, db=lambda: db, register=lambda _logger: None, http_client=_offline
    )
    return runtime


@pytest.mark.parametrize(
    "settings",
    [
        TelemetrySettings(),
        TelemetrySettings(groups="", endpoint=_ENDPOINT),
        TelemetrySettings(groups="heartbeat"),
        TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT, disabled=True),
    ],
)
@pytest.mark.asyncio
async def test_without_a_database_telemetry_needs_valid_pinned_groups_an_endpoint_and_no_veto(
    settings: TelemetrySettings,
) -> None:
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0", settings=settings, db=lambda: None, register=registered.append, http_client=_offline
    )
    assert runtime.sink is None
    assert registered == []


@pytest.mark.asyncio
async def test_a_veto_never_opens_the_database() -> None:
    opened: Final[list[bool]] = []  # mutable-ok: records whether the db factory ran

    def _db() -> SettingsDatabase:
        opened.append(True)
        return SettingsDatabase(stored_groups='{"groups": ["heartbeat"]}')

    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0", settings=TelemetrySettings(disabled=True), db=_db, register=lambda _logger: None
    )
    assert (runtime.sink, runtime.store, opened) == (None, None, [])


@pytest.mark.asyncio
async def test_pinned_groups_register_the_attempt_logger_and_stop_cleanly() -> None:
    registered: Final[list[TelemetryAttemptLogger]] = []  # mutable-ok: captures the register callback
    runtime: Final = TelemetryRuntime()
    await runtime.start(
        litellm_version="1.0.0",
        settings=TelemetrySettings(groups="HEARTBEAT", endpoint=_ENDPOINT),
        db=lambda: None,
        register=registered.append,
        http_client=_offline,
    )
    assert runtime.sink is not None and runtime.sink.consent == _HEARTBEAT
    assert len(registered) == 1
    await runtime.stop()
    assert runtime.sink is None


@pytest.mark.asyncio
async def test_pinned_groups_win_over_the_stored_groups() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["heartbeat", "request_success"]}')
    runtime: Final = await _started(TelemetrySettings(groups="heartbeat"), db)
    assert runtime.sink is not None and runtime.sink.consent == _HEARTBEAT
    await runtime.stop()


@pytest.mark.asyncio
async def test_stored_groups_are_picked_up_at_the_next_window_without_a_restart() -> None:
    db: Final = SettingsDatabase()
    runtime: Final = await _started(TelemetrySettings(), db)
    assert runtime.sink is None
    db.stored_groups = '{"groups": ["heartbeat", "request_success"]}'
    await runtime.refresh()
    assert runtime.sink is not None and runtime.sink.consent == _SUCCESS
    db.stored_groups = '{"groups": []}'
    await runtime.refresh()
    assert runtime.sink is None
    await runtime.stop()


@pytest.mark.asyncio
async def test_invalid_stored_groups_are_ignored_and_telemetry_stays_off() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["token_info"]}')
    runtime: Final = await _started(TelemetrySettings(), db)
    assert runtime.sink is None
    await runtime.stop()


@pytest.mark.asyncio
async def test_a_failing_settings_read_keeps_the_current_groups() -> None:
    db: Final = SettingsDatabase(stored_groups='{"groups": ["heartbeat"]}')
    runtime: Final = await _started(TelemetrySettings(), db)
    db.fail_reads = True
    await runtime.refresh()
    assert runtime.sink is not None and runtime.sink.consent == _HEARTBEAT
    await runtime.stop()


def test_deployment_hashes_are_stable_per_secret_and_differ_across_secrets() -> None:
    assert deployment_hasher(b"secret-a")("model-1") == deployment_hasher(b"secret-a")("model-1")
    assert deployment_hasher(b"secret-a")("model-1") != deployment_hasher(b"secret-b")("model-1")
    assert "model-1" not in deployment_hasher(b"secret-a")("model-1")


@pytest.mark.asyncio
async def test_stop_waits_for_in_flight_request_finalizers_before_the_last_flush() -> None:
    runtime: Final = await _started(TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT), None)
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
        settings=TelemetrySettings(groups="heartbeat", endpoint=_ENDPOINT),
        flush_interval_s=0.001,
        db=lambda: None,
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
        db=lambda: None,
        register=lambda _logger: None,
        http_client=_offline,
    )
    assert runtime.settings.settle_timeout_seconds == 7.5
    await runtime.stop()


@pytest.fixture
def full_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "callbacks", [f"callback_{i}" for i in range(MAX_CALLBACKS)])


@pytest.mark.usefixtures("full_callbacks")
def test_a_full_callback_list_warns_that_attempts_will_not_be_recorded(caplog: pytest.LogCaptureFixture) -> None:
    logger: Final = TelemetryAttemptLogger(lambda: None, deployment_hasher(b"secret"))

    register_attempt_logger(logger)

    assert logger not in litellm.callbacks
    assert "provider attempts will not be recorded" in caplog.text


def test_registering_the_attempt_logger_adds_it_once_without_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(litellm, "callbacks", [])
    logger: Final = TelemetryAttemptLogger(lambda: None, deployment_hasher(b"secret"))

    register_attempt_logger(logger)
    register_attempt_logger(logger)

    assert litellm.callbacks == [logger]
    assert "will not be recorded" not in caplog.text
