import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Final

import pytest

from litellm._service_logger import ServiceLogging, ServiceTypes
from litellm.proxy.db.native_database import (
    ServiceLoggingEvents,
    TokenRotation,
    keep_token_fresh,
    native_endpoint,
    reader_token_rotation,
    writer_token_rotation,
)
from litellm.proxy.db.pgbouncer import PGBOUNCER_POOLED_ENV_VAR
from litellm.proxy.db.token_auth import IAMEndpoint, RdsIamTokenAuth
from litellm.rust_bridge._native import NativeDatabase
from litellm.rust_bridge.database import NativeDatabaseEndpoint, NativePoolLimits

NOW: Final = datetime(2026, 11, 3, 12, 0, 0)


def _rds_token(issued: datetime, lifetime_seconds: int) -> str:
    return (
        "db.internal:5432/?Action=connect&DBUser=litellm"
        f"&X-Amz-Date={issued:%Y%m%dT%H%M%SZ}&X-Amz-Expires={lifetime_seconds}&X-Amz-Signature=abc"
    )


@pytest.mark.parametrize(
    "url, expected",
    [
        pytest.param(
            "postgresql://litellm:p%40ss%3Aword@db.internal:6543/proxy"
            "?connection_limit=12&pool_timeout=20&max_idle_connection_lifetime=60",
            NativeDatabaseEndpoint(
                host="db.internal",
                port=6543,
                username="litellm",
                password="p@ss:word",
                database="proxy",
                ssl_mode="prefer",
                ssl_root_cert=None,
                options=(),
                statement_cache=True,
                pool=NativePoolLimits(max_connections=12, acquire_timeout_seconds=20.0, idle_timeout_seconds=60.0),
            ),
            id="password_and_pool_params",
        ),
        pytest.param(
            "postgresql://svc%40corp@db.internal/proxy?sslmode=require&sslcert=/certs/root.pem&sslaccept=strict"
            "&schema=Tenant&options=-c%20statement_timeout%3D30000%20-clock_timeout%3D5000"
            "&pgbouncer=true&pool_timeout=0",
            NativeDatabaseEndpoint(
                host="db.internal",
                port=5432,
                username="svc@corp",
                password=None,
                database="proxy",
                ssl_mode="verify-full",
                ssl_root_cert="/certs/root.pem",
                options=(("search_path", '"Tenant"'), ("statement_timeout", "30000"), ("lock_timeout", "5000")),
                statement_cache=False,
                pool=NativePoolLimits(max_connections=None, acquire_timeout_seconds=None, idle_timeout_seconds=None),
            ),
            id="tls_schema_timeouts_and_pgbouncer",
        ),
        pytest.param(
            "postgresql://u@db/proxy?sslmode=require&schema=public",
            NativeDatabaseEndpoint(
                host="db",
                port=5432,
                username="u",
                password=None,
                database="proxy",
                ssl_mode="require",
                ssl_root_cert=None,
                options=(),
                statement_cache=True,
                pool=NativePoolLimits(max_connections=None, acquire_timeout_seconds=None, idle_timeout_seconds=None),
            ),
            id="require_without_verification_and_default_schema",
        ),
        pytest.param(
            "postgresql://u@db/proxy?sslmode=disable&sslaccept=strict",
            NativeDatabaseEndpoint(
                host="db",
                port=5432,
                username="u",
                password=None,
                database="proxy",
                ssl_mode="disable",
                ssl_root_cert=None,
                options=(),
                statement_cache=True,
                pool=NativePoolLimits(max_connections=None, acquire_timeout_seconds=None, idle_timeout_seconds=None),
            ),
            id="disable_wins_over_strict",
        ),
    ],
)
def test_native_endpoint_reads_the_prisma_url(url: str, expected: NativeDatabaseEndpoint) -> None:
    assert native_endpoint(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        pytest.param("postgresql://db.internal/proxy", id="missing_user"),
        pytest.param("postgresql://u@db/proxy?options=statement_timeout%3D5", id="options_not_settings"),
    ],
)
def test_native_endpoint_rejects_urls_it_cannot_honor(url: str) -> None:
    with pytest.raises(ValueError, match="DATABASE_URL"):
        native_endpoint(url)


def test_writer_token_rotation_follows_token_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    url: Final = "postgresql://litellm:tok@db.internal:5432/proxy?schema=s"
    monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
    monkeypatch.delenv(PGBOUNCER_POOLED_ENV_VAR, raising=False)
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "false")
    assert writer_token_rotation(url) is None

    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("AWS_RDS_REGION", "us-west-2")
    assert writer_token_rotation(url) == TokenRotation(
        auth=RdsIamTokenAuth(region="us-west-2"),
        endpoint=IAMEndpoint(host="db.internal", port="5432", user="litellm", name="proxy", schema="s"),
    )


def test_reader_token_rotation_uses_the_reader_region_even_when_the_writer_is_pooled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url: Final = "postgresql://reader:tok@replica.internal:5433/proxy"
    monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
    monkeypatch.setenv(PGBOUNCER_POOLED_ENV_VAR, "true")
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "false")
    assert reader_token_rotation(url) is None

    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv("AWS_RDS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_RDS_READ_REPLICA_REGION", "eu-west-1")
    assert reader_token_rotation(url) == TokenRotation(
        auth=RdsIamTokenAuth(region="eu-west-1"),
        endpoint=IAMEndpoint(host="replica.internal", port="5433", user="reader", name="proxy"),
    )


def test_pooled_writer_needs_no_token_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv(PGBOUNCER_POOLED_ENV_VAR, "true")

    assert writer_token_rotation("postgresql://litellm:local@127.0.0.1:6432/proxy") is None


@pytest.mark.parametrize(
    "token, expected_seconds",
    [
        pytest.param(_rds_token(NOW, 900), 720.0, id="refreshes_buffer_before_expiry"),
        pytest.param(_rds_token(NOW - timedelta(seconds=890), 900), 30.0, id="never_sooner_than_retry_floor"),
        pytest.param("opaque", 600.0, id="unreadable_expiry_uses_fallback"),
    ],
)
def test_token_rotation_schedules_the_next_refresh(token: str, expected_seconds: float) -> None:
    rotation: Final = TokenRotation(
        auth=RdsIamTokenAuth(), endpoint=IAMEndpoint(host="db", port="5432", user="u", name="n")
    )

    assert rotation.seconds_until_refresh(token, NOW) == expected_seconds


class _Sink:
    def __init__(self, wanted: int) -> None:
        self.passwords: tuple[str, ...] = ()
        self._wanted: Final = wanted
        self.done: Final = asyncio.Event()

    def install(self, password: str) -> None:
        self.passwords = (*self.passwords, password)
        if len(self.passwords) >= self._wanted:
            self.done.set()


class _Schedule:
    def __init__(self) -> None:
        self.delays: tuple[float, ...] = ()

    async def sleep(self, seconds: float) -> None:
        self.delays = (*self.delays, seconds)
        await asyncio.sleep(0)


def _minting(*outcomes: str | Exception) -> TokenRotation:
    remaining: Final = iter(outcomes)

    def mint(_auth: object, _endpoint: object) -> str:
        outcome: Final = next(remaining)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return TokenRotation(
        auth=RdsIamTokenAuth(), endpoint=IAMEndpoint(host="db", port="5432", user="u", name="n"), mint_token=mint
    )


async def _run_until(sink: _Sink, rotation: TokenRotation, schedule: _Schedule) -> None:
    task: Final = asyncio.create_task(
        keep_token_fresh(sink.install, rotation, "opaque", now=lambda: NOW, sleep=schedule.sleep)
    )
    await asyncio.wait_for(sink.done.wait(), timeout=5)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_keep_token_fresh_installs_each_minted_token() -> None:
    sink: Final = _Sink(wanted=2)
    schedule: Final = _Schedule()

    await _run_until(sink, _minting("first%2Ftoken", _rds_token(NOW, 900)), schedule)

    assert sink.passwords == ("first/token", _rds_token(NOW, 900))
    assert schedule.delays[:2] == (600.0, 600.0)


@pytest.mark.asyncio
async def test_keep_token_fresh_retries_a_failed_mint() -> None:
    sink: Final = _Sink(wanted=1)
    schedule: Final = _Schedule()

    await _run_until(sink, _minting(RuntimeError("sts down"), "recovered"), schedule)

    assert sink.passwords == ("recovered",)
    assert schedule.delays[:2] == (600.0, 30.0)


class _RecordingServiceLogging(ServiceLogging):
    def __init__(self) -> None:
        super().__init__(mock_testing=True)
        self.events: tuple[tuple[str, str, float, Mapping[str, str] | None, str | None], ...] = ()
        self.arrived: Final = asyncio.Event()

    async def async_service_success_hook(
        self,
        service: ServiceTypes,
        call_type: str,
        duration: float,
        parent_otel_span: object = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: dict[str, str] | None = None,
        caller: str | None = None,
    ) -> None:
        self.events = (*self.events, (service.value, call_type, duration, event_metadata, None))
        self.arrived.set()

    async def async_service_failure_hook(
        self,
        service: ServiceTypes,
        duration: float,
        error: str | Exception,
        call_type: str,
        parent_otel_span: object = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: dict[str, str] | None = None,
        caller: str | None = None,
    ) -> None:
        self.events = (*self.events, (service.value, call_type, duration, event_metadata, str(error)))
        self.arrived.set()


@pytest.mark.asyncio
async def test_service_logging_events_reach_the_loop_from_another_thread() -> None:
    logging: Final = _RecordingServiceLogging()
    events: Final = ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: logging)

    await asyncio.to_thread(
        events.record, "add_key_spend", "LiteLLM_VerificationToken", "update", NOW.timestamp(), 0.25, None
    )
    await asyncio.wait_for(logging.arrived.wait(), timeout=5)
    logging.arrived.clear()
    await asyncio.to_thread(events.record, "health_check", None, None, NOW.timestamp(), 1.5, "pool timed out")
    await asyncio.wait_for(logging.arrived.wait(), timeout=5)

    assert logging.events == (
        (
            "postgres",
            "add_key_spend",
            0.25,
            {"table_name": "LiteLLM_VerificationToken", "db_operation": "update"},
            None,
        ),
        ("postgres", "health_check", 1.5, {}, "pool timed out"),
    )


@pytest.mark.asyncio
async def test_native_health_check_failure_reports_a_db_event() -> None:
    logging: Final = _RecordingServiceLogging()
    native: Final = await NativeDatabase.connect(
        ServiceLoggingEvents(loop=asyncio.get_running_loop(), service_logging=lambda: logging),
        native_endpoint("postgresql://litellm:secret@127.0.0.1:1/proxy?pool_timeout=1"),
    )

    with pytest.raises(ValueError, match="database query failed"):
        await native.health_check()
    await asyncio.wait_for(logging.arrived.wait(), timeout=5)
    await native.close()

    assert len(logging.events) == 1
    service, call_type, duration, metadata, failure = logging.events[0]
    assert (service, call_type, metadata) == ("postgres", "health_check", {})
    assert duration >= 0.9
    assert failure is not None and "database query failed" in failure
