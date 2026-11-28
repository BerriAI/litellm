"""The live worker census in `LiteLLM_ProxyWorkerHeartbeat` and the "no Redis" warning built on it.

Contract (litellm/proxy/db/proxy_worker_heartbeat.py, `_show_no_redis_warning` in
litellm/proxy/health_endpoints/_health_endpoints.py):

* Every worker process upserts one row (`worker_id` a fresh uuid, `hostname` the machine name) at startup and
  every 60 seconds, and deletes rows whose last heartbeat is older than 3600 seconds.
* A row counts as live while its heartbeat is younger than 180 seconds.
* `GET /health/readiness/details` reports `show_no_redis_warning: true` only when no Redis is configured, the
  `LITELLM_DISABLE_NO_REDIS_WARNING` switch is off, and the census does not prove exactly one live worker.

This is how the admin UI tells an operator that rate limits and budgets are per worker. When `num_workers`
moves into a Rust process model, the census must still count the processes that hold their own in-memory
limits, or the warning lies in both directions.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import psycopg
import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.process import OwnedProxy, owned_proxy_process, setup_only_proxy_run
from tests.integration.configuration.dec1_deploy_support import (
    NO_REDIS,
    await_rows,
    deploy_config,
    heartbeat_rows,
    provider,
    write_config,
)

pytestmark: Final = pytest.mark.timeout(900)

WARNING: Final = "show_no_redis_warning"
HEARTBEATS: Final = 'SELECT worker_id, hostname FROM "LiteLLM_ProxyWorkerHeartbeat" ORDER BY worker_id'


def _warning(owned: OwnedProxy) -> object:
    return owned.gateway.get("/health/readiness/details")[WARNING]


@contextmanager
def _census_proxy(
    gateway: Gateway, tmp_path: Path, *, workers: int, environment: dict[str, str] | None = None, redis: bool = False
) -> Iterator[OwnedProxy]:
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            environment or {},
            config=config,
            workers=workers,
            remove_environment=() if redis else NO_REDIS,
        ) as owned:
            yield owned


def _seed_heartbeat(database: str, worker_id: str, age_seconds: int) -> None:
    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_ProxyWorkerHeartbeat" (worker_id, hostname, last_heartbeat_at) '
            "VALUES (%s, %s, NOW() - make_interval(secs => %s))",
            (worker_id, "dec1-deploy-gone-pod", age_seconds),
        )


@contextmanager
def _seeded_database(gateway: Gateway, tmp_path: Path, age_seconds: int) -> Iterator[tuple[str, str]]:
    """A scratch database prepared by a setup-only run, holding one foreign heartbeat row of the given age."""
    with scratch_database("dec1_deploy") as database, provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base), "dec1-deploy-setup.yaml")
        assert setup_only_proxy_run(gateway, {"DATABASE_URL": database}, config=config, workers=1).returncode == 0
        foreign: Final = f"dec1-deploy-gone-{uuid.uuid4().hex}"
        _seed_heartbeat(database, foreign, age_seconds)
        yield database, foreign


@contextmanager
def _boot_on(gateway: Gateway, tmp_path: Path, database: str) -> Iterator[OwnedProxy]:
    with provider() as upstream:
        config: Final = write_config(tmp_path, deploy_config(upstream.api_base))
        with owned_proxy_process(
            gateway,
            tmp_path / "proxy",
            {"DATABASE_URL": database, "DISABLE_SCHEMA_UPDATE": "true"},
            config=config,
            remove_environment=("DATABASE_URL_READ_REPLICA", *NO_REDIS),
        ) as owned:
            yield owned


def test_each_worker_registers_one_heartbeat_row_with_this_hostname(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy with `--num_workers 2` must come to hold exactly two rows in `LiteLLM_ProxyWorkerHeartbeat`, with
    distinct `worker_id`s and `hostname` equal to this machine's `socket.gethostname()`: one row per process,
    not one per pod."""
    with _census_proxy(gateway, tmp_path, workers=2, redis=True) as owned:
        rows: Final = await_rows(lambda: heartbeat_rows(owned.gateway), 2)
    assert len({row["worker_id"] for row in rows}) == 2, rows
    assert {row["hostname"] for row in rows} == {socket.gethostname()}, rows


def test_two_workers_without_redis_show_the_no_redis_warning(gateway: Gateway, tmp_path: Path) -> None:
    """Two workers, no `REDIS_*` variable and no Redis in the config. Once both heartbeat rows exist,
    `GET /health/readiness/details` must report `show_no_redis_warning: true`: limits are per worker."""
    with _census_proxy(gateway, tmp_path, workers=2) as owned:
        await_rows(lambda: heartbeat_rows(owned.gateway), 2)
        assert eventually(lambda: _warning(owned), lambda shown: shown is True, seconds=30) is True


def test_one_worker_without_redis_shows_no_warning(gateway: Gateway, tmp_path: Path) -> None:
    """One worker and no Redis. `show_no_redis_warning` must be false: a census of exactly one live worker
    proves per-worker state is global (`no_redis_on_multiple_workers`, _health_endpoints.py:2117)."""
    with _census_proxy(gateway, tmp_path, workers=1) as owned:
        await_rows(lambda: heartbeat_rows(owned.gateway), 1)
        assert _warning(owned) is False


def test_the_disable_switch_hides_the_warning_for_two_workers_without_redis(gateway: Gateway, tmp_path: Path) -> None:
    """Two workers without Redis and `LITELLM_DISABLE_NO_REDIS_WARNING=true`. `show_no_redis_warning` must be
    false: operators who accept per-worker limits can silence the banner."""
    with _census_proxy(gateway, tmp_path, workers=2, environment={"LITELLM_DISABLE_NO_REDIS_WARNING": "true"}) as owned:
        await_rows(lambda: heartbeat_rows(owned.gateway), 2)
        assert _warning(owned) is False


def test_two_workers_with_redis_from_the_environment_show_no_warning(gateway: Gateway, tmp_path: Path) -> None:
    """Two workers with the harness's private Redis given through `REDIS_HOST` and `REDIS_PORT` only, nothing
    in the config. `show_no_redis_warning` must be false: the environment fallback counts as configured Redis.
    UNVERIFIED: that `REDIS_HOST` alone builds the coordination cache (docstring of
    `no_redis_on_multiple_workers`, _health_endpoints.py:2117)."""
    with _census_proxy(gateway, tmp_path, workers=2, redis=True) as owned:
        await_rows(lambda: heartbeat_rows(owned.gateway), 2)
        assert _warning(owned) is False


def test_a_recently_killed_pods_row_still_counts_as_a_live_worker(gateway: Gateway, tmp_path: Path) -> None:
    """The database already holds a heartbeat row from another pod, 60 seconds old, as a pod killed a minute ago
    would leave. A single worker proxy without Redis boots on it. `show_no_redis_warning` must be true: the
    census counts two workers inside the 180 second liveness window, so the proxy cannot prove it is alone."""
    with (
        _seeded_database(gateway, tmp_path, 60) as (database, _foreign),
        _boot_on(gateway, tmp_path, database) as owned,
    ):
        await_rows(lambda: read_rows(HEARTBEATS, (), database_url=database), 2)
        assert _warning(owned) is True


def test_a_row_older_than_the_liveness_window_is_not_counted(gateway: Gateway, tmp_path: Path) -> None:
    """The database holds another pod's heartbeat row 600 seconds old. A single worker proxy without Redis boots
    on it. `show_no_redis_warning` must be false (the row is outside the 180 second window) and the row must
    still exist (it is younger than the 3600 second pruning age)."""
    with (
        _seeded_database(gateway, tmp_path, 600) as (database, foreign),
        _boot_on(gateway, tmp_path, database) as owned,
    ):
        await_rows(lambda: read_rows(HEARTBEATS, (), database_url=database), 2)
        assert _warning(owned) is False
        assert foreign in {row["worker_id"] for row in read_rows(HEARTBEATS, (), database_url=database)}


def test_a_row_older_than_an_hour_is_pruned_by_the_first_heartbeat(gateway: Gateway, tmp_path: Path) -> None:
    """The database holds another pod's heartbeat row two hours old. Once a proxy has booted on it, the table
    must hold only the new worker's row: every beat deletes rows older than 3600 seconds
    (`PRUNE_SQL`, proxy_worker_heartbeat.py:39), and the first beat runs at startup."""
    with _seeded_database(gateway, tmp_path, 7200) as (database, foreign), _boot_on(gateway, tmp_path, database):
        rows: Final = await_rows(lambda: read_rows(HEARTBEATS, (), database_url=database), 1)
    assert foreign not in {row["worker_id"] for row in rows}, rows
