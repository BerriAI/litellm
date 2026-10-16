from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Final

import pytest
from idp import Keycloak
from lifecycle import ResourceManager
from memory_readings import read_rss_everywhere
from proxy_client import ProxyClient
from setup.datastore import read_db_counters, read_redis_counters
from setup.observe import Observation, evaluate, read_mock_requests, render, reset_mock, set_mock_latency
from setup.plan import build_plan
from setup.podcpu import gateway_cpu_sampler
from setup.provision import MOCK_PROVIDERS, build_setup, mock_url
from setup.run import run_plan
from setup.upgrade import assert_finished, monitor_for, upgrade_case
from setup.upgrade_worker import connect, read_state

pytestmark = [pytest.mark.perf, pytest.mark.quiet_stack]


@pytest.mark.skipif(os.environ.get("E2E_DATABASE_UPGRADE") != "1", reason="Dedicated disposable snapshot lane only")
def test_database_upgrade(proxy: ProxyClient, idp: Keycloak, resources: ResourceManager) -> None:
    case: Final = upgrade_case()
    urls: Final = tuple(mock_url(provider) for provider in MOCK_PROVIDERS)
    for url in urls:
        _ = reset_mock(url)
        _ = set_mock_latency(url, case.mock_latency)
    setup: Final = build_setup(case, proxy=proxy, idp=idp, resources=resources)
    metrics: Final = tuple(url.strip() for url in os.environ["E2E_DATABASE_METRICS_URLS"].split(",") if url.strip())
    with connect() as connection:
        assert read_state(connection).phase == "schema_ready", "Expected the target schema ready, index work waiting"
        rss_before: Final = read_rss_everywhere(proxy)
        db_before: Final = read_db_counters()
        redis_before: Final = read_redis_counters()
        monitor: Final = monitor_for(connection, metrics)
        with gateway_cpu_sampler() as cpu:
            outcome: Final = run_plan(case, build_plan(case), setup, stop_when=monitor.poll, request_timeout=30)
        assert outcome.locust_exit == 0, outcome.locust_output
        remaining: Final = outcome.records[monitor.successes :]
        _ = monitor.poll(remaining)
        deadline: Final = time.monotonic() + 120
        while monitor.pending and time.monotonic() < deadline:
            time.sleep(1)
            _ = monitor.poll(())
        state: Final = read_state(connection)
        monitor.assert_complete(state)
        assert_finished(connection, state)
        rss_after: Final = read_rss_everywhere(proxy)
        assert {r.worker for r in rss_before.readings} == {r.worker for r in rss_after.readings}, (
            "Gateway workers restarted"
        )
        observed: Final = Observation(
            records=outcome.records,
            mock_requests=tuple(seen for url in urls for seen in read_mock_requests(url)),
            guardrail_calls=(),
            rss_before=rss_before,
            rss_after=rss_after,
            wall_seconds=outcome.wall_seconds,
            spend_logged_share=1,
            db_before=db_before,
            db_after=read_db_counters(),
            redis_before=redis_before,
            redis_after=read_redis_counters(),
            gateway_cpu_seconds=cpu.seconds,
        )
        rows, exceeded = evaluate(case, observed)
        _ = sys.stdout.write(render(case, observed, rows) + "\n")
        _ = sys.stdout.write(
            f"schema_seconds={state.schema_seconds:.3f} index_seconds={state.index_seconds:.3f} "
            f"overlap_writes={monitor.overlap_writes} max_lock_waiters={monitor.max_lock_waiters} "
            f"pending_indexes={state.pending_indexes}\n"
        )
        artifact: Final = os.environ.get("E2E_ARTIFACT_DIR")
        if artifact:
            _ = (Path(artifact) / "database-upgrade.json").write_text(state.model_dump_json(), encoding="utf-8")
        assert not exceeded, "\n".join(exceeded)
