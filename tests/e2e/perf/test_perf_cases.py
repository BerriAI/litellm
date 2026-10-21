"""One perf run per `PerformanceTestCase` in `cases/`: check the gateway was deployed with the case's
model groups and guardrails, build its teams, users and keys, replay its deterministic plan from a separate Locust
process against the out-of-process mock providers, then check every `Limit` against what the
clients saw, what the mocks received and how the workers' memory moved."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final

import pytest
from cases import CASES
from idp import Keycloak
from lifecycle import ResourceManager
from memory_readings import read_rss_everywhere
from models import SpendLogsParams
from proxy_client import ProxyClient
from setup.datastore import connections_sampler, read_db_counters, read_redis_counters, redis_monitor
from setup.event_loop import (
    LOG_FILE_ENV,
    OPT_IN_ENV,
    StallRow,
    offenders,
    parse_stalls,
    read_gateway_log,
    render_stalls,
    summarize,
)
from setup.event_loop import junit_properties as stall_properties
from setup.observe import (
    Observation,
    evaluate,
    junit_properties,
    read_guardrail_calls,
    read_mock_requests,
    render,
    reset_mock,
    set_mock_latency,
)
from setup.performance_test_case import MockLatency, PerformanceTestCase
from setup.plan import PlannedRequest, build_plan
from setup.podcpu import gateway_cpu_sampler
from setup.provision import MOCK_PROVIDERS, build_setup, guardrail_url, mock_url
from setup.ramp import step_properties
from setup.run import run_plan
from setup.telemetry import read_gateway_telemetry, telemetry_sampler

pytestmark = [pytest.mark.perf, pytest.mark.quiet_stack]
_SPEND_SAMPLE: Final = 40
_WARMUP_SECONDS: Final = 60


def _spend_logged_share(proxy: ProxyClient, call_ids: tuple[str, ...]) -> float | None:
    """Share of a sample of served requests whose spend row the gateway wrote, read back by the
    request id the gateway returned in `x-litellm-call-id`."""
    if not call_ids:
        return None
    found = 0
    for call_id in call_ids:
        rows = proxy.spend_logs(SpendLogsParams(request_id=call_id))
        found += 1 if rows else 0
    return found / len(call_ids)


def _warmup_plan(case: PerformanceTestCase, plan: tuple[PlannedRequest, ...]) -> tuple[PlannedRequest, ...]:
    """The plan's first minute of traffic. It is sent before the memory baseline is read so a worker
    has shed its startup allocations and `memory_growth_megabytes` is what the measured traffic added,
    not startup churn (a lightly loaded case read negative growth)."""
    return plan[: max(1, round(case.traffic.rps * _WARMUP_SECONDS))]


def _arm_mocks(mock_urls: Mapping[str, str], latency: MockLatency) -> None:
    for url in (*mock_urls.values(), guardrail_url()):
        _ = reset_mock(url)
    for url in mock_urls.values():
        _ = set_mock_latency(url, latency)


def _event_loop_rows(case: PerformanceTestCase, since: datetime) -> tuple[StallRow, ...]:
    """The gateway's asyncio debug holds for the traffic window; a case that asks for them cannot pass without a log."""
    assert case.event_loop is not None
    lines: Final = read_gateway_log(since)
    assert lines is not None, (
        f"{case.name} needs the gateway log: run in the perf lane (pods/log with the runner service account) or set "
        f"{LOG_FILE_ENV} to the gateway's log file"
    )
    return summarize(case.event_loop, parse_stalls(lines))


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_perf_case(
    case: PerformanceTestCase,
    proxy: ProxyClient,
    idp: Keycloak,
    resources: ResourceManager,
    request: pytest.FixtureRequest,
) -> None:
    if case.event_loop is not None and not os.environ.get(OPT_IN_ENV):
        pytest.skip(
            f"{case.name} reads asyncio debug output: deploy the gateway with PYTHONASYNCIODEBUG=1 and "
            f"LITELLM_EVENT_LOOP=asyncio, then set {OPT_IN_ENV}=1"
        )
    plan: Final = build_plan(case)
    mock_urls: Final = {provider: mock_url(provider) for provider in MOCK_PROVIDERS}
    _arm_mocks(mock_urls, case.mock_latency)
    setup_started: Final = time.monotonic()
    setup: Final = build_setup(case, proxy=proxy, idp=idp, resources=resources)
    _ = run_plan(case, _warmup_plan(case, plan), setup)
    _arm_mocks(mock_urls, case.mock_latency)
    rss_before: Final = read_rss_everywhere(proxy)
    telemetry_before: Final = read_gateway_telemetry(proxy)
    db_before: Final = read_db_counters()
    redis_before: Final = read_redis_counters()
    run_started: Final = time.monotonic()
    traffic_started_at: Final = datetime.now(tz=UTC)
    with (
        gateway_cpu_sampler() as cpu,
        redis_monitor() as monitor,
        connections_sampler() as connections,
        telemetry_sampler(proxy) as telemetry,
    ):
        outcome: Final = run_plan(case, plan, setup)
    run_finished: Final = time.monotonic()
    db_after: Final = read_db_counters()
    redis_after: Final = read_redis_counters()
    rss_after: Final = read_rss_everywhere(proxy)
    telemetry_after: Final = read_gateway_telemetry(proxy)
    rss_read: Final = time.monotonic()
    assert outcome.records, f"locust wrote no records (exit {outcome.locust_exit}):\n{outcome.locust_output}"
    served_call_ids: Final = tuple(
        record.call_id for record in outcome.records if record.ok and not record.blocked and record.call_id
    )
    spend_share: Final = _spend_logged_share(
        proxy, served_call_ids[:: max(1, len(served_call_ids) // _SPEND_SAMPLE)][:_SPEND_SAMPLE]
    )
    spend_read: Final = time.monotonic()
    mock_requests: Final = tuple(seen for url in mock_urls.values() for seen in read_mock_requests(url))
    guardrail_calls: Final = tuple(read_guardrail_calls(guardrail_url())) if case.guardrails else ()
    mocks_read: Final = time.monotonic()
    observation: Final = Observation(
        records=outcome.records,
        mock_requests=mock_requests,
        guardrail_calls=guardrail_calls,
        rss_before=rss_before,
        rss_after=rss_after,
        wall_seconds=outcome.wall_seconds,
        spend_logged_share=spend_share,
        db_before=db_before,
        db_after=db_after,
        redis_before=redis_before,
        redis_after=redis_after,
        redis_commands=monitor.commands,
        gateway_cpu_seconds=cpu.seconds,
        telemetry_before=telemetry_before,
        telemetry_after=telemetry_after,
        peak_connections=connections.peak,
        ramp=case.traffic.ramp,
        cpu_series=cpu.series,
        connection_series=connections.series,
        telemetry_series=telemetry.series,
    )
    rows, exceeded_limits = evaluate(case, observation)
    stall_rows: Final = _event_loop_rows(case, traffic_started_at) if case.event_loop is not None else ()
    item: Final = request.node  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # pytest: untyped
    assert isinstance(item, pytest.Item)
    item.user_properties[:] = [
        entry for entry in item.user_properties if not entry[0].startswith("perf_")
    ]  # rebind-ok: a --reruns retry replaces the first attempt's numbers instead of appending a second set
    item.user_properties.extend(
        stall_properties(case.event_loop, stall_rows) if case.event_loop is not None else junit_properties(rows)
    )
    item.user_properties.extend(step_properties(observation.steps()))
    print(render(case, observation, rows))
    if case.event_loop is not None:
        print(render_stalls(case.event_loop, stall_rows))
    print(
        f"  phases: setup {run_started - setup_started:.0f}s, locust {run_finished - run_started:.0f}s, "
        f"observe {time.monotonic() - run_finished:.0f}s (rss {rss_read - run_finished:.0f}s, spend sample "
        f"{spend_read - rss_read:.0f}s, mock logs {mocks_read - spend_read:.0f}s); locust exit {outcome.locust_exit}"
    )
    assert len(outcome.records) == len(plan), (
        f"{len(plan)} planned, {len(outcome.records)} recorded; locust said:\n{outcome.locust_output}"
    )
    assert not exceeded_limits, "limits exceeded (observed > max):\n" + "\n".join(exceeded_limits)
    assert not offenders(stall_rows), "event-loop holds over the threshold:\n" + "\n".join(offenders(stall_rows))
