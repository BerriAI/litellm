"""One perf run per `PerformanceTestCase` in `cases/`: check the gateway was deployed with the case's
model groups and guardrails, build its teams, users and keys, replay its deterministic plan from a separate Locust
process against the out-of-process mock providers, then check every `Limit` against what the
clients saw, what the mocks received and how the workers' memory moved."""

from __future__ import annotations

import time
from typing import Final

import pytest
from idp import Keycloak
from lifecycle import ResourceManager
from memory_readings import read_rss_everywhere
from models import SpendLogsParams
from cases import CASES
from setup.observe import (
    Observation,
    evaluate,
    read_guardrail_calls,
    read_mock_requests,
    render,
    reset_mock,
)
from setup.performance_test_case import PerformanceTestCase
from setup.plan import build_plan
from setup.provision import build_setup, guardrail_url, mock_url
from setup.run import run_plan
from proxy_client import ProxyClient

pytestmark = [pytest.mark.perf, pytest.mark.quiet_stack]
_SPEND_SAMPLE: Final = 40


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


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_perf_case(case: PerformanceTestCase, proxy: ProxyClient, idp: Keycloak, resources: ResourceManager) -> None:
    plan: Final = build_plan(case)
    mock_urls: Final = {provider: mock_url(provider) for provider in ("openai", "anthropic", "bedrock")}
    for url in (*mock_urls.values(), guardrail_url()):
        _ = reset_mock(url)
    setup_started: Final = time.monotonic()
    setup: Final = build_setup(case, proxy=proxy, idp=idp, resources=resources)
    rss_before: Final = read_rss_everywhere(proxy)
    run_started: Final = time.monotonic()
    outcome: Final = run_plan(case, plan, setup)
    run_finished: Final = time.monotonic()
    rss_after: Final = read_rss_everywhere(proxy)
    assert outcome.records, f"locust wrote no records (exit {outcome.locust_exit}):\n{outcome.locust_output}"
    served_call_ids: Final = tuple(
        record.call_id for record in outcome.records if record.ok and not record.blocked and record.call_id
    )
    spend_share: Final = _spend_logged_share(
        proxy, served_call_ids[:: max(1, len(served_call_ids) // _SPEND_SAMPLE)][:_SPEND_SAMPLE]
    )
    observation: Final = Observation(
        records=outcome.records,
        mock_requests=tuple(seen for url in mock_urls.values() for seen in read_mock_requests(url)),
        guardrail_calls=tuple(read_guardrail_calls(guardrail_url())) if case.guardrails else (),
        rss_before=rss_before,
        rss_after=rss_after,
        wall_seconds=outcome.wall_seconds,
        extra={} if spend_share is None else {"spend_logged_share": spend_share},
    )
    rows, exceeded_limits = evaluate(case, observation)
    print(render(case, observation, rows))
    print(
        f"  phases: setup {run_started - setup_started:.0f}s, locust {run_finished - run_started:.0f}s, "
        f"observe {time.monotonic() - run_finished:.0f}s; locust exit {outcome.locust_exit}"
    )
    assert len(outcome.records) == len(plan), (
        f"{len(plan)} planned, {len(outcome.records)} recorded; locust said:\n{outcome.locust_output}"
    )
    assert not exceeded_limits, "limits exceeded (observed > max):\n" + "\n".join(exceeded_limits)
