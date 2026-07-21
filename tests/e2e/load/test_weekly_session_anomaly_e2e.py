from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from e2e_config import (
    ANOMALY_MAX_ERROR_RATIO,
    ANOMALY_MAX_KEY_SPEND_USD,
    ANOMALY_MAX_P95_TURN_SECONDS,
    ANOMALY_MIN_WARM_CACHE_READ_SHARE,
    ANOMALY_SESSIONS,
    ANOMALY_TURNS_PER_SESSION,
    unique_marker,
)
from lifecycle import ResourceManager
from load_client import LoadClient
from models import KeyGenerateBody, LiteLLMParamsBody
from proxy_client import ProxyClient
from session_anomaly import run_concurrent_sessions, summarize

pytestmark = [pytest.mark.e2e, pytest.mark.load, pytest.mark.weekly]


@dataclass(frozen=True, slots=True)
class AnomalyRoute:
    route_id: str
    params: LiteLLMParamsBody


ANOMALY_ROUTES = (
    AnomalyRoute(
        route_id="anthropic",
        params=LiteLLMParamsBody(model="anthropic/claude-sonnet-5"),
    ),
    AnomalyRoute(
        route_id="bedrock_invoke",
        params=LiteLLMParamsBody(
            model="bedrock/invoke/us.anthropic.claude-sonnet-5",
            aws_region_name="us-east-1",
        ),
    ),
)


def _route_id(route: AnomalyRoute) -> str:
    return route.route_id


def _settled_key_spend(proxy: ProxyClient, key: str) -> float:
    deadline = time.monotonic() + proxy.poll_timeout

    def settle(previous: float) -> float:
        current = proxy.key_info(key).spend or 0.0
        if current > 0 and current == previous:
            return current
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"key spend never settled to a stable non-zero value within "
                f"{proxy.poll_timeout}s (last read {current}); spend stopped being "
                f"recorded, which is itself a spend anomaly"
            )
        time.sleep(proxy.poll_interval)
        return settle(current)

    return settle(-1.0)


class TestWeeklySessionAnomaly:
    @pytest.mark.covers("reliability.perf.session_anomaly.under_slo")
    @pytest.mark.parametrize("route", ANOMALY_ROUTES, ids=_route_id)
    def test_session_load_stays_within_baselines(
        self, client: LoadClient, resources: ResourceManager, route: AnomalyRoute
    ) -> None:
        model_name = f"weekly-anomaly-{route.route_id}-{unique_marker()}"
        model_id = client.proxy.create_model(model_name, route.params)
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = client.proxy.generate_key(
            KeyGenerateBody(models=[model_name], key_alias=model_name)
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        turns = run_concurrent_sessions(
            client.proxy.transport,
            key,
            model_name,
            ANOMALY_SESSIONS,
            ANOMALY_TURNS_PER_SESSION,
        )
        report = summarize(turns)
        failures = tuple(turn.failure for turn in turns if turn.failure)
        print(f"{route.route_id} anomaly report: {report}")

        assert report.error_ratio <= ANOMALY_MAX_ERROR_RATIO, (
            f"{route.route_id}: {report.failed_turns}/{report.attempted_turns} turns "
            f"failed ({report.error_ratio:.1%} > {ANOMALY_MAX_ERROR_RATIO:.1%} allowed); "
            f"error rate is anomalously high. Failures: {failures}"
        )
        assert report.warm_turns > 0, (
            f"{route.route_id}: no session got past its first turn, so cache and "
            f"latency baselines have nothing to read. Failures: {failures}"
        )
        assert report.warm_cache_read_share >= ANOMALY_MIN_WARM_CACHE_READ_SHARE, (
            f"{route.route_id}: warm turns read only {report.warm_cache_read_share:.1%} "
            f"of billed input tokens from the prompt cache "
            f"(read={report.warm_cache_read_tokens}, "
            f"creation={report.warm_cache_creation_tokens}, "
            f"uncached={report.warm_uncached_input_tokens}), below the "
            f"{ANOMALY_MIN_WARM_CACHE_READ_SHARE:.0%} floor; the cached prefix is "
            f"being invalidated between turns (the mid-conversation-system cache "
            f"collapse signature) or caching stopped working"
        )
        assert report.warm_cache_creation_tokens > 0, (
            f"{route.route_id}: warm turns wrote 0 cache-creation tokens across "
            f"{report.warm_turns} turns; the moving cache breakpoint stopped writing "
            f"new prefix increments"
        )
        assert report.p95_turn_seconds <= ANOMALY_MAX_P95_TURN_SECONDS, (
            f"{route.route_id}: p95 turn time {report.p95_turn_seconds:.1f}s exceeds "
            f"the {ANOMALY_MAX_P95_TURN_SECONDS:.0f}s ceiling under "
            f"{ANOMALY_SESSIONS} concurrent sessions; turn times are anomalously slow"
        )

        spend = _settled_key_spend(client.proxy, key)
        assert spend <= ANOMALY_MAX_KEY_SPEND_USD, (
            f"{route.route_id}: gateway recorded ${spend:.4f} for "
            f"{report.attempted_turns} turns, above the "
            f"${ANOMALY_MAX_KEY_SPEND_USD} ceiling; spend per session is "
            f"anomalously high (cache regressions surface here as 2-3x spend)"
        )
