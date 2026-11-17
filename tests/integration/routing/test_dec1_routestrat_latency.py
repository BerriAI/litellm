"""`routing_strategy: latency-based-routing` across providers: warm-up, per-token latency, buffer, ttl, limits, timeouts.

The pick is answered by `LowestLatencyLoggingHandler._get_available_deployments` in
`litellm/router_strategy/lowest_latency.py:409`. On every success it appends the response time divided by the
answer's completion tokens (seconds per output token, line 148) to a list of at most `max_latency_list_size`
(10) samples per deployment, stored under `<group>_map` with `routing_strategy_args.ttl` (default 3600 seconds).
A deployment with no samples yet starts at `[0]`, so it is tried before any measured one. The pick sorts by the
average sample, keeps every deployment within `lowest_latency_buffer` times the lowest average (default 0), and
chooses at random among them. A deployment whose `rpm` or `tpm` is used up this minute is skipped (line 513). A
`litellm.Timeout` appends a penalty of 1000 seconds (line 255).

Deployments here are on OpenAI, Bedrock, Vertex AI and Anthropic. The peer delays a label's answer to make it slow.
OpenAI answers carry 3 completion tokens, the other providers 7, which matters only where a test says so.
Retries are off and cooldowns disabled, so only the strategy decides. Each test that needs its own
`routing_strategy_args` boots its own proxy.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually
from tests.integration.routing.dec1_routestrat_support import (
    Rig,
    chat,
    dep,
    early_in_a_minute,
    reported_strategy,
    served,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

BATCH: Final = 8


def _settings(**args: JsonValue) -> dict[str, JsonValue]:
    return {
        "routing_strategy": "latency-based-routing",
        "num_retries": 0,
        "disable_cooldowns": True,
        **({"routing_strategy_args": dict(args)} if args else {}),
    }


def _settles_on(rig: Rig, group: str, label: str, seconds: float = 60) -> tuple[str, ...]:
    return eventually(
        lambda: tuple(served(rig.gateway, group) for _ in range(BATCH)),
        lambda batch: set(batch) == {label},
        seconds=seconds,
    )


def test_every_untried_deployment_is_sampled_before_any_is_measured_twice(tmp_path: Path) -> None:
    """A fresh proxy routes a group of OpenAI, Bedrock and Vertex deployments by latency; none is slow.

    Within the first six sequential requests all three providers must have served: an untried deployment counts
    as latency 0 and beats any measured one. A rewrite that started untried deployments at infinity would stick
    to whichever served first.
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("openai", "lat-warm", "lw-openai", base),
            dep("bedrock", "lat-warm", "lw-bedrock", base),
            dep("vertex", "lat-warm", "lw-vertex", base),
        )

    with strat_rig(tmp_path, models, _settings()) as rig:
        assert reported_strategy(rig.gateway) == "latency-based-routing"
        first: Final = [served(rig.gateway, "lat-warm") for _ in range(6)]
        assert set(first) == {"lw-openai", "lw-bedrock", "lw-vertex"}, first


def test_traffic_settles_on_the_fastest_provider_after_warm_up(tmp_path: Path) -> None:
    """OpenAI answers at once; Bedrock waits 0.5 seconds and Anthropic 0.8 seconds before answering.

    After a warm-up of nine requests, batches of eight sequential requests are sent until one whole batch is
    served by OpenAI, which must happen within 60 seconds. Both slow providers must have been sampled at least
    once on the way.
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("openai", "lat-fast", "lf-openai", base),
            dep("bedrock", "lat-fast", "lf-bedrock", base),
            dep("anthropic", "lat-fast", "lf-anthropic", base),
        )

    with strat_rig(tmp_path, models, _settings()) as rig:
        rig.peer.delay("lf-bedrock", 0.5)
        rig.peer.delay("lf-anthropic", 0.8)
        for _ in range(9):
            served(rig.gateway, "lat-fast")
        _settles_on(rig, "lat-fast", "lf-openai")
        assert rig.peer.count("lf-bedrock") >= 1
        assert rig.peer.count("lf-anthropic") >= 1


def test_latency_is_measured_per_completion_token_so_a_slower_but_longer_answer_wins(tmp_path: Path) -> None:
    """Two OpenAI deployments: one answers after 0.6 seconds with 200 completion tokens, the other after 0.1
    seconds with 1 completion token.

    Per token the slow one costs 0.003 seconds and the fast one 0.1 seconds, so after warm-up whole batches of
    eight must be served by the slow, long-answer deployment within 60 seconds. A rewrite that compared raw
    response times would settle on the other one.
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("openai", "lat-per-token", "lpt-long", base),
            dep("openai", "lat-per-token", "lpt-short", base),
        )

    with strat_rig(tmp_path, models, _settings()) as rig:
        rig.peer.delay("lpt-long", 0.6)
        rig.peer.completion_tokens("lpt-long", 200)
        rig.peer.delay("lpt-short", 0.1)
        rig.peer.completion_tokens("lpt-short", 1)
        for _ in range(6):
            served(rig.gateway, "lat-per-token")
        assert _settles_on(rig, "lat-per-token", "lpt-long") == ("lpt-long",) * BATCH
        assert rig.peer.count("lpt-short") >= 1


def test_a_wide_lowest_latency_buffer_keeps_both_providers_in_rotation(tmp_path: Path) -> None:
    """`routing_strategy_args: {lowest_latency_buffer: 1000}`; OpenAI answers at once and Vertex after 0.3 seconds.

    Every deployment within 1000 times the lowest average is a candidate, so after a warm-up of eight requests
    the next forty must include at least five from each provider (expected about twenty each, a random choice
    among both).
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("openai", "lat-buffer", "lb-openai", base),
            dep("vertex", "lat-buffer", "lb-vertex", base),
        )

    with strat_rig(tmp_path, models, _settings(lowest_latency_buffer=1000)) as rig:
        rig.peer.delay("lb-vertex", 0.3)
        for _ in range(8):
            served(rig.gateway, "lat-buffer")
        counts: Final = served_many(rig.gateway, "lat-buffer", 40)
        assert counts["lb-openai"] >= 5, counts
        assert counts["lb-vertex"] >= 5, counts


def test_without_a_buffer_the_slow_provider_stops_receiving_traffic(tmp_path: Path) -> None:
    """The same two deployments as the buffer test with the default `lowest_latency_buffer` of 0.

    Once traffic has settled on OpenAI (a whole batch of eight), the next twenty requests must all be served by
    OpenAI: with no buffer only the lowest average is a candidate.
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("openai", "lat-nobuffer", "lnb-openai", base),
            dep("vertex", "lat-nobuffer", "lnb-vertex", base),
        )

    with strat_rig(tmp_path, models, _settings()) as rig:
        rig.peer.delay("lnb-vertex", 0.3)
        _settles_on(rig, "lat-nobuffer", "lnb-openai")
        assert served_many(rig.gateway, "lat-nobuffer", 20) == {"lnb-openai": 20}


def test_a_short_ttl_forgets_latencies_so_the_slow_provider_is_tried_again(tmp_path: Path) -> None:
    """`routing_strategy_args: {ttl: 2}`; OpenAI answers at once and Bedrock after 0.4 seconds.

    Traffic first settles on OpenAI. The test then lets more than two seconds pass without traffic (polling the
    clock), so the `<group>_map` entry expires and both deployments count as untried again. Repeating that idle
    period and four requests must put a new request on Bedrock within 60 seconds. With the default ttl of an
    hour Bedrock would never be tried again.
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("openai", "lat-ttl", "lt-openai", base),
            dep("bedrock", "lat-ttl", "lt-bedrock", base),
        )

    with strat_rig(tmp_path, models, _settings(ttl=2)) as rig:
        rig.peer.delay("lt-bedrock", 0.4)
        _settles_on(rig, "lat-ttl", "lt-openai")
        settled: Final = rig.peer.count("lt-bedrock")

        def idle_then_send() -> int:
            quiet_until: Final = time.monotonic() + 2.5
            eventually(time.monotonic, lambda now: now >= quiet_until, seconds=5)
            for _ in range(4):
                served(rig.gateway, "lat-ttl")
            return rig.peer.count("lt-bedrock")

        assert eventually(idle_then_send, lambda count: count > settled, seconds=60) > settled


def test_a_fast_deployment_that_used_its_rpm_this_minute_is_skipped(tmp_path: Path) -> None:
    """OpenAI answers at once and has `rpm: 2`; Anthropic answers after 0.3 seconds with no limit.

    Starting early in a minute, ten sequential requests must all return 200, OpenAI must serve at most three of
    them (its limit of two, plus one request that may pass before a late success callback records the usage),
    and Anthropic the rest. Latency routing reads the per-minute counts it keeps itself (`item_rpm + 1 >
    _deployment_rpm`), not the router's.
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("openai", "lat-rpm", "lr-openai", base, rpm=2),
            dep("anthropic", "lat-rpm", "lr-anthropic", base),
        )

    with strat_rig(tmp_path, models, _settings()) as rig:
        rig.peer.delay("lr-anthropic", 0.3)
        early_in_a_minute()
        counts: Final = served_many(rig.gateway, "lat-rpm", 10)
        assert counts["lr-openai"] <= 3, counts
        assert counts["lr-anthropic"] >= 7, counts


def test_a_deployment_that_timed_out_is_penalised_and_not_picked_again(tmp_path: Path) -> None:
    """Vertex has `timeout: 1` and the peer waits 3 seconds before answering it; OpenAI answers at once.

    Sequential requests are sent until one returns 408 from Vertex (within twelve requests, since an untried or
    lightly measured Vertex is picked early). The timeout appends a 1000 second sample to Vertex, so the next
    fifteen requests must all be served by OpenAI.
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("openai", "lat-timeout", "lto-openai", base),
            dep("vertex", "lat-timeout", "lto-vertex", base, timeout=1),
        )

    with strat_rig(tmp_path, models, _settings()) as rig:
        rig.peer.delay("lto-vertex", 3)
        statuses: Final = Counter(chat(rig.gateway, "lat-timeout").status_code for _ in range(12))
        assert statuses[408] >= 1, statuses
        assert set(statuses) <= {200, 408}, statuses
        eventually(
            lambda: tuple(served(rig.gateway, "lat-timeout") for _ in range(BATCH)),
            lambda batch: set(batch) == {"lto-openai"},
            seconds=30,
        )
        assert served_many(rig.gateway, "lat-timeout", 15) == {"lto-openai": 15}
