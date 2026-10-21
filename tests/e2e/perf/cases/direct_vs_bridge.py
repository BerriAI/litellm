"""The mixed case's traffic on one public API, half of it on a deployment the gateway serves on
that provider's own API (direct) and half on a deployment it has to translate to another provider
API (bridge). Same 2 rps, request count, shapes, stream share, callers and guardrails as
`mixed_size_requests_2rps`, so a gap between the two halves and the mixed baseline is the
translation's cost. Each case carries its own limits, 1.2x the median of ten lane runs on every
stable tag from v1.103.0 to v1.106.0, the same rule as the other perf cases."""

from __future__ import annotations

from dataclasses import replace

from setup.performance_test_case import Endpoint, Limit, PerformanceTestCase, VirtualKey

from .mixed_size_requests_2rps import CASE as MIXED


def _scoped(key: VirtualKey, direct: str) -> VirtualKey:
    return replace(key, models=(direct,)) if key.models else key


def direct_vs_bridge(
    name: str, endpoint: Endpoint, *, direct: str, bridge: str, limits: tuple[Limit, ...]
) -> PerformanceTestCase:
    return replace(
        MIXED,
        name=name,
        teams=(
            replace(MIXED.team("team-a"), models=(direct, bridge)),
            replace(MIXED.team("team-b"), models=(bridge,)),
        ),
        virtual_keys=tuple(_scoped(key, direct) for key in MIXED.virtual_keys),
        model_groups=(direct, bridge),
        traffic=replace(MIXED.traffic, endpoints={endpoint: 1.0}),
        limits=limits,
    )
