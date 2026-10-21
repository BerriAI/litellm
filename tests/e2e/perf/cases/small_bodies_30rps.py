"""The small-bodies case under load: the same callers, guardrails, mocks and 2.5k to 4k-token bodies as
`small_bodies_5rps`, at 30 rps on the same one 4-worker pod. The 5 rps case measures what one request costs
when nothing else is happening; this one measures what the pod does when it is busy, around 60 percent of the
~48 rps the pod sustained on these bodies, so requests overlap in every worker but the queue stays bounded.
A change that only costs under load, a lock held across an await, a connection pool that fills, a flush that
stalls the loop while 40 requests are in flight, shows up here as added latency, non-200s or RSS and not in
the 5 rps case."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from cases.small_bodies_5rps import CASE as SMALL

CASE: Final = replace(
    SMALL,
    name="small_bodies_30rps",
    traffic=replace(SMALL.traffic, seed=13, rps=30.0, total_requests=18_000, concurrency=150),
    limits=(),
)
