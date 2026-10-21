"""The small-bodies case at the frontier: the same callers, guardrails, mocks and 2.5k to 4k-token bodies as
`small_bodies_5rps`, at 50 rps on the same one 4-worker pod, which is about as much as the pod sustained on
these bodies (~48 rps). The 5 rps case measures what one request costs when nothing else is happening; this one
measures what the pod does when every worker is full and the queue is as deep as it gets before it stops
draining. A change that only costs under load, a lock held across an await, a connection pool that fills, a
flush that stalls the loop while dozens of requests are in flight, shows up here as added latency, non-200s or
RSS and not in the 5 rps case."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from cases.small_bodies_5rps import CASE as SMALL

CASE: Final = replace(
    SMALL,
    name="small_bodies_50rps",
    traffic=replace(SMALL.traffic, seed=13, rps=50.0, total_requests=30_000, concurrency=400),
    limits=(),
)
