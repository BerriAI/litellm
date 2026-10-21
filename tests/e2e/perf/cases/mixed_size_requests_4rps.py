"""The mixed-size case under load: the same callers, guardrails, mocks and body mix (short chats, 300k to
1M-token contexts, tool calls, half streamed) as `mixed_size_requests_2rps`, at 4 rps on the same one 4-worker
pod. At 5 rps these bodies saturated the pod (p50 in seconds, p99 over 30 s), so 4 rps is the busiest the pod
gets with the queue still bounded, about two thirds of its CPU. Where the 2 rps case prices one big body on
its own, this one shows what happens when big bodies overlap: a handler that re-walks a 1M-token body on the
event loop, a worker that holds several of them in memory at once, or a flush that waits behind them."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from cases.mixed_size_requests_2rps import CASE as MIXED

CASE: Final = replace(
    MIXED,
    name="mixed_size_requests_4rps",
    traffic=replace(MIXED.traffic, seed=9, rps=4.0, total_requests=2_400, concurrency=100),
    limits=(),
)
