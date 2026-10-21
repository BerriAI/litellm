"""The mixed-size case at the frontier: the same callers, guardrails, mocks and body mix (short chats, 300k to
1M-token contexts, tool calls, half streamed) as `mixed_size_requests_2rps`, at 5 rps on the same one 4-worker
pod, which is about where the pod stops keeping up with these bodies. Where the 2 rps case prices one big body
on its own, this one shows what happens when big bodies overlap in every worker: a handler that re-walks a
1M-token body on the event loop, a worker that holds several of them in memory at once, or a flush that waits
behind them."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from cases.mixed_size_requests_2rps import CASE as MIXED

CASE: Final = replace(
    MIXED,
    name="mixed_size_requests_5rps",
    traffic=replace(MIXED.traffic, seed=9, rps=5.0, total_requests=3_000, concurrency=300),
    limits=(),
)
