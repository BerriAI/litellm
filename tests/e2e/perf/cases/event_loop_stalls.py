"""Where the gateway holds its event loop for 100 ms or more: the mixed case's callers, model groups, guardrails
and body mix (short chats, 300k to 1M-token contexts, tool calls, half streamed) at 1 rps for 120 requests against
a gateway running the stdlib asyncio loop in debug mode, which logs every loop step over 0.1 s with the task that
ran it. No latency, CPU or datastore ceilings: the one check is that no step over 0.1 s comes from a location that
is not on the allowlist below, and that the listed ones do not show up more often than they do today. A stream
that stops yielding (the 46925 fix), a provider handler that re-walks a 1M-token body on the loop (46967), or a
new hook that deep-copies the request inline all land here as a new row. Runs on its own stack: the gateway needs
`PYTHONASYNCIODEBUG=1` and `LITELLM_EVENT_LOOP=asyncio` (uvloop reports nothing and crashed under debug locally),
and the runner `E2E_PERF_EVENT_LOOP=1`, so the nightly build, which sets neither, skips it. The allowlist counts
are twice the highest count any of ten back-to-back local runs showed on that row (rows that showed nothing in
those ten but did in the earlier before/after legs for 46925 and 46967, or in the first scheduled-lane run, get 2),
so each run sits at or under half of every cap and a listed location doubling in frequency is a failure, not a flake. The
`garbage collection` bucket is the gen-2 passes the collector logs in debug mode (0.6 to 0.7 s each, one every
minute or two of this traffic, plus the ramp while the heap fills on a local whole-file run), which asyncio would
otherwise charge to a random task; its count is the signal for a change that makes much more garbage."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from setup.event_loop import EventLoopStalls, KnownStall

from cases.mixed_size_requests_2rps import CASE as MIXED

CASE: Final = replace(
    MIXED,
    name="event_loop_stalls",
    traffic=replace(MIXED.traffic, rps=1.0, total_requests=120, concurrency=10),
    limits=(),
    event_loop=EventLoopStalls(
        allowlist=(
            KnownStall("garbage collection", max_count=4),
            KnownStall("LifespanOn.main", max_count=4),
            KnownStall("RequestResponseCycle.run_asgi", max_count=2),
            KnownStall("with_proxy_compaction_executor", max_count=2),
            KnownStall("Logging.async_success_handler", max_count=2),
            KnownStall("Logging.dispatch_success_handlers", max_count=2),
            KnownStall("_TrackedPrismaEngine", max_count=2),
            KnownStall("prisma/actions.py", max_count=2),
            KnownStall("SpendCounterReseed", max_count=2),
            KnownStall("Subscriber._run", max_count=2),
            KnownStall("RedisBatch.flush", max_count=2),
            KnownStall("run_coroutine_job", max_count=2),
            KnownStall("user_api_key_auth.py", max_count=2),
            KnownStall("route_streaming_logging_to_handler", max_count=2),
            KnownStall("TaskHandle._run_coro", max_count=2),
        ),
    ),
)
