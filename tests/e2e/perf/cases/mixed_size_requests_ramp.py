"""The mixed-size traffic stepped up until the pod stops keeping up: the same callers, guardrails, mocks and body
mix (short chats, 300k to 1M-token contexts, tool calls, half streamed) as `mixed_size_requests_2rps`, 2 rps more
every 90 s from 2 to 14 on the same one 4-worker pod. Where the 2 rps case prices one big body on its own, this one
reports `sustained_rps`, the last step the pod held with big bodies overlapping in every worker, so a handler that
re-walks a 1M-token body on the event loop or a worker that holds several of them in memory at once moves the knee
down and fails the floor."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from setup.performance_test_case import Limit, Ramp, ramped

from cases.mixed_size_requests_2rps import CASE as MIXED

RAMP: Final = Ramp(first_rps=2.0, last_rps=14.0, step_rps=2.0, step_seconds=90.0, held_p99_seconds=5.0)

CASE: Final = replace(
    MIXED,
    name="mixed_size_requests_ramp",
    traffic=ramped(MIXED.traffic, RAMP, seed=9, concurrency=300),
    limits=(Limit("sustained_rps", min=8),),
)
