"""The small-bodies traffic stepped up until the pod stops keeping up: the same callers, guardrails, mocks and
2.5k to 4k-token bodies as `small_bodies_5rps`, 10 rps more every 90 s from 10 to 70 on the same one 4-worker pod.
The 5 rps case prices one request with the pod idle; this one reports `sustained_rps`, the last step the pod
held (gateway added p99 under 5 s, the queue still draining), so a change that moves the knee down, a lock held
across an await, a pool that fills, a flush that stalls the loop while dozens of requests are in flight, fails the
floor instead of hiding under a fixed rate the pod still manages."""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from setup.performance_test_case import Limit, Ramp, ramped

from cases.small_bodies_5rps import CASE as SMALL

RAMP: Final = Ramp(first_rps=10.0, last_rps=70.0, step_rps=10.0, step_seconds=90.0, held_p99_seconds=5.0)

CASE: Final = replace(
    SMALL,
    name="small_bodies_ramp",
    traffic=ramped(SMALL.traffic, RAMP, seed=13, concurrency=400),
    limits=(Limit("sustained_rps", min=20),),
)
