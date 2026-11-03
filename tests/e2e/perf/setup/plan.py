"""The deterministic request plan of a `PerformanceTestCase`: every request the run sends, in order,
drawn once from the case's seed so two runs of the same case send the same traffic. Each request
carries a marker (`perfreq-000123-n`) that the runner puts in the final user message and the mock
providers and the mock guardrail read back, so the test can join what the client saw against what
each mock received, and a `RequestShape` the runner hands the corpus to build the body."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeVar

from setup.performance_test_case import Endpoint, PerformanceTestCase, RequestShape, Team
from setup.ramp import send_offsets

Flag = Literal["n", "i", "b"]
CallerKind = Literal["jwt", "key"]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Caller:
    kind: CallerKind
    name: str

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True, slots=True)
class PlannedRequest:
    seq: int
    marker: str
    caller: Caller
    endpoint: Endpoint
    model_group: str
    stream: bool
    flag: Flag
    shape: RequestShape
    input_tokens: int
    send_at_s: float | None = None

    @property
    def blocked(self) -> bool:
        return self.flag == "b"


def _allowed_models(case: PerformanceTestCase, team: Team | None, own: tuple[str, ...]) -> frozenset[str]:
    if own:
        return frozenset(own)
    if team is not None and team.models:
        return frozenset(team.models)
    return frozenset(case.model_groups)


def _callers_by_group(case: PerformanceTestCase) -> dict[str, tuple[Caller, ...]]:
    """Which callers may use each model group, in case order, so a caller never asks for a model its
    team or key does not allow and every denial in the run is a bug, not the plan."""
    reach: list[tuple[Caller, frozenset[str]]] = []
    for user in case.users:
        user_team = case.team(user.team) if user.team else None
        reach.append((Caller("jwt", user.name), _allowed_models(case, user_team, ())))
    for key in case.virtual_keys:
        key_team = case.team(key.team) if key.team else None
        reach.append((Caller("key", key.name), _allowed_models(case, key_team, key.models)))
    by_group: dict[str, tuple[Caller, ...]] = {}
    for group in case.model_groups:
        by_group[group] = tuple(caller for caller, models in reach if group in models)
        assert by_group[group], f"{case.name}: no caller may use model group {group}"
    return by_group


def _pick(rng: random.Random, shares: Mapping[T, float]) -> T:
    draw: Final = rng.random()
    cumulative = 0.0
    chosen: T | None = None
    for item, share in shares.items():
        cumulative += share
        chosen = item
        if draw < cumulative:
            break
    assert chosen is not None, "empty shares"
    return chosen


def _pick_flag(rng: random.Random, case: PerformanceTestCase, *, stream: bool) -> Flag:
    if not case.guardrails:
        return "n"
    draw: Final = rng.random()
    if draw < case.traffic.block_share:
        return "b"
    if draw < case.traffic.block_share + case.traffic.intervene_share and not stream:
        return "i"
    return "n"


def build_plan(case: PerformanceTestCase) -> tuple[PlannedRequest, ...]:
    case.validate()
    rng: Final = random.Random(case.traffic.seed)
    callers: Final = _callers_by_group(case)
    offsets: Final = None if case.traffic.ramp is None else send_offsets(case.traffic.ramp, case.traffic.seed)
    planned: list[PlannedRequest] = []
    for seq in range(1, case.traffic.total_requests + 1):
        endpoint = _pick(rng, case.traffic.endpoints)
        group = rng.choice(case.model_groups)
        stream = rng.random() < case.traffic.stream_share
        flag = _pick_flag(rng, case, stream=stream)
        shape = _pick(rng, case.traffic.shapes)
        low, high = shape.input_tokens
        planned.append(
            PlannedRequest(
                seq=seq,
                marker=f"perfreq-{seq:06d}-{flag}",
                caller=rng.choice(callers[group]),
                endpoint=endpoint,
                model_group=group,
                stream=stream,
                flag=flag,
                shape=shape,
                input_tokens=rng.randint(low, high),
                send_at_s=None if offsets is None else offsets[seq - 1],
            )
        )
    return tuple(planned)
