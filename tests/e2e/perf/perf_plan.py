"""The deterministic request plan of a `PerformanceTestCase`: every request the run sends, in order,
drawn once from the case's seed so two runs of the same case send the same traffic. Each request
carries a marker (`perfreq-000123-n`) that the mock providers and the mock guardrail read back, so
the test can join what the client saw against what each mock received."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final, Literal

from perf_case import Endpoint, PerformanceTestCase, Team

Flag = Literal["n", "i", "b"]
CallerKind = Literal["jwt", "key"]


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
    prompt: str

    def expected_text(self, *, post_guardrail: bool) -> str:
        """What the client should read back: the mock's reply, plus the post-call rewrite when the
        case registers a post guardrail and this request is in the rewrite slice."""
        reply: Final = f"ack {self.marker}"
        return reply + " [post]" if self.flag == "i" and post_guardrail else reply

    @property
    def blocked(self) -> bool:
        return self.flag == "b"


def _allowed_models(case: PerformanceTestCase, team: Team | None, own: tuple[str, ...]) -> frozenset[str]:
    everything: Final = frozenset(group.name for group in case.model_groups)
    if own:
        return frozenset(own)
    if team is not None and team.models:
        return frozenset(team.models)
    return everything


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
        by_group[group.name] = tuple(caller for caller, models in reach if group.name in models)
        assert by_group[group.name], f"{case.name}: no caller may use model group {group.name}"
    return by_group


def _pick_endpoint(rng: random.Random, case: PerformanceTestCase) -> Endpoint:
    draw: Final = rng.random()
    cumulative = 0.0
    chosen: Endpoint = "chat_completions"
    for endpoint, share in case.traffic.endpoints.items():
        cumulative += share
        chosen = endpoint
        if draw < cumulative:
            break
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
    groups: Final = tuple(group.name for group in case.model_groups)
    planned: list[PlannedRequest] = []
    for seq in range(1, case.traffic.total_requests + 1):
        endpoint = _pick_endpoint(rng, case)
        group = rng.choice(groups)
        stream = rng.random() < case.traffic.stream_share
        flag = _pick_flag(rng, case, stream=stream)
        marker = f"perfreq-{seq:06d}-{flag}"
        planned.append(
            PlannedRequest(
                seq=seq,
                marker=marker,
                caller=rng.choice(callers[group]),
                endpoint=endpoint,
                model_group=group,
                stream=stream,
                flag=flag,
                prompt=f"{marker} say hi, request {seq} of {case.traffic.total_requests}",
            )
        )
    return tuple(planned)
