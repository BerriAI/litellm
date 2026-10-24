"""The schema of a perf test: a `PerformanceTestCase` names what the gateway is deployed with
(model groups and guardrails, both written out in the gateway's config) and lists what the test
itself creates and sends: teams, Keycloak users, virtual keys, the request shapes, the traffic plan
and the `Limit`s that decide pass or fail. Add a case by adding a module to `cases/`.

The same shape as `CostTrackingTestCase` in the cost suite: frozen dataclasses with plain defaults,
so a case is a literal a reviewer can read top to bottom.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import accumulate
from typing import Final, Literal

from setup.event_loop import EventLoopStalls

Endpoint = Literal["chat_completions", "responses", "messages"]
MockProvider = Literal["openai", "anthropic", "bedrock"]
GuardrailMode = Literal["pre_call", "post_call"]
Metric = Literal[
    "unexpected_status_count",
    "gateway_added_seconds",
    "time_to_first_token_seconds",
    "memory_growth_megabytes",
    "postgres_calls_per_request",
    "postgres_background_calls_per_request",
    "postgres_max_statement_per_request",
    "postgres_max_background_statement_per_request",
    "cpu_seconds_per_request",
    "redis_calls_per_request",
    "redis_background_calls_per_request",
    "litellm_overhead_seconds",
    "litellm_response_seconds",
    "prometheus_redis_latency_mean_seconds",
    "postgres_active_sessions",
    "postgres_slowest_statement_seconds",
    "postgres_peak_connections",
    "sustained_rps",
]


@dataclass(frozen=True, slots=True)
class Team:
    """One team (`/team/new`). `models` names the model groups the team may call; empty means all."""

    name: str
    models: tuple[str, ...] = ()
    max_budget: float | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None


@dataclass(frozen=True, slots=True)
class User:
    """One Keycloak user who calls the gateway with a JWT. `team` puts them in that team's group."""

    name: str
    team: str | None = None
    max_budget: float | None = None


@dataclass(frozen=True, slots=True)
class VirtualKey:
    """One virtual key (`/key/generate`), scoped to a team and/or a user when named."""

    name: str
    team: str | None = None
    user: str | None = None
    models: tuple[str, ...] = ()
    rpm_limit: int | None = None
    max_budget: float | None = None


@dataclass(frozen=True, slots=True)
class Guardrail:
    """A guardrail the gateway's config defines (a `generic_guardrail_api` pointed at the mock
    guardrail, `default_on`); the case names it so the plan knows which hooks every request sees."""

    name: str
    mode: GuardrailMode


@dataclass(frozen=True, slots=True)
class RequestShape:
    """What one request looks like: a coding-agent session (`setup/corpus/session.py`, the system prompt, the
    Read/Edit/Write/Bash/Grep/Glob tools when `tools` is set, then file-reading turns carrying real source files)
    whose size in input tokens is drawn per request from the `input_tokens` range, and the output cap. The mocks
    answer with a reply generated for the request, sized from that cap, a tool call when `tools` is set."""

    name: str
    input_tokens: tuple[int, int]
    tools: bool = False
    output_tokens: int = 1_024


SHORT: Final = RequestShape("short", input_tokens=(5_000, 100_000), tools=True)
LONG: Final = RequestShape("long", input_tokens=(300_000, 1_000_000), tools=True)


@dataclass(frozen=True, slots=True)
class Ramp:
    """Rate stepped up through the run: `first_rps`, then `step_rps` more every `step_seconds`, up to `last_rps`.
    Each step is scored on its own and the run reports `sustained_rps`, the last step the pod held before the
    first one it did not: a step holds when its `gateway_added_seconds` p99 stayed under `held_p99_seconds`, it
    completed at least `HELD_SHARE` of its rate and no more than `HELD_UNEXPECTED_SHARE` of its requests came back
    with an unexpected status (`setup/ramp.py`). Requests are sent on a schedule drawn from the plan's seed, not
    from per-user intervals, so the rate changes on the step boundary."""

    first_rps: float
    last_rps: float
    step_rps: float
    step_seconds: float
    held_p99_seconds: float

    @property
    def rates(self) -> tuple[float, ...]:
        count: Final = round((self.last_rps - self.first_rps) / self.step_rps) + 1
        return tuple(self.first_rps + index * self.step_rps for index in range(count))

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(round(rate * self.step_seconds) for rate in self.rates)

    @property
    def total_requests(self) -> int:
        return sum(self.counts)

    @property
    def seconds(self) -> float:
        return len(self.rates) * self.step_seconds

    def step_of(self, seq: int) -> int:
        """Zero-based step of the request with 1-based plan number `seq`."""
        return bisect.bisect_left(tuple(accumulate(self.counts)), seq)


@dataclass(frozen=True, slots=True)
class Traffic:
    """The plan: `total_requests` drawn from `seed`, sent at `rps` by `concurrency` simulated
    users, or stepped up through a `ramp` (then `rps` is its first step and `total_requests` its
    total, see `ramped`). Shares are fractions of the requests: `endpoints` and `shapes` must each
    sum to 1, `intervene_share` picks non-streaming requests the guardrail rewrites, `block_share`
    picks requests it blocks."""

    seed: int
    rps: float
    total_requests: int
    endpoints: Mapping[Endpoint, float] = field(default_factory=lambda: {"chat_completions": 1.0})
    shapes: Mapping[RequestShape, float] = field(default_factory=lambda: {SHORT: 1.0})
    stream_share: float = 0.0
    intervene_share: float = 0.0
    block_share: float = 0.0
    concurrency: int = 20
    ramp: Ramp | None = None

    @property
    def expected_seconds(self) -> float:
        return self.total_requests / self.rps if self.ramp is None else self.ramp.seconds


def ramped(base: Traffic, ramp: Ramp, *, seed: int, concurrency: int) -> Traffic:
    """`base`'s request mix sent along `ramp` instead of at one rate. `concurrency` caps the requests in
    flight, so size it past the deepest step the ramp is expected to reach."""
    return replace(
        base, seed=seed, rps=ramp.first_rps, total_requests=ramp.total_requests, concurrency=concurrency, ramp=ramp
    )


@dataclass(frozen=True, slots=True)
class Limit:
    """One pass/fail rule: `metric` (at `percentile`, or the whole-run value) must not exceed `max`, or must
    reach `min` for a floor such as `sustained_rps`. Metrics without a Limit are still reported."""

    metric: Metric
    max: float | None = None
    percentile: int | None = None
    min: float | None = None

    def __post_init__(self) -> None:
        assert (self.max is None) != (self.min is None), f"{self.metric}: a Limit is either a max or a min"

    @property
    def label(self) -> str:
        return f"{self.metric} percentile {self.percentile}" if self.percentile is not None else self.metric


@dataclass(frozen=True, slots=True)
class MockLatency:
    """Seconds a mock provider waits before its first byte, plus or minus up to `jitter_seconds` per request
    (drawn from the request marker, so a rerun waits the same). Zero means the mocks answer at once."""

    seconds: float = 0.0
    jitter_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PerformanceTestCase:
    """`model_groups` and `guardrails` name what the gateway config deploys (setup checks they are
    there); everything else the test creates or sends. `mock_latency` is how long every mock provider waits
    before answering, set on the mock pods at setup, so the provider side of each request is a known
    constant and the rest of the client's time is the gateway's."""

    name: str
    teams: tuple[Team, ...]
    users: tuple[User, ...]
    virtual_keys: tuple[VirtualKey, ...]
    model_groups: tuple[str, ...]
    guardrails: tuple[Guardrail, ...]
    traffic: Traffic
    limits: tuple[Limit, ...]
    mock_latency: MockLatency = field(default_factory=lambda: MockLatency())
    background_cap_exempt_tables: frozenset[str] = frozenset()
    """Tables whose background write is one upsert per writer flush and, at this case's request rate, lands
    above the `postgres_max_background_statement_per_request` cap without being per request."""
    event_loop: EventLoopStalls | None = None

    def team(self, name: str) -> Team:
        return next(team for team in self.teams if team.name == name)

    @property
    def has_post_guardrail(self) -> bool:
        return any(guardrail.mode == "post_call" for guardrail in self.guardrails)

    def validate(self) -> None:
        """Fail fast on a case that cannot run: dangling names, shares that do not add up, a model
        group no caller may use."""
        team_names: Final = {team.name for team in self.teams}
        user_names: Final = {user.name for user in self.users}
        group_names: Final = set(self.model_groups)
        for names, label in (
            ([team.name for team in self.teams], "team"),
            ([user.name for user in self.users], "user"),
            ([key.name for key in self.virtual_keys], "key"),
            (list(self.model_groups), "model group"),
            ([guardrail.name for guardrail in self.guardrails], "guardrail"),
        ):
            assert len(names) == len(set(names)), f"{self.name}: duplicate {label} names in {names}"
        for team in self.teams:
            assert set(team.models) <= group_names, f"{self.name}: team {team.name} names unknown models {team.models}"
        for user in self.users:
            assert user.team is None or user.team in team_names, (
                f"{self.name}: user {user.name} in unknown team {user.team}"
            )
        for key in self.virtual_keys:
            assert key.team is None or key.team in team_names, f"{self.name}: key {key.name} in unknown team {key.team}"
            assert key.user is None or key.user in user_names, (
                f"{self.name}: key {key.name} for unknown user {key.user}"
            )
            assert set(key.models) <= group_names, f"{self.name}: key {key.name} names unknown models {key.models}"
        assert abs(sum(self.traffic.endpoints.values()) - 1.0) < 1e-9, f"{self.name}: endpoint shares must sum to 1"
        assert abs(sum(self.traffic.shapes.values()) - 1.0) < 1e-9, f"{self.name}: shape shares must sum to 1"
        assert 0 <= self.traffic.stream_share <= 1, f"{self.name}: stream_share out of range"
        assert self.traffic.intervene_share + self.traffic.block_share <= 1, f"{self.name}: guardrail shares exceed 1"
        assert self.traffic.total_requests > 0 and self.traffic.rps > 0, f"{self.name}: empty traffic"
        ramp: Final = self.traffic.ramp
        assert ramp is None or (ramp.first_rps, ramp.total_requests) == (
            self.traffic.rps,
            self.traffic.total_requests,
        ), f"{self.name}: traffic rps and total_requests must be the ramp's, build it with ramped()"
        assert ramp is None or (0 < ramp.first_rps <= ramp.last_rps and ramp.step_rps > 0 and ramp.step_seconds > 0), (
            f"{self.name}: ramp {ramp} does not step up"
        )
        assert self.model_groups, f"{self.name}: no model groups"
        assert 0 <= self.mock_latency.jitter_seconds <= self.mock_latency.seconds or self.mock_latency.seconds == 0, (
            f"{self.name}: mock latency jitter {self.mock_latency.jitter_seconds} s exceeds {self.mock_latency.seconds} s"
        )
