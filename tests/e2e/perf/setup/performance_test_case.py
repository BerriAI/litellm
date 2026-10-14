"""The schema of a perf test: a `PerformanceTestCase` names what the gateway is deployed with
(model groups and guardrails, both written out in the gateway's config) and lists what the test
itself creates and sends: teams, Keycloak users, virtual keys, the request shapes, the traffic plan
and the `Limit`s that decide pass or fail. Add a case by adding a module to `cases/`.

The same shape as `CostTrackingTestCase` in the cost suite: frozen dataclasses with plain defaults,
so a case is a literal a reviewer can read top to bottom.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

Endpoint = Literal["chat_completions", "responses", "messages"]
GuardrailMode = Literal["pre_call", "post_call"]
Metric = Literal[
    "unexpected_status_count",
    "gateway_added_seconds",
    "time_to_first_token_seconds",
    "memory_growth_megabytes",
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
    """What one request looks like, in tokens: a coding-agent session (`setup/corpus/session.py`, the system
    prompt, the Read/Edit/Write/Bash/Grep/Glob tools when `tools` is set, then file-reading turns carrying real
    source files until `input_tokens` is spent) and the output cap. The stock shapes are the ones the mocks hold
    real provider captures for: a short chat under 40k tokens, a tool-calling turn around 64k, a long context
    over 300k. A new shape needs its own captures or the mock answers with the nearest one."""

    name: str
    input_tokens: int
    tools: bool = False
    output_tokens: int = 1_024


SHORT_CHAT: Final = RequestShape("short_chat", input_tokens=38_500, output_tokens=1_024)
TOOL_CALL: Final = RequestShape("tool_call", input_tokens=63_600, tools=True, output_tokens=1_024)
LONG_CONTEXT: Final = RequestShape("long_context", input_tokens=335_000, output_tokens=1_024)


@dataclass(frozen=True, slots=True)
class Traffic:
    """The plan: `total_requests` drawn from `seed`, sent at `rps` by `concurrency` simulated
    users. Shares are fractions of the requests: `endpoints` and `shapes` must each sum to 1,
    `intervene_share` picks non-streaming requests the guardrail rewrites, `block_share` picks
    requests it blocks."""

    seed: int
    rps: float
    total_requests: int
    endpoints: Mapping[Endpoint, float] = field(default_factory=lambda: {"chat_completions": 1.0})
    shapes: Mapping[RequestShape, float] = field(default_factory=lambda: {SHORT_CHAT: 1.0})
    stream_share: float = 0.0
    intervene_share: float = 0.0
    block_share: float = 0.0
    concurrency: int = 20

    @property
    def expected_seconds(self) -> float:
        return self.total_requests / self.rps


@dataclass(frozen=True, slots=True)
class Limit:
    """One pass/fail rule: `metric` (at `percentile`, or the whole-run value) must not exceed `max`.
    Metrics without a Limit are still reported."""

    metric: Metric
    max: float
    percentile: int | None = None

    @property
    def label(self) -> str:
        return f"{self.metric} percentile {self.percentile}" if self.percentile is not None else self.metric


@dataclass(frozen=True, slots=True)
class PerformanceTestCase:
    """`model_groups` and `guardrails` name what the gateway config deploys (setup checks they are
    there); everything else the test creates or sends."""

    name: str
    teams: tuple[Team, ...]
    users: tuple[User, ...]
    virtual_keys: tuple[VirtualKey, ...]
    model_groups: tuple[str, ...]
    guardrails: tuple[Guardrail, ...]
    traffic: Traffic
    limits: tuple[Limit, ...]

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
        assert self.model_groups, f"{self.name}: no model groups"
