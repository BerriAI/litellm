"""One `PerformanceTestCase` is one perf test: who calls (teams, users, keys), what they call (model
groups over mock deployments, guardrails), how (the traffic plan) and what makes the run red (the
limits). `test_perf_cases.py` builds the setup from the case, drives the plan through the gateway
from a separate process against the mock providers, and checks every `Limit` against what was
observed. Adding a perf test is adding a case to `perf_cases.py`; the stack size (pods, workers,
CPU) is the lane's, not the case's, because changing it means a different deployment.

The same shape as `CostTrackingTestCase` in the cost suite: frozen dataclasses, a `CASES` tuple,
one parametrized test over it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

from models import UserRole

ProviderKind = Literal["openai", "anthropic", "bedrock"]
BedrockRoute = Literal["converse", "invoke"]
Endpoint = Literal["chat_completions", "responses", "messages"]
GuardrailMode = Literal["pre_call", "post_call"]
Metric = Literal[
    "added_latency_s",
    "header_added_s",
    "ttft_added_s",
    "client_5xx",
    "incorrect_responses",
    "guardrail_calls_per_request",
    "guardrail_missed",
    "retried_share",
    "rss_growth_mb",
    "wall_seconds",
    "spend_logged_share",
]


@dataclass(frozen=True, slots=True)
class Team:
    """One litellm team (`/team/new`); `models` empty means every model."""

    name: str
    models: tuple[str, ...] = ()
    max_budget: float | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None


@dataclass(frozen=True, slots=True)
class User:
    """One Keycloak user who authenticates with the JWT Keycloak signs for them; `team` names the
    team whose id the JWT's `groups` claim carries, so the gateway attributes the request to it."""

    name: str
    team: str | None = None
    role: UserRole = "internal_user"
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
class Provider:
    """A mock provider family and how it behaves unless a deployment says otherwise: base latency,
    jitter (a third of the latency when not given, since a real provider never answers in a
    constant time) and stream speed. `OPENAI`, `ANTHROPIC` and `BEDROCK` are the stock ones."""

    kind: ProviderKind
    latency_s: float
    jitter_s: float | None = None
    tokens_per_second: float = 50.0

    def jitter_for(self, latency_s: float) -> float:
        return round(latency_s / 3, 3) if self.jitter_s is None else self.jitter_s


OPENAI: Final = Provider("openai", latency_s=0.6)
ANTHROPIC: Final = Provider("anthropic", latency_s=1.0)
BEDROCK: Final = Provider("bedrock", latency_s=1.2)


@dataclass(frozen=True, slots=True)
class MockDeployment:
    """One deployment on a mock provider. Every knob falls back to the provider's defaults, so
    `MockDeployment(OPENAI)` already behaves like a provider; set only what this deployment does
    differently (slower, flaky, a different Bedrock route). The knobs travel in the model id the
    gateway sends, so one mock pod per provider serves every case."""

    provider: Provider
    latency_s: float | None = None
    jitter_s: float | None = None
    flake_percent: int = 0
    flake_status: int = 500
    tokens_per_second: float | None = None
    bedrock_route: BedrockRoute = "converse"

    @property
    def kind(self) -> ProviderKind:
        return self.provider.kind

    @property
    def effective_latency_s(self) -> float:
        return self.provider.latency_s if self.latency_s is None else self.latency_s

    @property
    def effective_jitter_s(self) -> float:
        return self.provider.jitter_for(self.effective_latency_s) if self.jitter_s is None else self.jitter_s

    @property
    def effective_tokens_per_second(self) -> float:
        return self.provider.tokens_per_second if self.tokens_per_second is None else self.tokens_per_second

    @property
    def model_suffix(self) -> str:
        return (
            f"perf-l{round(self.effective_latency_s * 1000)}-j{round(self.effective_jitter_s * 1000)}"
            f"-f{self.flake_percent}-s{self.flake_status}-t{round(self.effective_tokens_per_second)}"
        )

    @property
    def max_delay_s(self) -> float:
        return self.effective_latency_s + self.effective_jitter_s


@dataclass(frozen=True, slots=True)
class ModelGroup:
    name: str
    deployments: tuple[MockDeployment, ...]

    @property
    def flaky(self) -> bool:
        return any(deployment.flake_percent > 0 for deployment in self.deployments)


@dataclass(frozen=True, slots=True)
class Guardrail:
    """One registration of the mock guardrail (`generic_guardrail_api`) on one hook."""

    name: str
    mode: GuardrailMode
    latency_s: float = 0.0


@dataclass(frozen=True, slots=True)
class Traffic:
    """The plan: `total_requests` drawn from `seed`, sent at `rps` by `concurrency` simulated
    users. Shares are fractions of the requests: `endpoints` must sum to 1, `intervene_share` picks
    non-streaming requests the guardrail rewrites, `block_share` picks requests it blocks."""

    seed: int
    rps: float
    total_requests: int
    endpoints: Mapping[Endpoint, float] = field(default_factory=lambda: {"chat_completions": 1.0})
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
        return f"{self.metric} p{self.percentile}" if self.percentile is not None else self.metric


@dataclass(frozen=True, slots=True)
class PerformanceTestCase:
    name: str
    teams: tuple[Team, ...]
    users: tuple[User, ...]
    virtual_keys: tuple[VirtualKey, ...]
    model_groups: tuple[ModelGroup, ...]
    guardrails: tuple[Guardrail, ...]
    traffic: Traffic
    limits: tuple[Limit, ...]

    def team(self, name: str) -> Team:
        return next(team for team in self.teams if team.name == name)

    def group(self, name: str) -> ModelGroup:
        return next(group for group in self.model_groups if group.name == name)

    @property
    def has_post_guardrail(self) -> bool:
        return any(guardrail.mode == "post_call" for guardrail in self.guardrails)

    def validate(self) -> None:
        """Fail fast on a case that cannot run: dangling names, shares that do not add up, a model
        group no caller may use."""
        team_names: Final = {team.name for team in self.teams}
        user_names: Final = {user.name for user in self.users}
        group_names: Final = {group.name for group in self.model_groups}
        for collection, label in (
            (self.teams, "team"),
            (self.users, "user"),
            (self.virtual_keys, "key"),
            (self.model_groups, "model group"),
            (self.guardrails, "guardrail"),
        ):
            names = [item.name for item in collection]
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
        assert 0 <= self.traffic.stream_share <= 1, f"{self.name}: stream_share out of range"
        assert self.traffic.intervene_share + self.traffic.block_share <= 1, f"{self.name}: guardrail shares exceed 1"
        assert self.traffic.total_requests > 0 and self.traffic.rps > 0, f"{self.name}: empty traffic"
        for group in self.model_groups:
            assert group.deployments, f"{self.name}: model group {group.name} has no deployments"
