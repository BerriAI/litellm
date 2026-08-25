"""
Types for auto-router management endpoints
"""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig
from litellm.types.utils import StandardLoggingRoutingDecision

DEFAULT_ROUTING_TEST_ROUTER_NAME: Final[str] = "auto_router_routing_test"


class RequestComplexityRouterConfig(ComplexityRouterConfig):
    """The part of a complexity-router config a request can carry.

    `plugins` holds live RoutingPlugin objects, which no JSON body can express and which have no
    OpenAPI schema, so it is closed off here rather than left as an arbitrary-type field.
    """

    plugins: None = Field(default=None, description="Not settable over HTTP; routing plugins are runtime objects")
    classifier_plugin: None = Field(  # pyright: ignore[reportIncompatibleVariableOverride]  # narrowing to None is the point: runtime objects are not settable over HTTP
        default=None, description="Not settable over HTTP; the classifier plugin is a runtime object"
    )


class ComplexityRouterConfigValidationRequest(BaseModel):
    """A complexity-router config to validate without saving, so a form can surface the
    backend's own verdict inline instead of a raw 400 at write time."""

    complexity_router_config: Mapping[str, object]
    team_id: str | None = Field(
        default=None,
        description="Team the router is being created for. Required for a team admin, who may only validate their own team's routers",
    )


class ComplexityRouterConfigValidationResponse(BaseModel):
    valid: bool
    error: str | None = None


class AutoRouterRoutingTestRequest(BaseModel):
    """A single prompt to classify against a complexity-router config that need not be saved yet."""

    prompt: str = Field(description="The prompt to route, as an end user would send it")
    complexity_router_config: RequestComplexityRouterConfig = Field(
        description="The complexity router config to route against, in the shape /model/new accepts",
    )
    default_model: str | None = Field(
        default=None,
        description="Model to route to when no tier resolves, i.e. complexity_router_default_model",
    )
    router_name: str = Field(
        default=DEFAULT_ROUTING_TEST_ROUTER_NAME,
        description="Name reported as the router in the routing decision. Display only",
    )
    team_id: str | None = Field(
        default=None,
        description="Team the router is being created for. Required for a team admin, who may only test their own team's routers",
    )

    @field_validator("prompt")
    @classmethod
    def _require_non_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class AutoRouterRoutingTestResponse(BaseModel):
    """Where one prompt would have been routed, and why."""

    routed_model: str = Field(description="The model group the router picked")
    routed_model_configured: bool = Field(
        description="Whether routed_model is a model group available to the caller, scoped to team_id when given. Never confirms models the caller could not use",
    )
    routing_decision: StandardLoggingRoutingDecision = Field(
        description="The decision record this request would have written to its log row",
    )


class AutoRouterCacheBucket(BaseModel):
    """One prompt-caching bucket of turns, with how often those turns hit the cache."""

    turns: int = Field(description="Turns classified into this bucket")
    hits: int = Field(description="Turns in this bucket whose response reported cache-read tokens")
    hit_rate_pct: float = Field(description="hits over this bucket's turns, as a percentage")


class AutoRouterCacheStats(BaseModel):
    """Prompt-caching behaviour of auto-routed turns, bucketed by what the router did.

    Every in-order turn falls in exactly one bucket: the session stayed on the same model,
    visited a model for the first time (cold by design), or returned to a model it had
    already used. Out-of-order turns (cross-pod flush races) are counted but not bucketed.
    """

    coverage_pct: float = Field(description="Share of turns that carried cache telemetry")
    hit_rate_pct: float = Field(description="All cache hits over telemetry-bearing turns")
    same_model: AutoRouterCacheBucket
    first_visit: AutoRouterCacheBucket
    return_to_tier: AutoRouterCacheBucket
    unordered_turns: int = Field(description="Turns that arrived out of order and were not bucketed")
    return_misses_expired: int = Field(
        description="Return-to-tier misses where the model's recorded cache TTL had lapsed"
    )
    return_misses_within_ttl: int = Field(
        description="Return-to-tier misses inside the recorded TTL: the prefix changed or the provider "
        "evicted the entry early; billing telemetry cannot distinguish the two"
    )
    return_misses_unknown: int = Field(description="Return-to-tier misses with no recorded TTL to attribute against")
    ttl_5m_turns: int = Field(description="Turns whose cache write used the five-minute TTL")
    ttl_1h_turns: int = Field(description="Turns whose cache write used the one-hour TTL")


class AutoRouterBenchmarkTotals(BaseModel):
    """Session-shape and savings aggregates over auto-routed traffic in the window."""

    sessions: int
    turns: int
    avg_turns_per_session: float
    avg_session_seconds: float
    avg_tokens_per_session: float
    spend: float = Field(description="What the routed traffic actually cost")
    saved_spend: float = Field(
        description="Signed dollars saved versus each router's savings baseline (derived from its hardest "
        "tier, or the configured override), from the same per-request savings record the usage tab reads"
    )
    baseline_spend: float = Field(description="spend plus saved_spend: the estimated single-model cost")
    saved_pct: float = Field(description="saved_spend over baseline_spend, as a percentage")
    saved_per_session: float
    cache: AutoRouterCacheStats


class AutoRouterBenchmarkGroup(AutoRouterBenchmarkTotals):
    """One auto-router's slice of the benchmarks."""

    router_name: str = Field(description="The auto-router alias requests were sent to")
    router_type: str = Field(description="complexity, adaptive or quality")
    tier_turns: Mapping[str, int] = Field(
        default_factory=dict,
        description="Turns per tier, keyed by the tier name the routing decision recorded at "
        "request time (never re-derived at read time, since the tier-to-model mapping is "
        "mutable config). Tier names are scoped to this group's router_type and are not "
        "comparable across types: a complexity router reports 'SIMPLE'/'MEDIUM'/'COMPLEX'/"
        "'REASONING', a quality router reports its numeric quality tier, and an adaptive router "
        "records no tier at all. Turns no tier served (the classifier fell back to default_model) "
        "are absent rather than pooled under a sentinel key, so the values may sum to less than turns",
    )


class AutoRouterBenchmarksResponse(BaseModel):
    """Benchmarks for the auto-router dashboard, aggregated from the per-session rollup."""

    start_date: str = Field(description="Window start day, YYYY-MM-DD UTC, inclusive")
    end_date: str = Field(description="Window end day, YYYY-MM-DD UTC, inclusive")
    routers_in_scope: int = Field(
        description="How many groups this response carries. Every auto-router configured on the "
        "proxy counts, whether or not it served anything in the window. To count only the routers "
        "that did serve traffic, filter `groups` to the entries whose `sessions` is above zero"
    )
    totals: AutoRouterBenchmarkTotals
    groups: tuple[AutoRouterBenchmarkGroup, ...] = Field(
        description="One entry per auto-router, listed from the model registry rather than from "
        "the rollup, so a router appears as soon as it is configured and reads zero until it "
        "serves traffic. Semantic auto-routers are absent: they record no routing decision, so no "
        "session can ever be attributed to them"
    )


ShadowEvalStatus: TypeAlias = Literal["running", "completed", "stopped"]

ShadowEvalDirection: TypeAlias = Literal["forward", "reverse"]

DEFAULT_SHADOW_EVAL_JUDGE_MODEL: Final[str] = "anthropic/claude-sonnet-5"

# Sample-count ceiling written on every new job: a zero-cost error loop (a shadow arm that
# fails before billing) never consumes spend budget, so it must terminate on count instead.
SHADOW_EVAL_TURN_VALVE: Final[int] = 10_000


class StartShadowEvalRequest(BaseModel):
    """Start duplicating one or more keys' traffic for blind comparison against an auto-router."""

    api_key_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=100,
        description=(
            "The hashed virtual keys whose traffic will be shadowed. Shadow evaluation runs ONLY on these "
            "keys' traffic; requests made with any other key are not sampled. Each key carries its own "
            "max_budget spend budget, so one key exhausting its budget leaves the others sampling. At most 100 "
            "keys per job, which also bounds every read the job's endpoints make."
        ),
    )
    router_name: str = Field(description="The auto-router under evaluation, in either direction")
    direction: ShadowEvalDirection = Field(
        default="forward",
        description=(
            "forward answers 'should this key adopt router_name': it samples the requests the key did NOT "
            "route through the router and duplicates them through it. reverse answers 'is the router still "
            "worth it for a key already on it': it samples the requests the router did serve and duplicates "
            "them against baseline_model. The response the caller received is always the real arm"
        ),
    )
    baseline_model: str | None = Field(
        default=None,
        description=(
            "Required when direction is reverse and rejected otherwise: the fixed model the router's own "
            "responses are judged against. Must be a plain model rather than another auto-router"
        ),
    )
    shadow_percentage: float = Field(
        ge=0.1,
        le=100.0,
        description="Percentage of the key's requests to duplicate through the router",
    )
    judge_model: str = Field(
        default=DEFAULT_SHADOW_EVAL_JUDGE_MODEL,
        description=(
            "Model used to blindly judge real vs. shadow responses. The judge only compares two answers, so a "
            "mid-tier model (Claude Sonnet or GPT-4o class) is the sweet spot: small/nano-class models produce "
            "unreliable or malformed verdicts, while frontier reasoning models add cost without changing outcomes."
        ),
    )
    duration_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="How many days the job samples traffic before completing on its own",
    )
    max_budget: float = Field(
        default=10.0,
        ge=0.01,
        le=10_000,
        description=(
            "Per-key USD budget for the eval's own overhead, the shadow-arm and judge calls, priced with "
            "the same figures the spend pipeline bills. EACH scoped key samples until its recorded eval "
            "spend reaches this, so a job over N keys spends at most about N times max_budget; in-flight "
            "samples can overshoot the cap by one sampling cache window"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_the_retired_turn_budget(cls, values: object) -> object:
        """Pydantic ignores unknown fields, so a caller still sending max_turns would
        silently run on the default dollar budget instead of the bound they asked for."""
        if isinstance(values, Mapping) and "max_turns" in values:
            raise ValueError("max_turns was replaced by max_budget, the per-key USD cap on the eval's own spend")
        return values

    @field_validator("shadow_percentage")
    @classmethod
    def _round_percentage(cls, value: float) -> float:
        return round(value, 2)

    @field_validator("api_key_ids")
    @classmethod
    def _dedupe_keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """A key named twice would collide with itself on the one-active-per-(key, direction) index."""
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def _baseline_model_matches_direction(self) -> "StartShadowEvalRequest":
        if self.direction == "reverse" and self.baseline_model is None:
            raise ValueError("baseline_model is required when direction is 'reverse'")
        if self.direction == "forward" and self.baseline_model is not None:
            raise ValueError("baseline_model is only meaningful when direction is 'reverse'")
        return self


class ShadowEvalSlice(BaseModel):
    """Judge outcomes for one slice of a job's verdicts (a router tier, or one of the
    models that served the real arm)."""

    group: str
    turn_count: int
    real_win_rate_pct: float = Field(
        description=(
            "Share of judged turns the real arm won, meaning the response the caller actually received: "
            "the key's own model in forward mode, the router's pick in reverse"
        )
    )
    shadow_win_rate_pct: float = Field(
        description=(
            "Share of judged turns the shadow arm won, meaning the duplicated response nobody was served: "
            "the router's pick in forward mode, baseline_model in reverse"
        )
    )
    tie_rate_pct: float
    avg_judge_confidence: float


class ShadowEvalResult(BaseModel):
    """Stratified results of a shadow-eval job's verdicts so far."""

    by_tier: tuple[ShadowEvalSlice, ...]
    by_current_model: tuple[ShadowEvalSlice, ...] = Field(
        description=(
            "Sliced by the model that served the real arm: the keys' incumbent models in forward mode, "
            "and in reverse the models the router itself picked"
        )
    )
    by_key: tuple[ShadowEvalSlice, ...] = Field(
        description=(
            "One slice per scoped key that has judged verdicts, grouped on the raw key hash. Keys the job "
            "scopes but has not judged a turn for yet are absent rather than reported as zero"
        ),
    )
    overall_shadow_win_rate_pct: float
    overall_tie_rate_pct: float


class ShadowEvalJobKeyResponse(BaseModel):
    """One key a job shadows, with its own budget and stop state."""

    api_key_id: str = Field(description="The hashed virtual key whose traffic this entry scopes")
    max_turns: int = Field(
        description=(
            "This key's sample-count ceiling: the whole budget for jobs created before max_budget "
            "existed, and the error-loop safety valve otherwise"
        )
    )
    max_budget: float | None = Field(
        default=None,
        description=(
            "This key's own USD budget for the eval's shadow and judge spend, independent of its "
            "siblings'; None on jobs created before spend budgets existed, which max_turns alone bounds"
        ),
    )
    stopped_at: datetime | None = Field(
        default=None,
        description=(
            "When this key's slot was stamped free, whether its own budget ran out, the window closed, "
            "or an operator stopped the job; status is derived, so a spent budget reads completed even "
            "while this is still unset"
        ),
    )
    attempt_count: int | None = Field(
        default=None,
        description=(
            "This key's sampled attempts so far, judged and errored alike, the same count the sampler "
            "budgets against max_turns; populated on list and detail responses. Frozen at stopped_at "
            "once the key is stamped, so in-flight attempts landing after a stop never reclassify it"
        ),
    )
    spend: float | None = Field(
        default=None,
        description=(
            "This key's recorded shadow plus judge spend in USD, the same figure the sampler budgets "
            "against max_budget; populated on list and detail responses and frozen at stopped_at "
            "exactly like attempt_count"
        ),
    )

    @property
    def budget_spent(self) -> bool:
        over_spend: Final = self.max_budget is not None and self.spend is not None and self.spend >= self.max_budget
        return over_spend or (self.attempt_count is not None and self.attempt_count >= self.max_turns)

    key_alias: str | None = Field(
        default=None,
        description="Alias of the shadowed key, resolved from the key row at read time; None when unset or deleted",
    )
    key_name: str | None = Field(
        default=None,
        description="Masked display name (sk-...) of the shadowed key, resolved at read time like key_alias",
    )


class ShadowEvalJobResponse(BaseModel):
    """A shadow-eval job over one or more keys, each with its own budget and stop state;
    status is derived from stopped_by, the keys' stop and budget state, and ends_at,
    never stored, so no writer anywhere can produce an inconsistent one. Aggregate
    fields are populated by the detail endpoint only and stay None on list responses."""

    job_id: str
    keys: tuple[ShadowEvalJobKeyResponse, ...] = Field(
        min_length=1,
        description="The keys whose traffic this job evaluates, and only those keys', each with its own budget",
    )
    router_name: str
    direction: ShadowEvalDirection = "forward"
    baseline_model: str | None = None
    judge_model: str
    shadow_percentage: float
    created_at: datetime
    ends_at: datetime
    stopped_by: str | None = Field(
        default=None,
        description=(
            "The operator who stopped the job early, recorded by the stop endpoint; 'unknown' backfilled "
            "by migration for jobs that displayed stopped when the column arrived; None when the job "
            "ended on its own. Its presence is what makes a job read stopped rather than completed"
        ),
    )

    judged_count: int | None = Field(default=None, description="Verdicts recorded; detail endpoint only")
    error_count: int | None = Field(default=None, description="Sampled attempts that errored; detail endpoint only")
    judge_spend: float | None = Field(default=None, description="Judge cost so far; detail endpoint only")
    last_error: str | None = Field(default=None, description="Most recent attempt error; detail endpoint only")
    results: ShadowEvalResult | None = Field(default=None, description="Stratified verdicts; detail endpoint only")

    @computed_field
    @property
    def status(self) -> ShadowEvalStatus:
        """Three recorded facts, no history-guessing: a stop is stopped_by (the migration
        backfills it for every job that displayed stopped when the column arrived, so the
        pre-column population is closed), completion is the window passing or every key
        spending its budget, and anything else is running. The all-keys-stamped fallback
        covers only stops written by pre-column pods during a rolling deploy."""
        if self.stopped_by is not None:
            return "stopped"
        if datetime.now(timezone.utc) >= (
            self.ends_at if self.ends_at.tzinfo else self.ends_at.replace(tzinfo=timezone.utc)
        ):
            return "completed"
        if all(key.budget_spent for key in self.keys):
            return "completed"
        if all(key.stopped_at is not None for key in self.keys):
            return "stopped"
        return "running"
