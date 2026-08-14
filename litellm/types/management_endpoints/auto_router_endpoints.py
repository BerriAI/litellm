"""
Types for auto-router management endpoints
"""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final, Literal, TypeAlias

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, computed_field, field_validator

from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig
from litellm.types.utils import StandardLoggingRoutingDecision

DEFAULT_ROUTING_TEST_ROUTER_NAME: Final[str] = "auto_router_routing_test"


class RequestComplexityRouterConfig(ComplexityRouterConfig):
    """The part of a complexity-router config a request can carry.

    `plugins` holds live RoutingPlugin objects, which no JSON body can express and which have no
    OpenAPI schema, so it is closed off here rather than left as an arbitrary-type field.
    """

    plugins: None = Field(default=None, description="Not settable over HTTP; routing plugins are runtime objects")


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
        description="Whether routed_model is a model group this proxy actually serves",
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
    routers_in_scope: int
    totals: AutoRouterBenchmarkTotals
    groups: tuple[AutoRouterBenchmarkGroup, ...]


ShadowEvalStatus: TypeAlias = Literal["running", "completed", "stopped"]

DEFAULT_SHADOW_EVAL_JUDGE_MODEL: Final[str] = "anthropic/claude-sonnet-5"


class StartShadowEvalRequest(BaseModel):
    """Start shadowing a key's traffic through an auto-router for blind comparison."""

    api_key_id: str = Field(
        description=(
            "The hashed virtual key whose traffic will be shadowed. Shadow evaluation runs ONLY on this "
            "key's traffic; requests made with any other key are not sampled."
        )
    )
    router_name: str = Field(description="The auto-router config to shadow requests through")
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
    max_turns: int = Field(
        default=200,
        ge=1,
        le=2000,
        description=(
            "Sample budget: the job judges at most this many turns, then completes. This is also the spend "
            "bound; expected judge cost is roughly max_turns times one judge call"
        ),
    )

    @field_validator("shadow_percentage")
    @classmethod
    def _round_percentage(cls, value: float) -> float:
        return round(value, 2)


class ShadowEvalSlice(BaseModel):
    """Judge outcomes for one slice of a job's verdicts (a router tier, or one of the
    models the shadowed key currently uses)."""

    group: str
    turn_count: int
    real_win_rate_pct: float = Field(description="Share of judged turns where the real (control) model won")
    shadow_win_rate_pct: float = Field(description="Share of judged turns where the shadowed router's pick won")
    tie_rate_pct: float
    avg_judge_confidence: float


class ShadowEvalResult(BaseModel):
    """Stratified results of a shadow-eval job's verdicts so far."""

    by_tier: tuple[ShadowEvalSlice, ...]
    by_current_model: tuple[ShadowEvalSlice, ...]
    overall_shadow_win_rate_pct: float
    overall_tie_rate_pct: float


class ShadowEvalJobResponse(BaseModel):
    """A shadow-eval job. Validates directly from the prisma record (job_id reads the
    row's id); status is derived from stopped_at and ends_at, never stored, so no writer
    anywhere can produce an inconsistent one. Aggregate fields are populated by the
    detail endpoint only and stay None on list responses."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    job_id: str = Field(validation_alias=AliasChoices("id", "job_id"))
    api_key_id: str = Field(description="The hashed virtual key whose traffic this job evaluates, and only that key's")
    router_name: str
    judge_model: str
    shadow_percentage: float
    max_turns: int
    created_at: datetime
    ends_at: datetime
    stopped_at: datetime | None = None

    judged_count: int | None = Field(default=None, description="Verdicts recorded; detail endpoint only")
    error_count: int | None = Field(default=None, description="Sampled attempts that errored; detail endpoint only")
    judge_spend: float | None = Field(default=None, description="Judge cost so far; detail endpoint only")
    last_error: str | None = Field(default=None, description="Most recent attempt error; detail endpoint only")
    results: ShadowEvalResult | None = Field(default=None, description="Stratified verdicts; detail endpoint only")

    @computed_field
    @property
    def status(self) -> ShadowEvalStatus:
        """A job whose window has passed reads completed even if a later sweep stamped
        stopped_at; stopped means sampling ended before the window did."""
        if datetime.now(timezone.utc) >= (
            self.ends_at if self.ends_at.tzinfo else self.ends_at.replace(tzinfo=timezone.utc)
        ):
            return "completed"
        if self.stopped_at is not None:
            return "stopped"
        return "running"
