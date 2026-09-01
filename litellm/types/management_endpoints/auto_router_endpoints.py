"""
Types for auto-router management endpoints
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from litellm.router_strategy.capability_router.config import CapabilityRouterConfig
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


class CapabilityRouterConfigValidationRequest(BaseModel):
    """A capability-router config to validate without saving."""

    capability_router_config: Mapping[str, object]
    team_id: str | None = None


class CapabilityRouterConfigValidationResponse(BaseModel):
    valid: bool
    error: str | None = None


class AutoRouterRoutingTestRequest(BaseModel):
    """A single request to classify against a complexity-router config that need not be saved yet.

    Carries the same fields the serving path carries, so a dry run classifies what a real turn
    would classify. `messages`, `system` and `tools` are forwarded to the routing hook untranslated,
    which is why they are typed loosely: the hook reads whatever dialect the surface produced, and
    validating them against one surface's schema would reject the others.
    """

    prompt: str | None = Field(
        default=None,
        description="A single ask to route, as an end user would send it. Mutually exclusive with messages",
    )
    messages: Sequence[Mapping[str, object]] | None = Field(
        default=None,
        description="The full message list to route, exactly as the serving path would receive it. Mutually exclusive with prompt",
    )
    system: str | Sequence[Mapping[str, object]] | None = Field(
        default=None,
        description="The top-level system prompt an Anthropic /v1/messages body carries beside its messages",
    )
    tools: Sequence[Mapping[str, object]] | None = Field(
        default=None,
        description="The tool definitions the request advertises, which decide whether the plan-mode floor applies",
    )
    complexity_router_config: RequestComplexityRouterConfig | None = Field(
        default=None,
        description="The complexity router config to route against",
    )
    capability_router_config: CapabilityRouterConfig | None = Field(
        default=None,
        description="The capability router config to route against",
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

    @field_validator("messages")
    @classmethod
    def _reject_messages_no_surface_accepts(
        cls, value: Sequence[Mapping[str, object]] | None
    ) -> Sequence[Mapping[str, object]] | None:
        """Reject what every supported surface rejects, and nothing beyond it.

        A real request carrying a message with no string role, or with content that is neither text
        nor a block list, is a 400 on the serving path, so answering it here with a routed tier
        would promise a decision the request never gets. Only the two keys the dialects agree on
        are constrained: anything else in a message stays untranslated and unread.
        """
        if value is None:
            return value
        for index, message in enumerate(value):
            if not isinstance(role := message.get("role"), str) or not role.strip():
                raise ValueError(f"messages[{index}] needs a non-empty string role")
            if (content := message.get("content")) is not None and not isinstance(content, str | list):
                raise ValueError(f"messages[{index}] content must be a string, a list of blocks, or null")
        return value

    @model_validator(mode="after")
    def _resolve_request_carrier(self) -> "AutoRouterRoutingTestRequest":
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        if self.messages is not None and not self.messages:
            raise ValueError("messages must not be empty")
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("provide exactly one of prompt or messages")
        if (self.complexity_router_config is None) == (self.capability_router_config is None):
            raise ValueError("provide exactly one router config")
        if self.messages is not None:
            return self
        return self.model_copy(
            update={  # mutable-ok: model_copy types update as a plain dict
                "messages": [  # mutable-ok: the routing hook's signature takes a list of message dicts
                    {"role": "user", "content": self.prompt}  # mutable-ok: a message is dict-shaped
                ]
            }
        )

    def wire_body(self) -> Mapping[str, object]:
        """The request kwargs a serving-path request would carry for this body.

        Every value is handed out by identity rather than copied, so the messages the routing hook
        classifies and the messages its raw-body plan-mode scan reads are one value, as they are on
        the serving path.
        """
        return MappingProxyType(
            {  # mutable-ok: MappingProxyType needs a dict to wrap
                key: value
                for key, value in (("messages", self.messages), ("system", self.system), ("tools", self.tools))
                if value is not None
            }
        )


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

ShadowEvalTargetType: TypeAlias = Literal["key", "team", "user"]

DEFAULT_SHADOW_EVAL_JUDGE_MODEL: Final[str] = "anthropic/claude-sonnet-5"

# Sample-count ceiling written on every new job: a zero-cost error loop (a shadow arm that
# fails before billing) never consumes spend budget, so it must terminate on count instead.
# A multi-router job writes one attempt row per router arm, so the valve is reached
# proportionally sooner; it is a safety valve, not a sample budget.
SHADOW_EVAL_TURN_VALVE: Final[int] = 10_000

SHADOW_EVAL_MAX_ROUTERS: Final[int] = 4


class StartShadowEvalRequest(BaseModel):
    """Start duplicating one or more targets' traffic for blind comparison against an auto-router.

    A target is a virtual key, a team, or a user; each becomes its own leg with its own
    budget and stop state. Team and user targets match on the identity every request
    carries after auth (user_api_key_team_id / user_api_key_user_id), so they cover
    JWT-authenticated traffic, which presents no virtual key at all."""

    api_key_ids: tuple[str, ...] = Field(
        default=(),
        max_length=100,
        description=(
            "Hashed virtual keys whose traffic will be shadowed. Combined with team_ids and user_ids the job "
            "needs at least one target and at most 100, which also bounds every read the job's endpoints make. "
            "Each target carries its own max_budget spend budget, so one exhausting its budget leaves the "
            "others sampling."
        ),
    )
    team_ids: tuple[str, ...] = Field(
        default=(),
        max_length=100,
        description=(
            "Teams whose traffic will be shadowed, matched on the team every authenticated request resolves "
            "to, so a team's JWT-auth and virtual-key traffic are both sampled"
        ),
    )
    user_ids: tuple[str, ...] = Field(
        default=(),
        max_length=100,
        description=(
            "Users whose traffic will be shadowed, matched on the user every authenticated request resolves "
            "to across all their teams: JWT requests carrying their subject claim and virtual keys they own"
        ),
    )
    router_name: str | None = Field(
        default=None,
        description=(
            "The auto-router under evaluation, in either direction: the single-router spelling of "
            "router_names. Provide exactly one of the two fields"
        ),
    )
    router_names: tuple[str, ...] = Field(
        default=(),
        max_length=SHADOW_EVAL_MAX_ROUTERS,
        description=(
            "The auto-routers under evaluation, at most "
            f"{SHADOW_EVAL_MAX_ROUTERS}. Every sampled request runs through every router listed and each "
            "arm is judged independently against the same real response, so routers compare head-to-head "
            "on identical traffic. More than one router requires direction 'forward'. After validation "
            "this field always carries the full deduplicated set, whichever spelling the caller used"
        ),
    )
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
        description="Percentage of each target's requests to duplicate through the router",
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
            "Per-target USD budget for the eval's own overhead, the shadow-arm and judge calls, priced with "
            "the same figures the spend pipeline bills. EACH scoped target samples until its recorded eval "
            "spend reaches this, so a job over N targets spends at most about N times max_budget; in-flight "
            "samples can overshoot the cap by one sampling cache window. Every router arm draws from the "
            "same per-target budget, so a multi-router job reaches it proportionally sooner"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_the_retired_turn_budget(cls, values: object) -> object:
        """Pydantic ignores unknown fields, so a caller still sending max_turns would
        silently run on the default dollar budget instead of the bound they asked for."""
        if isinstance(values, Mapping) and "max_turns" in values:
            raise ValueError("max_turns was replaced by max_budget, the per-target USD cap on the eval's own spend")
        return values

    @field_validator("shadow_percentage")
    @classmethod
    def _round_percentage(cls, value: float) -> float:
        return round(value, 2)

    @field_validator("api_key_ids", "team_ids", "user_ids")
    @classmethod
    def _dedupe_targets(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """A target named twice would collide with itself on the one-active-per-(target, direction) index."""
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def _at_least_one_target_at_most_hundred(self) -> "StartShadowEvalRequest":
        total: Final = len(self.api_key_ids) + len(self.team_ids) + len(self.user_ids)
        if total < 1:
            raise ValueError("at least one target is required: pass api_key_ids, team_ids, or user_ids")
        if total > 100:
            raise ValueError("at most 100 targets per job across api_key_ids, team_ids, and user_ids")
        return self

    @model_validator(mode="after")
    def _baseline_model_matches_direction(self) -> "StartShadowEvalRequest":
        if self.direction == "reverse" and self.baseline_model is None:
            raise ValueError("baseline_model is required when direction is 'reverse'")
        if self.direction == "forward" and self.baseline_model is not None:
            raise ValueError("baseline_model is only meaningful when direction is 'reverse'")
        return self

    @model_validator(mode="after")
    def _resolve_router_set(self) -> "StartShadowEvalRequest":
        """Whichever spelling the caller used, router_names leaves validation as the full
        deduplicated set, so every downstream reader consumes one field."""
        if (self.router_name is None) == (not self.router_names):
            raise ValueError("provide exactly one of router_name or router_names")
        single: Final = () if self.router_name is None else (self.router_name,)
        routers: Final = tuple(dict.fromkeys(self.router_names or single))
        if not all(name.strip() for name in routers):
            raise ValueError("router names must be non-empty strings")
        if len(routers) > 1 and self.direction == "reverse":
            raise ValueError("a reverse job evaluates one router against baseline_model; pass a single router")
        # A returned model_copy is ignored on the __init__ construction path, so the
        # normalization must land as a self attribute store to hold for every caller.
        self.router_names = routers
        return self


class ShadowEvalSlice(BaseModel):
    """Judge outcomes for one slice of a job's verdicts: a router tier, one of the
    models that served the real arm, or one scoped target (embedded on that target's
    own entry, so slices never need re-joining to a target by id)."""

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
    real_spend: float = Field(
        default=0.0,
        description=(
            "USD the real arm billed on this slice's judged turns, completion plus its own routing "
            "classifier when it routed, excluding turns litellm's response cache served for free"
        ),
    )
    shadow_spend: float = Field(
        default=0.0,
        description=(
            "USD the shadow arm billed on the same turns, completion plus its own routing classifier, "
            "excluding the judge and the same cache-served turns, so the two spends compare like for like"
        ),
    )
    cache_hit_turns: int = Field(
        default=0,
        description=(
            "Judged turns litellm's response cache served, excluded from both spends: an adopted router "
            "would be served by the same cache, so those turns cost the same either way"
        ),
    )


class ShadowEvalResult(BaseModel):
    """Stratified results of a shadow-eval job's verdicts so far."""

    by_tier: tuple[ShadowEvalSlice, ...]
    by_current_model: tuple[ShadowEvalSlice, ...] = Field(
        description=(
            "Sliced by the model that served the real arm: the keys' incumbent models in forward mode, "
            "and in reverse the models the router itself picked"
        )
    )
    by_router: tuple[ShadowEvalSlice, ...] = Field(
        default=(),
        description=(
            "One slice per router arm, grouped on the router name. Every arm of a multi-router job is "
            "judged against the same real responses over the same sampled requests, so these slices "
            "compare routers head-to-head: like-for-like win rates and spends on identical traffic. "
            "Verdicts from before arm stamping existed count toward the job's own router"
        ),
    )
    overall_shadow_win_rate_pct: float
    overall_tie_rate_pct: float
    sampled_real_spend: float = Field(
        default=0.0,
        description=(
            "USD the real arm billed across all judged turns, cache-served turns excluded. A judged turn "
            "is one (request, router arm) verdict, so a multi-router job counts the real response once per "
            "arm it was judged against; per-router comparisons read by_router"
        ),
    )
    sampled_shadow_spend: float = Field(
        default=0.0,
        description="USD the shadow arms billed across the same turns, judge excluded, like for like",
    )
    not_sampled_count: int | None = Field(
        default=None,
        description=(
            "Eligible requests the sampling dice skipped, summed over legs: the judged rows stand for "
            "judged + this many requests. None for jobs from before the funnel existed"
        ),
    )
    unjudgeable_count: int | None = Field(
        default=None,
        description="Sampled requests whose shape could not be judged (tool-final turn, empty text)",
    )
    shed_count: int | None = Field(
        default=None,
        description="Sampled requests dropped by the per-pod concurrency cap, so quiet periods are overweighted",
    )
    withheld_count: int | None = Field(
        default=None,
        description=(
            "Sampled requests the pipeline declined to spend on: no database to record into, an over-budget "
            "key or team, or the eval budget unverifiable or already reached (the in-flight burst as a job "
            "crosses max_budget lands here rather than vanishing from coverage)"
        ),
    )


class ShadowEvalJobTargetResponse(BaseModel):
    """One target a job shadows (a key, team, or user), with its own budget and stop state."""

    target_type: ShadowEvalTargetType = Field(description="What kind of entity this entry scopes")
    target_id: str = Field(description="The hashed virtual key, team id, or user id whose traffic this entry scopes")
    max_turns: int = Field(
        description=(
            "This target's sample-count ceiling: the whole budget for jobs created before max_budget "
            "existed, and the error-loop safety valve otherwise"
        )
    )
    max_budget: float | None = Field(
        default=None,
        description=(
            "This target's own USD budget for the eval's shadow and judge spend, independent of its "
            "siblings'; None on jobs created before spend budgets existed, which max_turns alone bounds"
        ),
    )
    stopped_at: datetime | None = Field(
        default=None,
        description=(
            "When this target's slot was stamped free, whether its own budget ran out, the window closed, "
            "or an operator stopped the job; status is derived, so a spent budget reads completed even "
            "while this is still unset"
        ),
    )
    attempt_count: int | None = Field(
        default=None,
        description=(
            "This target's sampled attempts so far, judged and errored alike, the same count the sampler "
            "budgets against max_turns; populated on list and detail responses. Frozen at stopped_at "
            "once the target is stamped, so in-flight attempts landing after a stop never reclassify it"
        ),
    )
    spend: float | None = Field(
        default=None,
        description=(
            "This target's recorded shadow plus judge spend in USD, the same figure the sampler budgets "
            "against max_budget; populated on list and detail responses and frozen at stopped_at "
            "exactly like attempt_count"
        ),
    )

    verdicts: "ShadowEvalSlice | None" = Field(
        default=None,
        description="This target's own judged-verdict slice; detail endpoint only, None until a turn is judged",
    )

    @property
    def budget_spent(self) -> bool:
        over_spend: Final = self.max_budget is not None and self.spend is not None and self.spend >= self.max_budget
        return over_spend or (self.attempt_count is not None and self.attempt_count >= self.max_turns)

    target_alias: str | None = Field(
        default=None,
        description=(
            "Display label resolved from the target's own row at read time: the key's alias, the team's "
            "alias, or the user's email; None when unset or deleted"
        ),
    )
    key_name: str | None = Field(
        default=None,
        description="Masked display name (sk-...) for key targets, resolved at read time; None for teams and users",
    )


class ShadowEvalJobResponse(BaseModel):
    """A shadow-eval job over one or more targets, each with its own budget and stop state;
    status is derived from stopped_by, the targets' stop and budget state, and ends_at,
    never stored, so no writer anywhere can produce an inconsistent one. Aggregate
    fields are populated by the detail endpoint only and stay None on list responses."""

    job_id: str
    targets: tuple[ShadowEvalJobTargetResponse, ...] = Field(
        min_length=1,
        description="The targets whose traffic this job evaluates, and only theirs, each with its own budget",
    )
    router_names: tuple[str, ...] = Field(
        min_length=1,
        description=(
            "Every auto-router this job runs as a shadow arm. Multi-router jobs sample one slice of "
            "traffic and judge every arm against the same real responses"
        ),
    )
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
    def router_name(self) -> str:
        """The first router, kept for callers that predate router_names; derived so the
        two fields can never disagree."""
        return self.router_names[0]

    @computed_field
    @property
    def status(self) -> ShadowEvalStatus:
        """Three recorded facts, no history-guessing: a stop is stopped_by (the migration
        backfills it for every job that displayed stopped when the column arrived, so the
        pre-column population is closed), completion is the window passing or every target
        spending its budget, and anything else is running. The all-targets-stamped fallback
        covers only stops written by pre-column pods during a rolling deploy."""
        if self.stopped_by is not None:
            return "stopped"
        if datetime.now(timezone.utc) >= (
            self.ends_at if self.ends_at.tzinfo else self.ends_at.replace(tzinfo=timezone.utc)
        ):
            return "completed"
        if all(target.budget_spent for target in self.targets):
            return "completed"
        if all(target.stopped_at is not None for target in self.targets):
            return "stopped"
        return "running"
