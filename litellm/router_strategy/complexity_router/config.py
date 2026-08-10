"""
Configuration for the Complexity Router.

Contains default keyword lists, weights, tier boundaries, and configuration classes.
All values are configurable via proxy config.yaml.
"""

from enum import Enum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from litellm.types.router import AdaptiveRouterWeights, RoutingPlugin


class ComplexityTier(str, Enum):
    """Complexity tiers for routing decisions."""

    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    REASONING = "REASONING"


TIER_SEVERITY_ORDER: Final[tuple[ComplexityTier, ...]] = (
    ComplexityTier.SIMPLE,
    ComplexityTier.MEDIUM,
    ComplexityTier.COMPLEX,
    ComplexityTier.REASONING,
)

DEFAULT_TIER_DISTANCE_PENALTY: Final[float] = 0.5

DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE: Final[int] = 3
DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS: Final[int] = 200


class KeywordTierRule(BaseModel):
    """A deterministic override: if any keyword matches, route to this tier."""

    keywords: list[str] = Field(
        min_length=1,
        description="Keywords/phrases that trigger this rule (lexical or semantic match)",
    )
    tier: ComplexityTier = Field(
        description="Tier to route to when this rule matches",
    )

    @model_validator(mode="after")
    def _normalize_keywords(self) -> "KeywordTierRule":
        # Strip and drop blank keywords. An empty/whitespace keyword is a routing foot-gun:
        # _keyword_matches treats "" / " " as a substring that matches essentially every
        # prompt, so a single stray blank would silently force this rule's tier for all
        # traffic. Require at least one real keyword to remain.
        cleaned: Final = [stripped for keyword in self.keywords if (stripped := keyword.strip())]
        if not cleaned:
            raise ValueError("keyword_tier_rules entries must contain at least one non-empty keyword")
        self.keywords = cleaned
        return self


class ReminderMarkerPair(BaseModel):
    """One open/close delimiter pair a harness wraps injected context in.

    Normalizing here rather than at the scan is what makes matching case-insensitive: markers reach
    the scan already lowered, so it lowercases only the haystack and never the needles. Stripping
    keeps YAML indentation whitespace from becoming part of the delimiter.
    """

    open: str = Field(description="Opening delimiter, e.g. '<system-reminder>'")
    close: str = Field(description="Closing delimiter, e.g. '</system-reminder>'")

    @model_validator(mode="after")
    def _normalize(self) -> "ReminderMarkerPair":
        open_marker: Final = self.open.strip().lower()
        close_marker: Final = self.close.strip().lower()
        if not open_marker or not close_marker:
            raise ValueError("reminder_markers entries must not be blank")
        if open_marker == close_marker:
            raise ValueError("reminder_markers open and close must be different strings")
        self.open = open_marker
        self.close = close_marker
        return self


# ─── Default Keyword Lists ───
# Note: Keywords should be full words/phrases to avoid substring false positives.
# The matching logic uses word boundary detection for single-word keywords.

DEFAULT_CODE_KEYWORDS: Final[list[str]] = [
    "function",
    "class",
    "def",
    "const",
    "let",
    "var",
    "import",
    "export",
    "return",
    "async",
    "await",
    "try",
    "catch",
    "exception",
    "error",
    "debug",
    "api",
    "endpoint",
    "request",
    "response",
    "database",
    "sql",
    "query",
    "schema",
    "algorithm",
    "implement",
    "refactor",
    "optimize",
    "python",
    "javascript",
    "typescript",
    "java",
    "rust",
    "golang",
    "react",
    "vue",
    "angular",
    "node",
    "docker",
    "kubernetes",
    "git",
    "commit",
    "merge",
    "branch",
    "pull request",
]

DEFAULT_REASONING_KEYWORDS: Final[list[str]] = [
    "step by step",
    "think through",
    "let's think",
    "reason through",
    "analyze this",
    "break down",
    "explain your reasoning",
    "show your work",
    "chain of thought",
    "think carefully",
    "consider all",
    "evaluate",
    "pros and cons",
    "compare and contrast",
    "weigh the options",
    "logical",
    "deduce",
    "infer",
    "conclude",
]

DEFAULT_TECHNICAL_KEYWORDS: Final[list[str]] = [
    "architecture",
    "distributed",
    "scalable",
    "microservice",
    "machine learning",
    "neural network",
    "deep learning",
    "encryption",
    "authentication",
    "authorization",
    "performance",
    "latency",
    "throughput",
    "benchmark",
    "concurrency",
    "parallel",
    "threading",
    "memory",
    "cpu",
    "gpu",
    "optimization",
    "protocol",
    "tcp",
    "http",
    "grpc",
    "websocket",
    "container",
    "orchestration",
    # Note: "async", "kubernetes", "docker" are in DEFAULT_CODE_KEYWORDS
]

DEFAULT_ESCALATION_KEYWORDS: Final[list[str]] = ["LITELLM ESCALATE"]


DEFAULT_SIMPLE_KEYWORDS: Final[list[str]] = [
    "what is",
    "what's",
    "define",
    "definition of",
    "who is",
    "who was",
    "when did",
    "when was",
    "where is",
    "where was",
    "how many",
    "how much",
    "yes or no",
    "true or false",
    "simple",
    "brief",
    "short",
    "quick",
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "goodbye",
    "bye",
    "okay",
    # Note: "ok" removed due to false positives (matches "token", "book", etc.)
]


# ─── Default Dimension Weights ───

DEFAULT_DIMENSION_WEIGHTS: Final[dict[str, float]] = {
    "tokenCount": 0.10,  # Reduced - length is less important than content
    "codePresence": 0.30,  # High - code requests need capable models
    "reasoningMarkers": 0.25,  # High - explicit reasoning requests
    "technicalTerms": 0.25,  # High - technical content matters
    "simpleIndicators": 0.05,  # Low - don't over-penalize simple patterns
    "multiStepPatterns": 0.03,
    "questionComplexity": 0.02,
}


# ─── Default Tier Boundaries ───

DEFAULT_TIER_BOUNDARIES: Final[dict[str, float]] = {
    "simple_medium": 0.15,  # Lower threshold to catch more MEDIUM cases
    "medium_complex": 0.35,  # Lower threshold to catch technical COMPLEX cases
    "complex_reasoning": 0.60,  # Reasoning tier reserved for explicit reasoning markers
}


# ─── Default Token Thresholds ───

DEFAULT_TOKEN_THRESHOLDS: Final[dict[str, int]] = {
    "simple": 15,  # Only very short prompts (<15 tokens) are penalized
    "complex": 400,  # Long prompts (>400 tokens) get complexity boost
}


# ─── Default Tier to Model Mapping ───

DEFAULT_TIER_MODELS: Final[dict[str, str]] = {
    "SIMPLE": "gpt-4o-mini",
    "MEDIUM": "gpt-4o",
    "COMPLEX": "claude-sonnet-4-20250514",
    "REASONING": "claude-sonnet-4-20250514",
}


class ClassifierLLMConfig(BaseModel):
    """Configuration for the LLM-based complexity classifier."""

    model: str = Field(
        description="Model name (from the router's model_list) to call for classification",
    )
    timeout_ms: int = Field(
        default=3000,
        description="Timeout budget for the classification call, in milliseconds",
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Replaces the built-in complexity rubric as the classifier's entire system role. When set, "
            "neither the default rubric nor the context-window closing line is appended, so the prompt "
            "owns the whole taxonomy and the tier names SIMPLE/MEDIUM/COMPLEX/REASONING become whatever "
            "buckets it defines: a prompt that classifies data sensitivity routes on that instead of on "
            "difficulty. Two consequences of full replacement. The default rubric's closing paragraph is "
            "the classifier's prompt-injection defense, telling it that the caller's quoted system prompt "
            "and prior turns are material to judge and never instructions; a replacement that omits it "
            "lets a caller ask for a tier and get it. And the heuristic fallback still scores complexity, "
            "so a router on some other taxonomy wants classifier_fallback='default_model'. Leave unset "
            "for the built-in rubric. Only applies when classifier_type is 'llm'."
        ),
    )

    @field_validator("system_prompt")
    @classmethod
    def _reject_blank_system_prompt(cls, value: str | None) -> str | None:
        # A blank string is a misconfiguration, not a request for the default: it would send an
        # empty system role and leave the classifier with no rubric at all. None means default.
        if value is not None and not value.strip():
            raise ValueError("classifier_llm_config.system_prompt must be non-empty; omit it to use the default rubric")
        return value


class ComplexityRouterConfig(BaseModel):
    """Configuration for the ComplexityRouter."""

    # string = pin; list = random pick when adaptive=False, soft-floor home pool when adaptive=True
    tiers: dict[str, str | list[str]] = Field(
        default_factory=lambda: DEFAULT_TIER_MODELS.copy(),
        description=(
            "Mapping of complexity tiers to a model or model pool. "
            "A list is randomly picked from when adaptive=False, and used as a soft-floor home pool when adaptive=True"
        ),
    )

    tier_labels: dict[ComplexityTier, str] = Field(
        default_factory=dict,
        description=(
            "Display names for the complexity tiers, so a deployment can use its own vocabulary "
            "(e.g. Cheap/Standard/Premium/Deep) in the dashboard, spend logs, and the LLM classifier "
            "rubric. Purely operator-facing: config keys stay canonical (tiers, keyword_tier_rules[].tier, "
            "tier_boundaries), API callers never see these names, and the heuristic scorer never reads them. "
            "Unlisted tiers keep their canonical name. Partial maps are allowed."
        ),
    )

    # Tier boundaries (normalized scores)
    tier_boundaries: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_TIER_BOUNDARIES.copy(),
        description=(
            "Score boundaries between tiers. These keys (simple_medium, medium_complex, complex_reasoning) "
            "name the gaps between the default tier names and are not renameable by tier_labels; they are "
            "scorer knobs persisted by name on every routing decision"
        ),
    )

    # Token count thresholds
    token_thresholds: dict[str, int] = Field(
        default_factory=lambda: DEFAULT_TOKEN_THRESHOLDS.copy(),
        description="Token count thresholds for simple/complex classification",
    )

    # Dimension weights
    dimension_weights: dict[str, float] = Field(
        default_factory=lambda: DEFAULT_DIMENSION_WEIGHTS.copy(),
        description="Weights for each scoring dimension",
    )

    # Keyword lists (overridable)
    code_keywords: list[str] | None = Field(
        default=None,
        description="Keywords indicating code-related content",
    )
    reasoning_keywords: list[str] | None = Field(
        default=None,
        description="Keywords indicating reasoning-required content",
    )
    technical_keywords: list[str] | None = Field(
        default=None,
        description="Keywords indicating technical content",
    )
    custom_technical_keywords: list[str] | None = Field(
        default=None,
        description=(
            "Domain-specific technical keywords appended to the effective base list "
            "(technical_keywords if set, otherwise DEFAULT_TECHNICAL_KEYWORDS). "
            "Order is preserved; duplicates are removed case-insensitively against "
            "the base list and within this list."
        ),
    )
    simple_keywords: list[str] | None = Field(
        default=None,
        description="Keywords indicating simple/basic queries",
    )

    # Default model if scoring fails
    default_model: str | None = Field(
        default=None,
        description="Default model to use if tier cannot be determined",
    )

    return_raw_model_name: bool = Field(
        default=False,
        description=(
            "Return the resolved raw model name in the response model field instead of "
            "the client-requested complexity-router alias"
        ),
    )

    # Classifier strategy
    classifier_type: Literal["heuristic", "llm"] = Field(
        default="heuristic",
        description="Classification strategy: local regex/keyword scoring, or an LLM call",
    )
    classifier_llm_config: ClassifierLLMConfig | None = Field(
        default=None,
        description="Configuration for the LLM classifier; required when classifier_type is 'llm'",
    )

    classifier_fallback: Literal["heuristic", "default_model"] = Field(
        default="heuristic",
        description=(
            "What classifies the request when the LLM classifier errors, times out, or returns an "
            "unparseable response. 'heuristic' runs the local complexity scorer, which is right when the "
            "classifier grades complexity too. 'default_model' skips scoring and routes to default_model, "
            "which is what a classifier on some other taxonomy wants: a prompt that grades data "
            "sensitivity has no use for a complexity score, and scoring one produces a tier unrelated to "
            "what the operator configured. Requires default_model when set to 'default_model'. Only "
            "applies when classifier_type is 'llm'."
        ),
    )

    classifier_context_window_size: int = Field(
        default=DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
        ge=0,
        description=(
            "Number of prior user turns (tool output and harness reminders excluded) to include as context "
            "in the LLM classifier prompt, so a follow-up like 'now do the same for the streaming path' is "
            "classified against what it refers to. Counts turns of both roles when "
            "classifier_context_include_assistant_turns is enabled. These turns are sent to the classifier "
            "model, which may "
            "be a different deployment or provider than the routed completion model; that call already "
            "carries the current user ask and the caller's system prompt in full. Set to 0 to send neither "
            "prior turns nor any conversation context beyond the current ask. Only applies when "
            "classifier_type is 'llm'."
        ),
    )
    classifier_context_per_turn_chars: int = Field(
        default=DEFAULT_CLASSIFIER_CONTEXT_PER_TURN_CHARS,
        gt=0,
        description=(
            "Maximum character length for each prior turn's text in the classifier context window. "
            "Turns exceeding this are truncated. Only applies when classifier_type is 'llm'."
        ),
    )
    classifier_context_include_assistant_turns: bool = Field(
        default=False,
        description=(
            "Include assistant turns in the classifier context window, so difficulty stated by the "
            "model rather than by the user stays visible: a plan the assistant calls complex, which "
            "the user approves with 'yes', is classified on the work being approved instead of on the "
            "word 'yes'. When enabled, classifier_context_window_size counts the last N turns of the "
            "conversation across both roles rather than the last N user turns, and assistant text is "
            "sent to the classifier model, which may be a different deployment or provider than the "
            "routed completion model. Assistant replies share classifier_context_per_turn_chars with "
            "user turns, so raise it if replies are truncated before the part that carries the "
            "difficulty. Off by default because enabling it shifts tier decisions, and therefore "
            "spend, for an already-deployed router. Only applies when classifier_type is 'llm'."
        ),
    )

    adaptive: bool = Field(
        default=False,
        description="Enable adaptive bandit selection with soft complexity floors",
    )
    adaptive_weights: AdaptiveRouterWeights = Field(
        default_factory=lambda: AdaptiveRouterWeights(quality=0.3, cost=0.7),
        description="Quality vs cost weights for adaptive selection (used when adaptive=True)",
    )
    tier_distance_penalty: float = Field(
        default=DEFAULT_TIER_DISTANCE_PENALTY,
        ge=0.0,
        description="Score penalty per tier-step away from the classified tier when adaptive=True",
    )
    adaptive_eligible: Literal["all", "classified_tier"] = Field(
        default="all",
        description=(
            "When adaptive=True: 'all' scores every pool model with a tier-distance penalty (soft floors); "
            "'classified_tier' Thompson-samples only inside the classified tier's pool"
        ),
    )

    escalation_keywords: list[str] | None = Field(
        default=None,
        description=(
            "Case-sensitive phrases a user can include to force a bump to the next-higher "
            "complexity tier when they aren't satisfied with results (they can force a stronger "
            "model, but not choose which one). Defaults to ['LITELLM ESCALATE'] when unset; "
            "set to an empty list to disable."
        ),
    )

    # Deterministic keyword -> tier overrides, evaluated before weighted scoring
    keyword_tier_rules: list[KeywordTierRule] | None = Field(
        default=None,
        description="Rules that force a specific tier when their keywords match the prompt",
    )

    # Semantic (embedding) matching for keyword_tier_rules instead of literal text matching
    semantic_keyword_matching: bool = Field(
        default=False,
        description="Match keyword_tier_rules by embedding similarity instead of literal text",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Embedding model (LiteLLM model name) used when semantic_keyword_matching is enabled",
    )
    match_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a semantic keyword match",
    )

    # Session affinity: pin the first turn's routed model for the rest of the session
    session_affinity: bool = Field(
        default=False,
        description=(
            "When True and a session_id is resolvable on the request, pin the model chosen on the "
            "session's first turn and reuse it for every later turn, skipping re-classification. "
            "Off by default so every turn is classified on its own merits and routed to the cheapest "
            "adequate tier. Set True to keep a multi-turn session on one model, which preserves "
            "provider prompt caches and avoids cross-model conversation-history errors. Always "
            "implies the deployment pin regardless of deployment_affinity: the session sticks to "
            "one deployment of the pinned model, since freezing the model while re-shuffling its "
            "deployments would still go cache-cold."
        ),
    )
    deployment_affinity: bool = Field(
        default=True,
        description=(
            "When True and a session_id is resolvable on the request, pin the deployment chosen "
            "inside each routed model group and reuse it whenever the session returns to that "
            "group, without pinning which group the session routes to. Independent of "
            "session_affinity, which pins the model group instead (and always carries this "
            "deployment pin with it): with session_affinity off, "
            "every turn is still classified on its own merits while a session that escalates to a "
            "stronger tier and comes back still lands on the deployment it used before, which is "
            "what keeps a provider prompt cache warm. Pins are held per model group, so switching "
            "tiers does not disturb the pin left behind in the previous group. On by default "
            "because re-shuffling a conversation across deployments of the same model discards "
            "that cache for no benefit; set False to keep every turn load-balanced across the "
            "group, which is what a deployment set with tight per-deployment rate limits wants. "
            "Inert when no session_id is resolvable, since there is nothing to key a pin on, and "
            "suppressed when plugins are configured, for the same reason session_affinity is."
        ),
    )
    session_affinity_ttl_seconds: int = Field(
        default=3600,
        gt=0,
        description=(
            "TTL for the session affinity pin; refreshed on every cache hit. Bounds both the "
            "session_affinity model pin and the deployment_affinity deployment pin, so it measures "
            "idle time for the session's routing decisions rather than total session length"
        ),
    )

    plugins: list[RoutingPlugin] | None = Field(
        default=None,
        description="RoutingPlugin instances that narrow the classified tier's candidate models before selection",
    )

    reminder_markers: tuple[ReminderMarkerPair, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Override the delimiter pairs used to recognize and strip harness-injected reminder "
            "blocks before classification. A harness that wraps injected context differently per "
            "agent type (main, subagent, cron) lists every pair it emits. Replaces, rather than "
            "adds to, the built-in default of ('<system-reminder>', '</system-reminder>'), so a "
            "harness that also emits that pair lists it too. Matching is case-insensitive."
        ),
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)  # Allow additional fields

    @field_validator("tiers", mode="before")
    @classmethod
    def _coerce_tier_values(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        coerced: Final[dict[str, object]] = {}
        for key, item in value.items():
            if isinstance(item, str):
                coerced[key] = item
            elif isinstance(item, (list, tuple)):
                coerced[key] = list(item)
            else:
                coerced[key] = item
        return coerced

    @field_validator("escalation_keywords")
    @classmethod
    def _normalize_escalation_keywords(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [stripped for keyword in value if (stripped := keyword.strip())]

    @model_validator(mode="after")
    def _validate_llm_classifier_config(self) -> "ComplexityRouterConfig":
        if self.classifier_type == "llm" and self.classifier_llm_config is None:
            raise ValueError("classifier_llm_config is required when classifier_type is 'llm'")
        return self

    @model_validator(mode="after")
    def _validate_adaptive_pools(self) -> "ComplexityRouterConfig":
        if not self.adaptive:
            return self
        normalized = {tier: (models if isinstance(models, list) else [models]) for tier, models in self.tiers.items()}
        if not any(normalized.values()):
            raise ValueError("adaptive=True requires at least one non-empty tier pool")
        empty: Final = [tier for tier, models in normalized.items() if not models]
        if empty:
            raise ValueError(f"adaptive=True tier pools must be non-empty; empty tiers: {empty}")
        self.tiers = normalized
        return self

    @model_validator(mode="after")
    def _validate_semantic_matching(self) -> "ComplexityRouterConfig":
        if not self.semantic_keyword_matching:
            return self
        if not self.embedding_model:
            raise ValueError("embedding_model is required when semantic_keyword_matching is enabled")
        if not self.keyword_tier_rules:
            raise ValueError("keyword_tier_rules must be non-empty when semantic_keyword_matching is enabled")
        return self

    @model_validator(mode="after")
    def _validate_tier_labels(self) -> "ComplexityRouterConfig":
        if not self.tier_labels:
            return self
        blank: Final = tuple(sorted(tier.value for tier, label in self.tier_labels.items() if not label.strip()))
        if blank:
            raise ValueError(f"tier_labels values must be non-empty; blank labels for tiers: {', '.join(blank)}")
        shadowed: Final = tuple(
            sorted(
                f"{tier.value} -> {label.strip()}"
                for tier, label in self.tier_labels.items()
                if label.strip().upper() in ComplexityTier.__members__ and label.strip().upper() != tier.value
            )
        )
        if shadowed:
            raise ValueError(
                "tier_labels values must not reuse another tier's canonical name, which would make logs "
                f"and the classifier rubric ambiguous: {', '.join(shadowed)}"
            )
        labeled: Final = self.labeled_tiers()
        folded_labels: Final = tuple(label.casefold() for _, label in labeled)
        duplicated: Final = tuple(
            " and ".join(tier.value for tier, label in labeled if label.casefold() == folded)
            for position, folded in enumerate(folded_labels)
            if folded_labels.count(folded) > 1 and folded_labels.index(folded) == position
        )
        if duplicated:
            raise ValueError(
                f"tier_labels values must be unique across tiers; shared labels for: {'; '.join(duplicated)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_plugins_adaptive_combo(self) -> "ComplexityRouterConfig":
        if self.plugins and self.adaptive:
            raise ValueError(
                "plugins and adaptive=True cannot both be set: adaptive's bandit selection doesn't yet "
                "consume plugin-narrowed candidate pools. Disable adaptive or remove plugins."
            )
        return self

    def tier_label(self, tier: ComplexityTier) -> str:
        """Operator-facing display name for a tier, falling back to its canonical name."""
        return self.tier_labels.get(tier, "").strip() or tier.value

    def labeled_tiers(self) -> tuple[tuple[ComplexityTier, str], ...]:
        """Every tier paired with its display name, in ascending severity order."""
        return tuple((tier, self.tier_label(tier)) for tier in TIER_SEVERITY_ORDER)

    def tier_for_label(self, label: str) -> ComplexityTier | None:
        """Resolve a display name back to its tier, case-insensitively, then canonical names."""
        folded: Final = label.strip().casefold()
        labeled: Final = self.labeled_tiers()
        return next(
            (tier for tier, tier_label in labeled if tier_label.casefold() == folded),
            next((tier for tier in TIER_SEVERITY_ORDER if tier.value.casefold() == folded), None),
        )


# Combined default config
DEFAULT_COMPLEXITY_CONFIG: Final = ComplexityRouterConfig()
