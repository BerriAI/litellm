"""
Configuration for the Complexity Router.

Contains default keyword lists, weights, tier boundaries, and configuration classes.
All values are configurable via proxy config.yaml.
"""

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, field_serializer, field_validator, model_validator

from litellm.types.router import AdaptiveRouterWeights, ClassifierPlugin, RoutingPlugin


class ComplexityTier(str, Enum):
    """Complexity tiers for routing decisions."""

    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    REASONING = "REASONING"


class ClassificationRubric(str, Enum):
    """Which calibration examples, and for BUSINESS which tier criteria, the built-in classifier rubric carries."""

    LEGACY = "legacy"
    AGENTIC = "agentic"
    CHAT = "chat"
    BUSINESS = "business"


# Unset means LEGACY, so upgrading never moves an existing router's tier decisions or its bill. A
# router created through the dashboard is stamped with a preset at create time, which is how new
# routers get the calibrated rubric without changing what is already running.
DEFAULT_CLASSIFICATION_RUBRIC: Final[ClassificationRubric] = ClassificationRubric.LEGACY

# The classifier_type values that can call classifier_llm_config.model. Every consumer asking
# "is the classifier model a real dependency of this router" resolves it here, including the ones
# that only hold the raw config mapping and cannot reach ComplexityRouterConfig.uses_llm_classifier.
LLM_CLASSIFIER_TYPES: Final[frozenset[str]] = frozenset({"llm", "heuristic_first"})


TIER_SEVERITY_ORDER: Final[tuple[ComplexityTier, ...]] = (
    ComplexityTier.SIMPLE,
    ComplexityTier.MEDIUM,
    ComplexityTier.COMPLEX,
    ComplexityTier.REASONING,
)

DEFAULT_TIER_DISTANCE_PENALTY: Final[float] = 0.5

DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE: Final[int] = 3
DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS: Final[int] = 8000


class KeywordTierRule(BaseModel):
    """A deterministic override: if any keyword matches, route to this tier."""

    keywords: list[str] = Field(
        min_length=1,
        description="Keywords/phrases that trigger this rule (lexical or semantic match)",
    )
    tier: str = Field(
        description=(
            "Tier to route to when this rule matches: a built-in tier name, or with "
            "tier_definitions set, one of the defined tier names"
        ),
    )

    @field_validator("tier", mode="before")
    @classmethod
    def _coerce_tier(cls, value: object) -> object:
        if isinstance(value, ComplexityTier):
            return value.value
        if isinstance(value, str):
            return value.strip()
        return value

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


MAX_TIER_DEFINITIONS: Final[int] = 8
MAX_TIER_NAME_CHARS: Final[int] = 64
MAX_TIER_DESCRIPTION_CHARS: Final[int] = 500
MAX_CLASSIFICATION_PROMPT_CHARS: Final[int] = 2000


class TierDefinition(BaseModel):
    """An operator-defined tier: the name the LLM classifier must return and its rubric description."""

    name: str = Field(
        description="Tier name; becomes a value the LLM classifier can return and a key of `tiers`",
    )
    description: str | None = Field(
        default=None,
        description=(
            "What belongs in this tier; rendered as this tier's bullet in the classifier rubric. "
            "Required unless the name is a built-in tier (SIMPLE/MEDIUM/COMPLEX/REASONING), which "
            "inherits the built-in criteria when omitted"
        ),
    )

    @model_validator(mode="after")
    def _normalize(self) -> "TierDefinition":
        name: Final = self.name.strip()
        description: Final = (self.description.strip() or None) if self.description is not None else None
        if not name:
            raise ValueError("tier_definitions entries must have a non-empty name")
        if len(name) > MAX_TIER_NAME_CHARS:
            raise ValueError(
                f"tier_definitions name {name[:MAX_TIER_NAME_CHARS]!r}... exceeds {MAX_TIER_NAME_CHARS} characters"
            )
        if description is not None and len(description) > MAX_TIER_DESCRIPTION_CHARS:
            raise ValueError(
                f"tier_definitions description for {name!r} exceeds {MAX_TIER_DESCRIPTION_CHARS} characters"
            )
        if description is None and name.upper() not in ComplexityTier.__members__:
            raise ValueError(
                f"tier_definitions entry {name!r} must have a description: only the built-in tiers "
                "(SIMPLE, MEDIUM, COMPLEX, REASONING) carry one the rubric can inherit"
            )
        rendered_on_one_line: Final = (name, description or "")
        if any("\n" in part or "\r" in part for part in rendered_on_one_line):
            raise ValueError(
                f"tier_definitions entry {name!r} must not contain newlines; the rubric renders one line per tier"
            )
        self.name = name
        self.description = description
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


class ComplexityTierModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    litellm_params: Annotated[Mapping[str, object], SkipValidation()] = Field(
        default_factory=lambda: MappingProxyType({})
    )

    @field_validator("litellm_params", mode="before")
    @classmethod
    def _freeze_litellm_params(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(dict(value))

    @field_serializer("litellm_params")
    def _serialize_litellm_params(self, value: Mapping[str, object]) -> Mapping[str, object]:
        return dict(value)  # mutable-ok: Pydantic JSON serialization requires a concrete mapping


def _normalize_tier_entries(
    raw_value: object,
    tier: str,
) -> tuple[str | list[str], tuple[ComplexityTierModel, ...]]:
    raw_entries: Final = raw_value if isinstance(raw_value, (list, tuple)) else (raw_value,)
    entries: Final = tuple(
        ComplexityTierModel(model_name=entry) if isinstance(entry, str) else ComplexityTierModel.model_validate(entry)
        for entry in raw_entries
    )
    model_names: Final = tuple(entry.model_name for entry in entries)
    if len(model_names) != len(frozenset(model_names)):
        raise ValueError(f"tier {tier} contains duplicate model_name values; each pool entry needs distinct parameters")
    normalized: Final = (
        entries[0].model_name
        if not isinstance(raw_value, (list, tuple))
        else list(model_names)  # mutable-ok: config.tiers must preserve its existing list contract
    )
    return normalized, entries


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

# Verified against Claude Code 2.1.233 wire captures and vscode-copilot-chat source
# (agentPrompt.tsx / planAgentProvider.ts). These are client-owned strings that drift with
# client releases; operators extend coverage via plan_mode_patterns rather than editing these.
PLAN_MODE_TAIL_SENTINELS: Final[tuple[str, ...]] = (
    "Plan mode is active",
    "Plan mode still active",
)
PLAN_MODE_SYSTEM_SENTINELS: Final[tuple[str, ...]] = ('You are currently running in "Plan" mode.',)
PLAN_MODE_TOOL_NAME: Final[str] = "exit_plan_mode"


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
    classification_rubric: ClassificationRubric | None = Field(
        default=None,
        description=(
            "Which calibration examples the built-in rubric carries. 'agentic' anchors routine installs, builds, "
            "multi-file edits, and standard debugging at MEDIUM, so ordinary engineering does not route to the "
            "most expensive tier; it suits agent, terminal, and coding-assistant traffic as well as mixed "
            "traffic. 'chat' omits those engineering anchors, for a deployment serving only conversational "
            "traffic. 'business' carries business/sales anchors and business-flavored tier criteria that keep "
            "routine drafting and summarizing off the expensive tiers and reserve the top tier for committing to "
            "decisions under tradeoffs; it suits sales, support, and go-to-market traffic. Every preset keeps the "
            "same four tiers, so this moves where the boundary sits without changing the taxonomy. Leave unset "
            "for 'legacy', the rubric as it shipped before calibration examples "
            "existed, so an existing router's tier decisions and spend do not move on upgrade. Mutually exclusive "
            "with system_prompt, which replaces the rubric this would select. Only applies when classifier_type "
            "is 'llm'."
        ),
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

    @model_validator(mode="after")
    def _reject_rubric_with_system_prompt(self) -> "ClassifierLLMConfig":
        # A custom prompt is the classifier's whole system role, so a preset set alongside it would never
        # reach the wire. Rejecting it beats honoring one of two settings the operator asked for.
        #
        # None, not model_fields_set, is what marks the preset unchosen: this model is dumped and
        # re-validated in place (see /auto_router/test_routing), and a dump re-states every field, so
        # keying on fields_set would reject on the second pass what it accepted on the first.
        if self.system_prompt is not None and self.classification_rubric is not None:
            raise ValueError(
                "classifier_llm_config.classification_rubric and system_prompt are mutually exclusive: system_prompt replaces "
                "the built-in rubric the preset would select. Drop one."
            )
        return self


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
    tier_model_configs: Mapping[str, tuple[ComplexityTierModel, ...]] = Field(
        default_factory=dict,
    )

    tier_definitions: tuple[TierDefinition, ...] | None = Field(
        default=None,
        description=(
            "Operator-defined tier set replacing the built-in SIMPLE/MEDIUM/COMPLEX/REASONING. "
            "Each entry's name becomes a value the LLM classifier can return and its description "
            "becomes that tier's rubric bullet; entries named after a built-in tier may omit the "
            "description and inherit the built-in criteria. List order is ascending severity and "
            "decides which tier wins when several keyword_tier_rules match. Requires classifier_type "
            "'llm' or 'custom', a fallback_tier, and `tiers` keys matching the defined names exactly. Escalation, "
            "adaptive selection, session affinity, plugins, tier_labels, and the calibration-example "
            "rubric presets are unavailable with a custom tier set: the first four are built on the "
            "built-in tier ladder, and the last two rename or exemplify tiers the set replaces."
        ),
    )
    fallback_tier: str | None = Field(
        default=None,
        description=(
            "Tier routed to when the LLM classifier fails (timeout, provider error, or an "
            "unparseable reply). Required with tier_definitions and must name a defined tier; "
            "the heuristic scorer cannot produce custom tiers, so this replaces the heuristic "
            "fallback for custom tier sets."
        ),
    )
    classification_prompt: str | None = Field(
        default=None,
        description=(
            "Replaces the opening instructions of the LLM classifier rubric (the judging-criteria "
            "prose) for a custom tier set. The per-tier bullets and the trust-boundary paragraph "
            "telling the classifier to ignore tier requests embedded in quoted caller text are "
            "always appended after it and cannot be overridden. Requires tier_definitions; a "
            "built-in-tier router customizes its prompt via classifier_llm_config.system_prompt "
            "or classification_rubric instead."
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

    reasoning_override_min_score: float | None = Field(
        default=None,
        description=(
            "Minimum weighted score a request must reach before 2+ reasoning markers may promote it to the "
            "reasoning tier. Unset tracks tier_boundaries.simple_medium, so the override never rescues a "
            "request the scorer placed in the cheapest tier; 0 restores the unconditional override"
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
    classifier_type: Literal["heuristic", "llm", "custom", "heuristic_first"] = Field(
        default="heuristic",
        description=(
            "Classification strategy: local regex/keyword scoring, an LLM call, a custom classifier "
            "plugin, or 'heuristic_first', which scores locally and only pays for the LLM classifier "
            "when the local scorer does not confidently land a cheap tier"
        ),
    )
    classifier_llm_config: ClassifierLLMConfig | None = Field(
        default=None,
        description="Configuration for the LLM classifier; required when classifier_type is 'llm' or 'heuristic_first'",
    )
    heuristic_first_max_tier: str | None = Field(
        default=None,
        description=(
            "The highest tier the local scorer may decide on its own; required when classifier_type is "
            "'heuristic_first' and rejected otherwise. A request whose heuristic tier is at or below this "
            "one skips the LLM classifier and routes straight to that heuristic tier, so the classifier "
            "call is only paid for on traffic the scorer could not place cheaply. The scorer must also "
            "have produced at least one signal: a prompt where no dimension fired scores 0.0 and would "
            "otherwise land SIMPLE by default rather than by evidence, which is how a chained router "
            "would silently send unclassified traffic to the cheapest model. Names a built-in tier, and "
            "may not name the highest one, since that would make the LLM classifier unreachable."
        ),
    )
    classifier_plugin: ClassifierPlugin | None = Field(
        default=None,
        description=(
            "Custom classifier deciding the tier; required when classifier_type is 'custom'. In the proxy "
            "config, a dotted path to a ClassifierPlugin instance (resolved at startup, like plugins). Its "
            "classify(context) receives the request messages and metadata (caller identity included) and "
            "returns the name of the tier to route to, or None to decline and let classifier_fallback decide."
        ),
    )
    classifier_plugin_timeout_ms: int = Field(
        default=3000,
        gt=0,
        description=(
            "Timeout budget for the classifier plugin call, in milliseconds. On expiry the fallback "
            "path decides the tier. Only applies when classifier_type is 'custom'."
        ),
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
            "applies when classifier_type is 'llm', 'custom', or 'heuristic_first'."
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
    classifier_context_budget_chars: int = Field(
        default=DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS,
        ge=0,
        description=(
            "Maximum characters of prior-turn text quoted to the LLM classifier, across the whole "
            "context window, per classification call. Turns are taken newest first and quoted whole "
            "while they fit, so a conversation small enough to quote entirely is never cut; once the "
            "budget runs out the older turns are dropped whole and only the turn straddling the "
            "boundary is truncated, into whatever space is left. The current ask and the caller's "
            "system prompt sit outside this budget and are always sent in full, as does the numbering "
            "each quoted turn carries. A budget under 120 leaves no room to quote a turn and "
            "suppresses the block; set classifier_context_window_size to 0 to turn context off "
            "deliberately. Only applies when classifier_type is 'llm'."
        ),
    )
    classifier_context_per_turn_chars: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional cap on each individual prior turn's text, applied before "
            "classifier_context_budget_chars bounds the block. Unset by default, so one long turn may "
            "spend the whole budget, which is usually what a follow-up needs; set it when no single "
            "turn should dominate the context the classifier sees. A capped turn keeps its opening "
            "and its ending with the middle elided. Only applies when classifier_type is 'llm'."
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
            "routed completion model. Assistant replies spend classifier_context_budget_chars "
            "alongside user turns, so raise it if the oldest turns stop being quoted once replies "
            "join the window. Off by default because enabling it shifts tier decisions, and therefore "
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

    plan_mode_min_tier: str | None = Field(
        default=None,
        description=(
            "When set, requests carrying a coding-agent plan-mode sentinel (Claude Code plan "
            "mode, VS Code Copilot Plan mode, Copilot CLI's exit_plan_mode tool) are routed to "
            "at least this tier: the classified tier still wins when it is higher, and the "
            "floor also overrides a session-affinity pin to a lower tier for exactly the turns "
            "carrying the sentinel, without rewriting the pin -- the first turn after plan mode "
            "exits routes as if plan mode had never happened. Names a built-in tier, or with "
            "tier_definitions set, one of the defined tier names (list order is ascending "
            "severity, same as keyword_tier_rules). Unset disables detection entirely. The "
            "sentinels ride in client-injected prompt text, so a caller who pastes one can "
            "spend up to this tier's models -- never down, and never outside the configured "
            "pools."
        ),
    )
    plan_mode_patterns: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Additional case-sensitive literal sentinels that mark a request as plan mode, on "
            "top of the built-in Claude Code and Copilot ones. For clients whose plan-mode "
            "wording the built-ins don't cover, or after a client release changes its strings."
        ),
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

    @model_validator(mode="before")
    @classmethod
    def _normalize_tier_model_configs(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        raw_tiers: Final = value.get("tiers")
        if not isinstance(raw_tiers, dict):
            return value
        existing_configs: Final = value.get("tier_model_configs")
        normalized_entries: Final = MappingProxyType(
            {tier: _normalize_tier_entries(raw_value, tier) for tier, raw_value in raw_tiers.items()}
        )
        normalized_tiers: Final = MappingProxyType(
            {tier: normalized for tier, (normalized, _) in normalized_entries.items()}
        )
        incoming_params: Final = (
            MappingProxyType(
                {
                    (tier, entry.model_name): entry.litellm_params
                    for tier, entries in existing_configs.items()
                    for entry in (ComplexityTierModel.model_validate(item) for item in entries)
                }
            )
            if isinstance(existing_configs, dict)
            else MappingProxyType({})
        )
        tier_model_configs: Final = MappingProxyType(
            {
                tier: tuple(
                    entry.model_copy(
                        update=MappingProxyType(
                            {
                                "litellm_params": incoming_params.get((tier, entry.model_name), entry.litellm_params),
                            }
                        )
                    )
                    for entry in entries
                )
                for tier, (_, entries) in normalized_entries.items()
                if any(entry.litellm_params for entry in entries)
                or (isinstance(existing_configs, dict) and tier in existing_configs)
            }
        )
        return {  # mutable-ok: Pydantic before-validator requires a concrete mapping
            **value,
            "tiers": normalized_tiers,
            "tier_model_configs": tier_model_configs,
        }

    @field_validator("escalation_keywords")
    @classmethod
    def _normalize_escalation_keywords(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [stripped for keyword in value if (stripped := keyword.strip())]

    @field_validator("plan_mode_min_tier", mode="before")
    @classmethod
    def _coerce_plan_mode_min_tier(cls, value: object) -> object:
        if isinstance(value, ComplexityTier):
            return value.value
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("plan_mode_patterns")
    @classmethod
    def _normalize_plan_mode_patterns(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Blank patterns are dropped rather than kept: an empty string substring-matches every
        request, which would silently floor all traffic (same failure mode keyword_tier_rules
        rejects)."""
        if value is None:
            return None
        return tuple(stripped for pattern in value if (stripped := pattern.strip()))

    @model_validator(mode="after")
    def _validate_plan_mode_min_tier(self) -> "ComplexityRouterConfig":
        if self.plan_mode_min_tier is None:
            return self
        if self.plan_mode_min_tier not in self.tier_names():
            raise ValueError(
                f"plan_mode_min_tier {self.plan_mode_min_tier!r} is not an active tier: it must name "
                f"one of {', '.join(self.tier_names())}"
            )
        if self.plan_mode_min_tier not in self.tiers:
            raise ValueError(
                f"plan_mode_min_tier {self.plan_mode_min_tier} has no model configured in tiers; "
                "a floor pointing at an unconfigured tier would route every plan-mode request to the "
                "default fallback instead of the premium pool the operator intended"
            )
        return self

    @model_validator(mode="after")
    def _validate_classifier_config(self) -> "ComplexityRouterConfig":
        if self.uses_llm_classifier and self.classifier_llm_config is None:
            raise ValueError(f"classifier_llm_config is required when classifier_type is {self.classifier_type!r}")
        if self.classifier_type == "custom" and self.classifier_plugin is None:
            raise ValueError("classifier_plugin is required when classifier_type is 'custom'")
        if self.classifier_plugin is not None and self.classifier_type != "custom":
            raise ValueError(
                f"classifier_plugin is set but classifier_type is {self.classifier_type!r}; "
                "the plugin would never run. Set classifier_type 'custom' or remove classifier_plugin"
            )
        return self

    @field_validator("heuristic_first_max_tier", mode="before")
    @classmethod
    def _coerce_heuristic_first_max_tier(cls, value: object) -> object:
        if isinstance(value, ComplexityTier):
            return value.value
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def _validate_heuristic_first_max_tier(self) -> "ComplexityRouterConfig":
        if self.classifier_type != "heuristic_first":
            if self.heuristic_first_max_tier is not None:
                raise ValueError(
                    f"heuristic_first_max_tier is set but classifier_type is {self.classifier_type!r}; "
                    "the local scorer would never gate the classifier. Set classifier_type "
                    "'heuristic_first' or remove heuristic_first_max_tier"
                )
            return self
        threshold: Final = self.heuristic_first_max_tier
        if threshold is None:
            raise ValueError(
                "heuristic_first_max_tier is required when classifier_type is 'heuristic_first': without a "
                "threshold there is nothing to decide whether a request escalates to the LLM classifier"
            )
        names: Final = self.tier_names()
        if threshold not in names:
            raise ValueError(
                f"heuristic_first_max_tier {threshold!r} is not an active tier: it must name one of {', '.join(names)}"
            )
        if threshold == names[-1]:
            raise ValueError(
                f"heuristic_first_max_tier {threshold} is the highest tier, so every request would short-circuit "
                "and the LLM classifier would never run; name a lower tier or use classifier_type 'heuristic'"
            )
        if threshold not in self.tiers:
            raise ValueError(
                f"heuristic_first_max_tier {threshold} has no model configured in tiers; a threshold pointing at "
                "an unconfigured tier would route short-circuited requests to the default fallback instead of the "
                "pool the operator intended"
            )
        return self

    @field_validator("fallback_tier", "classification_prompt")
    @classmethod
    def _reject_blank_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped: Final = value.strip()
        if not stripped:
            raise ValueError("must be non-empty; omit the field instead")
        return stripped

    @field_validator("classification_prompt")
    @classmethod
    def _cap_classification_prompt(cls, value: str | None) -> str | None:
        if value is not None and len(value) > MAX_CLASSIFICATION_PROMPT_CHARS:
            raise ValueError(f"classification_prompt exceeds {MAX_CLASSIFICATION_PROMPT_CHARS} characters")
        return value

    @property
    def has_custom_tiers(self) -> bool:
        """True when the operator replaced the built-in tier set via tier_definitions."""
        return self.tier_definitions is not None

    @property
    def uses_llm_classifier(self) -> bool:
        """True when this router can call classifier_llm_config.model, so the model is a real
        dependency: authorized against the caller's key, counted in the health graph, and given a
        prebuilt rubric. 'heuristic_first' only calls it for traffic the local scorer escalates,
        which still makes it a dependency on every one of those requests."""
        return self.classifier_type in LLM_CLASSIFIER_TYPES

    def tier_names(self) -> tuple[str, ...]:
        """The active tier names: the defined names, or the built-in set in severity order."""
        if self.tier_definitions is not None:
            return tuple(definition.name for definition in self.tier_definitions)
        return tuple(tier.value for tier in TIER_SEVERITY_ORDER)

    def classifier_wire_labels(self) -> tuple[str, ...]:
        """The tier names the classifier is told to emit: defined names, or the display labels."""
        if self.tier_definitions is not None:
            return self.tier_names()
        return tuple(label for _, label in self.labeled_tiers())

    def resolve_classified_tier(self, label: str) -> ComplexityTier | str | None:
        """Resolve a classifier reply to the active tier it names, or None when it names none."""
        if self.tier_definitions is None:
            return self.tier_for_label(label)
        folded: Final = label.strip().casefold()
        return next((name for name in self.tier_names() if name.casefold() == folded), None)

    def _tier_definition_conflicts(self) -> tuple[str, ...]:
        """Error messages for config features that cannot coexist with a custom tier set."""
        llm_config: Final = self.classifier_llm_config
        order_dependent: Final = tuple(
            label
            for label, enabled in (
                ("adaptive", self.adaptive),
                ("session_affinity", self.session_affinity),
                ("escalation_keywords", bool(self.escalation_keywords)),
                ("plugins", bool(self.plugins)),
            )
            if enabled
        )
        return tuple(
            message
            for present, message in (
                (
                    bool(order_dependent),
                    f"{', '.join(order_dependent)} cannot be combined with tier_definitions: these features "
                    "rely on the built-in tier severity order, which a custom tier set does not define",
                ),
                (
                    llm_config is not None and llm_config.system_prompt is not None,
                    "classifier_llm_config.system_prompt cannot be combined with tier_definitions: a wholesale "
                    "replacement prompt drops the defined-tier bullets and the trust boundary; use "
                    "classification_prompt, which replaces only the opening instructions and keeps both",
                ),
                (
                    llm_config is not None and llm_config.classification_rubric is not None,
                    "classifier_llm_config.classification_rubric cannot be combined with tier_definitions: the "
                    "preset calibration examples are written against the built-in tiers, which a custom tier "
                    "set replaces",
                ),
                (
                    self.classifier_fallback == "default_model",
                    "classifier_fallback 'default_model' cannot be combined with tier_definitions: fallback_tier "
                    "is where a custom-tier router routes when the classifier fails",
                ),
                (
                    bool(self.tier_labels),
                    "tier_labels cannot be combined with tier_definitions: labels rename the built-in tiers, "
                    "which a custom tier set replaces; name the tiers directly in tier_definitions",
                ),
            )
            if present
        )

    @model_validator(mode="after")
    def _validate_tier_definitions(self) -> "ComplexityRouterConfig":
        if self.tier_definitions is None:
            orphaned: Final = next(
                (
                    field
                    for field, value in (
                        ("fallback_tier", self.fallback_tier),
                        ("classification_prompt", self.classification_prompt),
                    )
                    if value is not None
                ),
                None,
            )
            if orphaned is not None:
                raise ValueError(f"{orphaned} requires tier_definitions")
            return self
        names: Final = tuple(definition.name for definition in self.tier_definitions)
        if not 2 <= len(names) <= MAX_TIER_DEFINITIONS:
            raise ValueError(
                f"tier_definitions must define between 2 and {MAX_TIER_DEFINITIONS} tiers, got {len(names)}"
            )
        folded: Final = tuple(name.casefold() for name in names)
        duplicated: Final = tuple(
            sorted(frozenset(name for name, fold in zip(names, folded) if folded.count(fold) > 1))
        )
        if duplicated:
            raise ValueError(f"tier_definitions names must be unique (case-insensitive): {', '.join(duplicated)}")
        if self.classifier_type in ("heuristic", "heuristic_first"):
            raise ValueError(
                "tier_definitions requires classifier_type 'llm' or 'custom': the heuristic scorer only "
                "produces the built-in tiers"
            )
        conflicts: Final = self._tier_definition_conflicts()
        if conflicts:
            raise ValueError("; ".join(conflicts))
        defined: Final = frozenset(names)
        missing: Final = tuple(sorted(defined - frozenset(self.tiers)))
        if missing:
            raise ValueError(f"tiers must map every defined tier to a model; missing: {', '.join(missing)}")
        unknown: Final = tuple(sorted(frozenset(self.tiers) - defined))
        if unknown:
            raise ValueError(f"tiers keys must be defined in tier_definitions; unknown: {', '.join(unknown)}")
        empty_pools: Final = tuple(sorted(name for name in names if not self.tiers.get(name)))
        if empty_pools:
            raise ValueError(
                f"tiers must map every defined tier to at least one model; empty: {', '.join(empty_pools)}"
            )
        if self.fallback_tier is None:
            raise ValueError(
                "fallback_tier is required with tier_definitions: it is where requests route when the "
                "LLM classifier fails"
            )
        if self.fallback_tier not in defined:
            raise ValueError(
                f"fallback_tier {self.fallback_tier!r} is not one of the defined tiers: {', '.join(names)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_keyword_rule_tiers(self) -> "ComplexityRouterConfig":
        if not self.keyword_tier_rules:
            return self
        valid: Final = frozenset(self.tier_names())
        unknown_tiers: Final = tuple(
            sorted(frozenset(rule.tier for rule in self.keyword_tier_rules if rule.tier not in valid))
        )
        if unknown_tiers:
            raise ValueError(
                f"keyword_tier_rules reference unknown tiers: {', '.join(unknown_tiers)}; "
                f"valid tiers: {', '.join(self.tier_names())}"
            )
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
