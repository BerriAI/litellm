"""
Configuration for the Complexity Router.

Contains default keyword lists, weights, tier boundaries, and configuration classes.
All values are configurable via proxy config.yaml.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ComplexityTier(str, Enum):
    """Complexity tiers for routing decisions."""

    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    REASONING = "REASONING"


TIER_SEVERITY_ORDER: tuple[ComplexityTier, ...] = (
    ComplexityTier.SIMPLE,
    ComplexityTier.MEDIUM,
    ComplexityTier.COMPLEX,
    ComplexityTier.REASONING,
)


class KeywordTierRule(BaseModel):
    """A deterministic override: if any keyword matches, route to this tier."""

    keywords: List[str] = Field(
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
        cleaned = [stripped for keyword in self.keywords if (stripped := keyword.strip())]
        if not cleaned:
            raise ValueError("keyword_tier_rules entries must contain at least one non-empty keyword")
        self.keywords = cleaned
        return self


# ─── Default Keyword Lists ───
# Note: Keywords should be full words/phrases to avoid substring false positives.
# The matching logic uses word boundary detection for single-word keywords.

DEFAULT_CODE_KEYWORDS: List[str] = [
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

DEFAULT_REASONING_KEYWORDS: List[str] = [
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

DEFAULT_TECHNICAL_KEYWORDS: List[str] = [
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

DEFAULT_SIMPLE_KEYWORDS: List[str] = [
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

DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "tokenCount": 0.10,  # Reduced - length is less important than content
    "codePresence": 0.30,  # High - code requests need capable models
    "reasoningMarkers": 0.25,  # High - explicit reasoning requests
    "technicalTerms": 0.25,  # High - technical content matters
    "simpleIndicators": 0.05,  # Low - don't over-penalize simple patterns
    "multiStepPatterns": 0.03,
    "questionComplexity": 0.02,
}


# ─── Default Tier Boundaries ───

DEFAULT_TIER_BOUNDARIES: Dict[str, float] = {
    "simple_medium": 0.15,  # Lower threshold to catch more MEDIUM cases
    "medium_complex": 0.35,  # Lower threshold to catch technical COMPLEX cases
    "complex_reasoning": 0.60,  # Reasoning tier reserved for explicit reasoning markers
}


# ─── Default Token Thresholds ───

DEFAULT_TOKEN_THRESHOLDS: Dict[str, int] = {
    "simple": 15,  # Only very short prompts (<15 tokens) are penalized
    "complex": 400,  # Long prompts (>400 tokens) get complexity boost
}


# ─── Default Tier to Model Mapping ───

DEFAULT_TIER_MODELS: Dict[str, str] = {
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


class ComplexityRouterConfig(BaseModel):
    """Configuration for the ComplexityRouter."""

    # Tier to model mapping
    tiers: Dict[str, str] = Field(
        default_factory=lambda: DEFAULT_TIER_MODELS.copy(),
        description="Mapping of complexity tiers to model names",
    )

    # Tier boundaries (normalized scores)
    tier_boundaries: Dict[str, float] = Field(
        default_factory=lambda: DEFAULT_TIER_BOUNDARIES.copy(),
        description="Score boundaries between tiers",
    )

    # Token count thresholds
    token_thresholds: Dict[str, int] = Field(
        default_factory=lambda: DEFAULT_TOKEN_THRESHOLDS.copy(),
        description="Token count thresholds for simple/complex classification",
    )

    # Dimension weights
    dimension_weights: Dict[str, float] = Field(
        default_factory=lambda: DEFAULT_DIMENSION_WEIGHTS.copy(),
        description="Weights for each scoring dimension",
    )

    # Keyword lists (overridable)
    code_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords indicating code-related content",
    )
    reasoning_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords indicating reasoning-required content",
    )
    technical_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords indicating technical content",
    )
    custom_technical_keywords: Optional[list[str]] = Field(
        default=None,
        description=(
            "Domain-specific technical keywords appended to the effective base list "
            "(technical_keywords if set, otherwise DEFAULT_TECHNICAL_KEYWORDS). "
            "Order is preserved; duplicates are removed case-insensitively against "
            "the base list and within this list."
        ),
    )
    simple_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords indicating simple/basic queries",
    )

    # Default model if scoring fails
    default_model: Optional[str] = Field(
        default=None,
        description="Default model to use if tier cannot be determined",
    )

    # Classifier strategy
    classifier_type: Literal["heuristic", "llm"] = Field(
        default="heuristic",
        description="Classification strategy: local regex/keyword scoring, or an LLM call",
    )
    classifier_llm_config: Optional[ClassifierLLMConfig] = Field(
        default=None,
        description="Configuration for the LLM classifier; required when classifier_type is 'llm'",
    )

    # Deterministic keyword -> tier overrides, evaluated before weighted scoring
    keyword_tier_rules: Optional[List[KeywordTierRule]] = Field(
        default=None,
        description="Rules that force a specific tier when their keywords match the prompt",
    )

    # Semantic (embedding) matching for keyword_tier_rules instead of literal text matching
    semantic_keyword_matching: bool = Field(
        default=False,
        description="Match keyword_tier_rules by embedding similarity instead of literal text",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model (LiteLLM model name) used when semantic_keyword_matching is enabled",
    )
    match_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a semantic keyword match",
    )

    model_config = ConfigDict(extra="allow")  # Allow additional fields

    @model_validator(mode="after")
    def _validate_llm_classifier_config(self) -> "ComplexityRouterConfig":
        if self.classifier_type == "llm" and self.classifier_llm_config is None:
            raise ValueError("classifier_llm_config is required when classifier_type is 'llm'")
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


# Combined default config
DEFAULT_COMPLEXITY_CONFIG = ComplexityRouterConfig()
