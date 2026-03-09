"""
Configuration for the Complexity Router.

Contains default keyword lists, weights, tier boundaries, and configuration classes.
All values are configurable via proxy config.yaml.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ComplexityTier(str, Enum):
    """Complexity tiers for routing decisions."""
    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"
    REASONING = "REASONING"


# ─── Default Keyword Lists ───
# Note: Keywords should be full words/phrases to avoid substring false positives.
# The matching logic uses word boundary detection for single-word keywords.

DEFAULT_CODE_KEYWORDS: List[str] = [
    "function", "class", "def", "const", "let", "var",
    "import", "export", "return", "async", "await",
    "try", "catch", "exception", "error", "debug",
    "api", "endpoint", "request", "response",
    "database", "sql", "query", "schema",
    "algorithm", "implement", "refactor", "optimize",
    "python", "javascript", "typescript", "java", "rust", "golang",
    "react", "vue", "angular", "node", "docker", "kubernetes",
    "git", "commit", "merge", "branch", "pull request",
]

DEFAULT_REASONING_KEYWORDS: List[str] = [
    "step by step", "think through", "let's think",
    "reason through", "analyze this", "break down",
    "explain your reasoning", "show your work",
    "chain of thought", "think carefully",
    "consider all", "evaluate", "pros and cons",
    "compare and contrast", "weigh the options",
    "logical", "deduce", "infer", "conclude",
]

DEFAULT_TECHNICAL_KEYWORDS: List[str] = [
    "architecture", "distributed", "scalable", "microservice",
    "machine learning", "neural network", "deep learning",
    "encryption", "authentication", "authorization",
    "performance", "latency", "throughput", "benchmark",
    "concurrency", "parallel", "threading",
    "memory", "cpu", "gpu", "optimization",
    "protocol", "tcp", "http", "grpc", "websocket",
    "container", "orchestration",
    # Note: "async", "kubernetes", "docker" are in DEFAULT_CODE_KEYWORDS
]

DEFAULT_SIMPLE_KEYWORDS: List[str] = [
    "what is", "what's", "define", "definition of",
    "who is", "who was", "when did", "when was",
    "where is", "where was", "how many", "how much",
    "yes or no", "true or false",
    "simple", "brief", "short", "quick",
    "hello", "hi", "hey", "thanks", "thank you",
    "goodbye", "bye", "okay",
    # Note: "ok" removed due to false positives (matches "token", "book", etc.)
]


# ─── Default Dimension Weights ───

DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "tokenCount": 0.10,        # Reduced - length is less important than content
    "codePresence": 0.30,      # High - code requests need capable models
    "reasoningMarkers": 0.25,  # High - explicit reasoning requests
    "technicalTerms": 0.25,    # High - technical content matters
    "simpleIndicators": 0.05,  # Low - don't over-penalize simple patterns
    "multiStepPatterns": 0.03,
    "questionComplexity": 0.02,
}


# ─── Default Tier Boundaries ───

DEFAULT_TIER_BOUNDARIES: Dict[str, float] = {
    "simple_medium": 0.15,      # Lower threshold to catch more MEDIUM cases
    "medium_complex": 0.35,     # Lower threshold to catch technical COMPLEX cases
    "complex_reasoning": 0.60,  # Reasoning tier reserved for explicit reasoning markers
}


# ─── Default Token Thresholds ───

DEFAULT_TOKEN_THRESHOLDS: Dict[str, int] = {
    "simple": 15,    # Only very short prompts (<15 tokens) are penalized
    "complex": 400,  # Long prompts (>400 tokens) get complexity boost
}


# ─── Default Tier to Model Mapping ───

DEFAULT_TIER_MODELS: Dict[str, str] = {
    "SIMPLE": "gpt-4o-mini",
    "MEDIUM": "gpt-4o",
    "COMPLEX": "claude-sonnet-4-20250514",
    "REASONING": "claude-sonnet-4-20250514",
}


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
    simple_keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords indicating simple/basic queries",
    )
    
    # Default model if scoring fails
    default_model: Optional[str] = Field(
        default=None,
        description="Default model to use if tier cannot be determined",
    )
    
    model_config = ConfigDict(extra="allow")  # Allow additional fields


# Combined default config
DEFAULT_COMPLEXITY_CONFIG = ComplexityRouterConfig()
