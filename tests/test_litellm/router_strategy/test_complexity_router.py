"""
Tests for the Complexity Router.

Tests the rule-based complexity scoring and tier assignment.
"""

import pytest
from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
    DimensionScore,
)
from litellm.router_strategy.complexity_router.config import (
    ComplexityRouterConfig,
    ComplexityTier,
    DEFAULT_COMPLEXITY_CONFIG,
)


class TestComplexityTierClassification:
    """Test the complexity tier classification logic."""

    @pytest.fixture
    def router(self):
        """Create a ComplexityRouter instance for testing."""
        config = {
            "tiers": {
                "SIMPLE": "gpt-4o-mini",
                "MEDIUM": "gpt-4o",
                "COMPLEX": "claude-sonnet-4",
                "REASONING": "claude-opus-4",
            }
        }
        return ComplexityRouter(
            model_name="test_complexity_router",
            litellm_router_instance=None,  # Not needed for classification tests
            complexity_router_config=config,
        )

    def test_simple_greeting(self, router):
        """Simple greetings should be classified as SIMPLE."""
        tier, score, signals = router.classify("Hello!")
        assert tier == ComplexityTier.SIMPLE

    def test_simple_question(self, router):
        """Simple 'what is' questions should be SIMPLE."""
        tier, score, signals = router.classify("What is the capital of France?")
        assert tier == ComplexityTier.SIMPLE
        assert any("simple" in s.lower() for s in signals)

    def test_code_question(self, router):
        """Code-related questions should trend toward COMPLEX."""
        tier, score, signals = router.classify(
            "Write a Python function that implements binary search with error handling"
        )
        assert tier in [ComplexityTier.MEDIUM, ComplexityTier.COMPLEX]
        assert any("code" in s.lower() for s in signals)

    def test_reasoning_explicit(self, router):
        """Explicit reasoning markers should trigger REASONING tier."""
        tier, score, signals = router.classify(
            "Let's think step by step about how to solve this problem. "
            "Analyze this carefully and show your reasoning."
        )
        assert tier == ComplexityTier.REASONING
        assert any("reasoning" in s.lower() for s in signals)

    def test_technical_content(self, router):
        """Technical content should trend toward COMPLEX."""
        tier, score, signals = router.classify(
            "Explain the architecture of a distributed microservice system "
            "with Kubernetes orchestration and gRPC communication."
        )
        assert tier in [ComplexityTier.COMPLEX, ComplexityTier.REASONING]

    def test_multi_step_patterns(self, router):
        """Multi-step patterns should increase complexity score."""
        tier, score, signals = router.classify(
            "First, analyze the requirements. "
            "Then, design the database schema. "
            "Finally, implement the API endpoints."
        )
        assert any("multi-step" in s.lower() for s in signals)
        assert tier in [ComplexityTier.MEDIUM, ComplexityTier.COMPLEX]

    def test_multiple_questions(self, router):
        """Multiple questions should increase complexity."""
        tier, score, signals = router.classify(
            "What is the difference between TCP and UDP? "
            "When should I use each? "
            "What are the performance implications? "
            "How do they handle packet loss?"
        )
        assert any("question" in s.lower() for s in signals)

    def test_long_prompt(self, router):
        """Long prompts should trend toward complex."""
        long_text = "This is a detailed analysis request. " * 100
        tier, score, signals = router.classify(long_text)
        assert any("long" in s.lower() for s in signals)


class TestModelSelection:
    """Test that correct models are returned for each tier."""

    @pytest.fixture
    def router(self):
        """Create a ComplexityRouter with custom tier mapping."""
        config = {
            "tiers": {
                "SIMPLE": "gemini-2.0-flash",
                "MEDIUM": "gpt-4o-mini",
                "COMPLEX": "claude-sonnet-4",
                "REASONING": "claude-opus-4",
            }
        }
        return ComplexityRouter(
            model_name="test_router",
            litellm_router_instance=None,
            complexity_router_config=config,
        )

    def test_simple_tier_model(self, router):
        """SIMPLE tier should return the configured simple model."""
        model = router.get_model_for_tier(ComplexityTier.SIMPLE)
        assert model == "gemini-2.0-flash"

    def test_medium_tier_model(self, router):
        """MEDIUM tier should return the configured medium model."""
        model = router.get_model_for_tier(ComplexityTier.MEDIUM)
        assert model == "gpt-4o-mini"

    def test_complex_tier_model(self, router):
        """COMPLEX tier should return the configured complex model."""
        model = router.get_model_for_tier(ComplexityTier.COMPLEX)
        assert model == "claude-sonnet-4"

    def test_reasoning_tier_model(self, router):
        """REASONING tier should return the configured reasoning model."""
        model = router.get_model_for_tier(ComplexityTier.REASONING)
        assert model == "claude-opus-4"


class TestConfig:
    """Test configuration handling."""

    def test_default_config(self):
        """Default config should have all required fields."""
        config = DEFAULT_COMPLEXITY_CONFIG
        assert config.tiers is not None
        assert config.tier_boundaries is not None
        assert config.dimension_weights is not None

    def test_custom_tier_boundaries(self):
        """Custom tier boundaries should be respected."""
        config = ComplexityRouterConfig(
            tiers={"SIMPLE": "model-a", "MEDIUM": "model-b", "COMPLEX": "model-c", "REASONING": "model-d"},
            tier_boundaries={
                "simple_medium": 0.1,
                "medium_complex": 0.3,
                "complex_reasoning": 0.5,
            }
        )
        assert config.tier_boundaries["simple_medium"] == 0.1
        assert config.tier_boundaries["medium_complex"] == 0.3
        assert config.tier_boundaries["complex_reasoning"] == 0.5

    def test_custom_dimension_weights(self):
        """Custom dimension weights should be respected."""
        config = ComplexityRouterConfig(
            tiers={"SIMPLE": "model-a", "MEDIUM": "model-b", "COMPLEX": "model-c", "REASONING": "model-d"},
            dimension_weights={
                "tokenCount": 0.5,
                "codePresence": 0.5,
            }
        )
        assert config.dimension_weights["tokenCount"] == 0.5
        assert config.dimension_weights["codePresence"] == 0.5


class TestDimensionScore:
    """Test the DimensionScore class."""

    def test_dimension_score_with_signal(self):
        """DimensionScore should store name, score, and signal."""
        ds = DimensionScore("test", 0.5, "test signal")
        assert ds.name == "test"
        assert ds.score == 0.5
        assert ds.signal == "test signal"

    def test_dimension_score_without_signal(self):
        """DimensionScore should work without a signal."""
        ds = DimensionScore("test", 0.5)
        assert ds.name == "test"
        assert ds.score == 0.5
        assert ds.signal is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prompt(self):
        """Empty prompts should be handled gracefully."""
        router = ComplexityRouter(
            model_name="test",
            litellm_router_instance=None,
            complexity_router_config={
                "tiers": {
                    "SIMPLE": "model-a",
                    "MEDIUM": "model-b",
                    "COMPLEX": "model-c",
                    "REASONING": "model-d",
                }
            },
        )
        tier, score, signals = router.classify("")
        assert tier == ComplexityTier.SIMPLE  # Empty = simple

    def test_unicode_content(self):
        """Unicode content should be handled correctly."""
        router = ComplexityRouter(
            model_name="test",
            litellm_router_instance=None,
            complexity_router_config={
                "tiers": {
                    "SIMPLE": "model-a",
                    "MEDIUM": "model-b",
                    "COMPLEX": "model-c",
                    "REASONING": "model-d",
                }
            },
        )
        tier, score, signals = router.classify("こんにちは、世界！🌍")
        assert tier is not None

    def test_with_system_prompt(self):
        """System prompt should be considered in classification."""
        router = ComplexityRouter(
            model_name="test",
            litellm_router_instance=None,
            complexity_router_config={
                "tiers": {
                    "SIMPLE": "model-a",
                    "MEDIUM": "model-b",
                    "COMPLEX": "model-c",
                    "REASONING": "model-d",
                }
            },
        )
        # System prompt with code context
        tier, score, signals = router.classify(
            "Hello",
            system_prompt="You are a Python programming assistant."
        )
        # The code keywords in system prompt should influence scoring
        assert tier is not None
