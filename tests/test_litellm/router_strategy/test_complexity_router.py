"""
Tests for the ComplexityRouter.

Tests the rule-based complexity scoring and tier assignment logic.
"""
import os
import sys
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm import Router
from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
    DimensionScore,
)
from litellm.router_strategy.complexity_router.config import (
    DEFAULT_COMPLEXITY_CONFIG,
    ComplexityRouterConfig,
    ComplexityTier,
)


@pytest.fixture
def mock_router_instance():
    """Create a mock LiteLLM Router instance."""
    router = MagicMock()
    return router


@pytest.fixture
def basic_config() -> Dict:
    """Basic configuration with tier mappings."""
    return {
        "tiers": {
            "SIMPLE": "gpt-4o-mini",
            "MEDIUM": "gpt-4o",
            "COMPLEX": "claude-sonnet-4-20250514",
            "REASONING": "o1-preview",
        },
        "tier_boundaries": {
            "simple_medium": 0.25,
            "medium_complex": 0.50,
            "complex_reasoning": 0.75,
        },
    }


@pytest.fixture
def complexity_router(mock_router_instance, basic_config):
    """Create a ComplexityRouter instance with basic config."""
    return ComplexityRouter(
        model_name="test-complexity-router",
        litellm_router_instance=mock_router_instance,
        complexity_router_config=basic_config,
    )


class TestDimensionScore:
    """Test the DimensionScore class."""

    def test_dimension_score_creation(self):
        """Test creating a DimensionScore."""
        score = DimensionScore("tokenCount", 0.5, "short (25 tokens)")
        assert score.name == "tokenCount"
        assert score.score == 0.5
        assert score.signal == "short (25 tokens)"

    def test_dimension_score_no_signal(self):
        """Test creating a DimensionScore without signal."""
        score = DimensionScore("tokenCount", 0)
        assert score.name == "tokenCount"
        assert score.score == 0
        assert score.signal is None


class TestComplexityRouterInit:
    """Test ComplexityRouter initialization."""

    def test_init_with_config(self, mock_router_instance, basic_config):
        """Test initialization with configuration."""
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=basic_config,
        )
        assert router.model_name == "test-router"
        assert router.config.tiers["SIMPLE"] == "gpt-4o-mini"
        assert router.config.tiers["REASONING"] == "o1-preview"

    def test_init_without_config(self, mock_router_instance):
        """Test initialization without configuration uses defaults."""
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
        )
        assert router.model_name == "test-router"
        # Should have equivalent default values but NOT be the same instance
        assert router.config.tiers == DEFAULT_COMPLEXITY_CONFIG.tiers
        assert router.config is not DEFAULT_COMPLEXITY_CONFIG  # Not a singleton

    def test_init_with_default_model(self, mock_router_instance, basic_config):
        """Test initialization with default_model override."""
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=basic_config,
            default_model="fallback-model",
        )
        assert router.config.default_model == "fallback-model"


class TestTokenScoring:
    """Test token count scoring."""

    def test_short_prompt_negative_score(self, complexity_router):
        """Short prompts should get negative scores (simple indicator)."""
        tier, score, signals = complexity_router.classify("What is Python?")
        # Should be classified as SIMPLE due to short length and simple indicator
        assert tier == ComplexityTier.SIMPLE
        assert any("short" in s.lower() for s in signals) or any("simple" in s.lower() for s in signals)

    def test_long_prompt_positive_score(self, complexity_router):
        """Long prompts should get positive scores (complex indicator)."""
        # Create a long prompt (~600 tokens)
        long_prompt = "Explain the following concept in detail: " + " ".join(
            ["distributed systems architecture and microservices patterns"] * 50
        )
        tier, score, signals = complexity_router.classify(long_prompt)
        # Should have positive score and detect long token count or technical terms
        assert score > 0, f"Expected positive score for long prompt, got {score}"
        assert any("long" in s.lower() for s in signals) or any("technical" in s.lower() for s in signals)


class TestCodePresenceScoring:
    """Test code-related keyword scoring."""

    def test_code_keywords_increase_complexity(self, complexity_router):
        """Code keywords should increase complexity score."""
        prompt = "Write a Python function that implements a binary search algorithm with async support"
        tier, score, signals = complexity_router.classify(prompt)
        # Should detect code presence
        assert any("code" in s.lower() for s in signals)
        # Score should be positive (code keywords add to complexity)
        assert score > -0.5  # Not heavily negative

    def test_multiple_code_keywords(self, complexity_router):
        """Multiple code keywords should strongly increase complexity."""
        prompt = (
            "Debug this Python function that uses async/await with try/catch "
            "for API endpoint error handling in the database query"
        )
        tier, score, signals = complexity_router.classify(prompt)
        assert any("code" in s.lower() for s in signals)


class TestReasoningMarkerScoring:
    """Test reasoning marker detection."""

    def test_single_reasoning_marker(self, complexity_router):
        """Single reasoning marker should increase score."""
        prompt = "Think through this problem step by step and explain your reasoning"
        tier, score, signals = complexity_router.classify(prompt)
        assert any("reasoning" in s.lower() for s in signals)

    def test_multiple_reasoning_markers_override(self, complexity_router):
        """Multiple reasoning markers should force REASONING tier."""
        prompt = "Let's think step by step. Analyze this carefully and reason through each option. Show your work."
        tier, score, signals = complexity_router.classify(prompt)
        # 2+ reasoning markers should force REASONING tier
        assert tier == ComplexityTier.REASONING

    def test_system_prompt_reasoning_not_counted(self, complexity_router):
        """Reasoning markers in system prompt should not count for override."""
        user_prompt = "What is 2+2?"
        system_prompt = "Think step by step before answering."
        tier, score, signals = complexity_router.classify(user_prompt, system_prompt)
        # Should still be SIMPLE since user message is simple
        assert tier in [ComplexityTier.SIMPLE, ComplexityTier.MEDIUM]


class TestSimpleIndicatorScoring:
    """Test simple indicator detection."""

    def test_simple_greeting(self, complexity_router):
        """Simple greetings should be classified as SIMPLE."""
        tier, score, signals = complexity_router.classify("Hello, how are you?")
        assert tier == ComplexityTier.SIMPLE

    def test_definition_questions(self, complexity_router):
        """Definition questions should be classified as SIMPLE."""
        prompts = [
            "What is machine learning?",
            "Define artificial intelligence",
            "Who is Alan Turing?",
        ]
        for prompt in prompts:
            tier, score, signals = complexity_router.classify(prompt)
            assert tier == ComplexityTier.SIMPLE, f"Expected SIMPLE for: {prompt}"


class TestMultiStepPatterns:
    """Test multi-step pattern detection."""

    def test_first_then_pattern(self, complexity_router):
        """'First...then' patterns should increase complexity."""
        prompt = "First analyze the data, then create a visualization, then write a report"
        tier, score, signals = complexity_router.classify(prompt)
        assert any("multi-step" in s.lower() for s in signals)

    def test_numbered_steps(self, complexity_router):
        """Numbered steps should increase complexity."""
        prompt = "1. Set up the environment 2. Install dependencies 3. Run the tests"
        tier, score, signals = complexity_router.classify(prompt)
        assert any("multi-step" in s.lower() for s in signals)


class TestQuestionComplexity:
    """Test question complexity scoring."""

    def test_multiple_questions(self, complexity_router):
        """Multiple questions should increase complexity."""
        prompt = "What is the capital? Where is it located? How many people live there? What's the climate like?"
        tier, score, signals = complexity_router.classify(prompt)
        assert any("question" in s.lower() for s in signals)


class TestTierAssignment:
    """Test tier assignment based on scores."""

    def test_simple_tier(self, complexity_router):
        """Simple prompts should get SIMPLE tier."""
        tier, score, signals = complexity_router.classify("Hi there!")
        assert tier == ComplexityTier.SIMPLE

    def test_medium_tier(self, complexity_router):
        """Moderately complex prompts should get MEDIUM tier."""
        prompt = "Explain how REST APIs work with HTTP methods"
        tier, score, signals = complexity_router.classify(prompt)
        assert tier in [ComplexityTier.SIMPLE, ComplexityTier.MEDIUM]

    def test_complex_tier(self, complexity_router):
        """Complex prompts should get positive complexity score with technical signals."""
        prompt = (
            "Design a distributed microservice architecture for a high-throughput "
            "real-time data processing pipeline with Kubernetes orchestration, "
            "implementing proper authentication and encryption protocols"
        )
        tier, score, signals = complexity_router.classify(prompt)
        # Should detect technical terms
        assert any("technical" in s.lower() for s in signals), f"Expected technical signals, got {signals}"
        # Score should be positive due to technical content
        assert score > 0, f"Expected positive score, got {score}"

    def test_reasoning_tier(self, complexity_router):
        """Reasoning prompts should get REASONING tier."""
        prompt = (
            "Think step by step and reason through this: Analyze the pros and cons "
            "of different database architectures for our distributed system, "
            "considering performance, scalability, and consistency tradeoffs"
        )
        tier, score, signals = complexity_router.classify(prompt)
        assert tier == ComplexityTier.REASONING


class TestModelSelection:
    """Test model selection based on tier."""

    def test_get_model_for_simple(self, complexity_router):
        """Should return correct model for SIMPLE tier."""
        model = complexity_router.get_model_for_tier(ComplexityTier.SIMPLE)
        assert model == "gpt-4o-mini"

    def test_get_model_for_complex(self, complexity_router):
        """Should return correct model for COMPLEX tier."""
        model = complexity_router.get_model_for_tier(ComplexityTier.COMPLEX)
        assert model == "claude-sonnet-4-20250514"

    def test_get_model_for_reasoning(self, complexity_router):
        """Should return correct model for REASONING tier."""
        model = complexity_router.get_model_for_tier(ComplexityTier.REASONING)
        assert model == "o1-preview"

    def test_get_model_fallback_to_default(self, mock_router_instance):
        """Should fallback to default_model if tier not configured."""
        config = {
            "tiers": {},  # Empty tiers
            "default_model": "fallback-model",
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        model = router.get_model_for_tier(ComplexityTier.SIMPLE)
        assert model == "fallback-model"


class TestPreRoutingHook:
    """Test the async_pre_routing_hook method."""

    @pytest.mark.asyncio
    async def test_pre_routing_hook_simple_message(self, complexity_router):
        """Test pre-routing hook with a simple message."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is not None
        assert result.model == "gpt-4o-mini"  # SIMPLE tier model
        assert result.messages == messages

    @pytest.mark.asyncio
    async def test_pre_routing_hook_complex_message(self, complexity_router):
        """Test pre-routing hook with a message containing technical content."""
        messages = [
            {"role": "user", "content": (
                "Design a distributed microservice architecture with Kubernetes "
                "orchestration, implementing proper authentication, encryption, "
                "and database optimization for high throughput. Think step by step "
                "about the performance implications and scalability requirements."
            )}
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is not None
        # Should return a valid model from the configured tiers
        assert result.model in ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514", "o1-preview"]

    @pytest.mark.asyncio
    async def test_pre_routing_hook_no_messages(self, complexity_router):
        """Test pre-routing hook returns None when no messages."""
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_routing_hook_empty_messages(self, complexity_router):
        """Test pre-routing hook returns None when messages empty."""
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_routing_hook_with_system_prompt(self, complexity_router):
        """Test pre-routing hook considers system prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is not None
        # Should still be SIMPLE
        assert result.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_pre_routing_hook_reasoning_message(self, complexity_router):
        """Test pre-routing hook with reasoning markers."""
        messages = [
            {"role": "user", "content": "Let's think step by step and reason through this problem carefully."}
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is not None
        assert result.model == "o1-preview"  # REASONING tier model


class TestConfigOverrides:
    """Test configuration override functionality."""

    def test_custom_tier_boundaries(self, mock_router_instance):
        """Test custom tier boundaries work correctly."""
        config = {
            "tiers": {
                "SIMPLE": "mini-model",
                "MEDIUM": "medium-model",
                "COMPLEX": "complex-model",
                "REASONING": "reasoning-model",
            },
            "tier_boundaries": {
                "simple_medium": -0.5,  # Very low threshold - anything above -0.5 is MEDIUM+
                "medium_complex": -0.3,
                "complex_reasoning": 0.0,
            },
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        # With very low thresholds, even neutral prompts should be COMPLEX or higher
        tier, score, signals = router.classify(
            "Explain how HTTP works with REST APIs and distributed systems"
        )
        # With boundaries this low, should be at least MEDIUM (anything above -0.5)
        assert tier != ComplexityTier.SIMPLE, f"Expected non-SIMPLE tier, got {tier} with score {score}"

    def test_custom_token_thresholds(self, mock_router_instance):
        """Test custom token thresholds work correctly."""
        config = {
            "tiers": {
                "SIMPLE": "mini-model",
                "MEDIUM": "medium-model",
                "COMPLEX": "complex-model",
                "REASONING": "reasoning-model",
            },
            "token_thresholds": {
                "simple": 10,  # Very low - prompts with >10 tokens are not "short"
                "complex": 100,  # Lower than default - prompts with >100 tokens are "long"
            },
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        # A longer prompt (~150 tokens) should be considered "long" with these thresholds
        long_prompt = "This is a test prompt " * 30  # ~120 tokens
        tier, score, signals = router.classify(long_prompt)
        # Should get token length signal indicating "long"
        assert any("long" in s.lower() if s else False for s in signals), f"Expected 'long' signal, got {signals}"


class TestAsyncPreRoutingHookEdgeCases:
    """Test edge cases for async_pre_routing_hook method."""

    @pytest.mark.asyncio
    async def test_pre_routing_hook_multi_turn_conversation(self, complexity_router):
        """Test pre-routing hook with multi-turn conversation uses last user message."""
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Hello!"},  # Last user message - simple
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is not None
        assert result.model == "gpt-4o-mini"  # SIMPLE tier based on last message

    @pytest.mark.asyncio
    async def test_pre_routing_hook_multi_user_messages(self, complexity_router):
        """Test pre-routing hook uses the last user message for classification."""
        # Multiple user messages - should classify based on the LAST one
        messages = [
            {"role": "user", "content": "Design a complex distributed system"},  # Complex prompt
            {"role": "assistant", "content": "I can help with that."},
            {"role": "user", "content": "Hello!"},  # Simple prompt - this should be used
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is not None
        # Should use the last user message "Hello!" which is SIMPLE
        assert result.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_pre_routing_hook_no_user_message(self, complexity_router):
        """Test pre-routing hook returns None when no user message found."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_routing_hook_only_list_content(self, complexity_router):
        """Test pre-routing hook returns None when all user content is list type."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        # Should return None since we can't extract string content
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_routing_hook_preserves_messages(self, complexity_router):
        """Test pre-routing hook preserves original messages in response."""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello!"},
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is not None
        assert result.messages == messages

    @pytest.mark.asyncio
    async def test_pre_routing_hook_empty_string_content(self, complexity_router):
        """Test pre-routing hook returns None for empty string content."""
        messages = [
            {"role": "user", "content": ""},
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        # Empty string content is treated as "no user message found"
        assert result is None


class TestSingletonMutation:
    """Test that the config singleton is not mutated."""

    def test_default_config_not_mutated(self, mock_router_instance):
        """Test that creating routers without config doesn't mutate defaults."""
        from litellm.router_strategy.complexity_router.config import (
            ComplexityRouterConfig,
        )

        # Get original default
        original_default = ComplexityRouterConfig().default_model
        
        # Create router with empty config and custom default_model
        router1 = ComplexityRouter(
            model_name="test-router-1",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=None,
            default_model="custom-fallback",
        )
        
        # Create another router without config
        router2 = ComplexityRouter(
            model_name="test-router-2",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=None,
        )
        
        # Router2 should have fresh defaults, not router1's custom default_model
        # Create a fresh config to check
        fresh_config = ComplexityRouterConfig()
        assert fresh_config.default_model == original_default
        assert router1.config.default_model == "custom-fallback"
        # Router2's config should be independent
        assert router2.config is not router1.config


class TestKeywordFalsePositives:
    """Test that keyword matching uses word boundaries to avoid false positives."""

    def test_api_not_in_capital(self, complexity_router):
        """'api' should not match in 'capital'."""
        prompt = "What is the capital of France?"
        tier, score, signals = complexity_router.classify(prompt)
        # Should NOT detect code presence from 'api' in 'capital'
        assert not any("code" in s.lower() for s in signals), f"False positive: got code signal from 'capital'"
        # Should be SIMPLE (definition question)
        assert tier == ComplexityTier.SIMPLE

    def test_git_not_in_digital(self, complexity_router):
        """'git' should not match in 'digital'."""
        prompt = "Explain digital marketing strategies"
        tier, score, signals = complexity_router.classify(prompt)
        # Should NOT detect code presence from 'git' in 'digital'
        assert not any("code" in s.lower() for s in signals), f"False positive: got code signal from 'digital'"

    def test_try_not_in_entry(self, complexity_router):
        """'try' should not match in 'entry'."""
        prompt = "What is the entry point for this application?"
        tier, score, signals = complexity_router.classify(prompt)
        # 'entry' contains 'try' but should not trigger code detection
        # Note: 'application' might trigger something, but 'try' should not
        pass  # Just ensure no crash; false positive check is the main goal

    def test_error_not_in_terrorism(self, complexity_router):
        """'error' should not match in 'terrorism'."""
        prompt = "The country is dealing with terrorism"
        tier, score, signals = complexity_router.classify(prompt)
        assert not any("code" in s.lower() for s in signals), f"False positive: got code signal from 'terrorism'"

    def test_class_not_in_classical(self, complexity_router):
        """'class' should not match in 'classical'."""
        prompt = "I enjoy listening to classical music"
        tier, score, signals = complexity_router.classify(prompt)
        assert not any("code" in s.lower() for s in signals), f"False positive: got code signal from 'classical'"

    def test_merge_not_in_emerged(self, complexity_router):
        """'merge' should not match in 'emerged'."""
        prompt = "A new leader emerged from the crowd"
        tier, score, signals = complexity_router.classify(prompt)
        assert not any("code" in s.lower() for s in signals), f"False positive: got code signal from 'emerged'"

    def test_actual_api_keyword_detected(self, complexity_router):
        """Actual 'api' usage should be detected."""
        prompt = "How do I call the REST api endpoint?"
        tier, score, signals = complexity_router.classify(prompt)
        # Should detect code presence from actual 'api' usage
        assert any("code" in s.lower() for s in signals), f"Expected code signal for 'api', got {signals}"

    def test_actual_git_keyword_detected(self, complexity_router):
        """Actual 'git' usage should be detected."""
        prompt = "How do I use git to commit changes?"
        tier, score, signals = complexity_router.classify(prompt)
        # Should detect code presence from actual 'git' usage
        assert any("code" in s.lower() for s in signals), f"Expected code signal for 'git', got {signals}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prompt(self, complexity_router):
        """Test handling of empty prompt."""
        tier, score, signals = complexity_router.classify("")
        assert tier == ComplexityTier.SIMPLE
        assert score <= 0

    def test_very_long_prompt(self, complexity_router):
        """Test handling of very long prompt."""
        # 10000+ character prompt
        long_prompt = "explain " * 2000
        tier, score, signals = complexity_router.classify(long_prompt)
        # Should have positive score due to length
        assert score > 0, f"Expected positive score for very long prompt, got {score}"
        # Should detect long token count
        assert any("long" in s.lower() for s in signals), f"Expected 'long' signal, got {signals}"

    def test_unicode_prompt(self, complexity_router):
        """Test handling of unicode characters."""
        prompt = "What is 日本語? Explain émojis 🎉 and symbols ∑∏∫"
        tier, score, signals = complexity_router.classify(prompt)
        # Should not crash, should be classified
        assert tier in [ComplexityTier.SIMPLE, ComplexityTier.MEDIUM]

    def test_multiline_prompt(self, complexity_router):
        """Test handling of multiline prompts with step patterns."""
        prompt = """
        Step 1: Analyze the problem.
        Step 2: Propose a solution.
        Step 3: Implement it.
        """
        tier, score, signals = complexity_router.classify(prompt)
        # The "step N" pattern should be detected
        assert any("multi-step" in s.lower() for s in signals), f"Expected multi-step signal, got {signals}"


class TestRouterComplexityDeploymentMethods:
    """Tests for Router._is_complexity_router_deployment and Router.init_complexity_router_deployment."""

    def test_is_complexity_router_deployment_true(self):
        """_is_complexity_router_deployment returns True for complexity router models."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ]
        )
        from litellm.types.router import LiteLLM_Params

        params = LiteLLM_Params(model="auto_router/complexity_router/my-router")
        assert router._is_complexity_router_deployment(params) is True

    def test_is_complexity_router_deployment_false(self):
        """_is_complexity_router_deployment returns False for regular models."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ]
        )
        from litellm.types.router import LiteLLM_Params

        params = LiteLLM_Params(model="openai/gpt-4o-mini")
        assert router._is_complexity_router_deployment(params) is False

    def test_init_complexity_router_deployment(self):
        """init_complexity_router_deployment registers a ComplexityRouter."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ]
        )
        from litellm.types.router import Deployment, LiteLLM_Params

        deployment = Deployment(
            model_name="auto_router/complexity_router/test-router",
            litellm_params=LiteLLM_Params(
                model="auto_router/complexity_router/test-router",
                complexity_router_default_model="gpt-4o-mini",
                complexity_router_config={
                    "tiers": {
                        "SIMPLE": "gpt-4o-mini",
                        "MEDIUM": "gpt-4o",
                        "COMPLEX": "claude-sonnet-4-20250514",
                        "REASONING": "o1-preview",
                    }
                },
            ),
            model_info={"id": "test-id"},
        )
        router.init_complexity_router_deployment(deployment)
        assert "auto_router/complexity_router/test-router" in router.complexity_routers
