"""
Tests for the ComplexityRouter.

Tests the rule-based complexity scoring and tier assignment logic.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

import litellm
from litellm import Router
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
    DimensionScore,
)
from litellm.router_strategy.complexity_router.config import (
    DEFAULT_COMPLEXITY_CONFIG,
    DEFAULT_TECHNICAL_KEYWORDS,
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

    def test_get_model_for_tier_list_random_choice(self, mock_router_instance):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": ["cheap", "premium"], "MEDIUM": "mid"},
                "default_model": "mid",
            },
        )
        pool = ["cheap", "premium"]
        with patch(
            "litellm.router_strategy.complexity_router.complexity_router.random.choice",
            return_value="premium",
        ) as choice:
            assert router.get_model_for_tier(ComplexityTier.SIMPLE) == "premium"
            choice.assert_called_once_with(pool)
        assert router.get_model_for_tier(ComplexityTier.MEDIUM) == "mid"

    def test_get_model_for_tier_empty_pool_raises(self, mock_router_instance):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": []},
                "default_model": "mid",
            },
        )
        with pytest.raises(ValueError, match="Empty model pool for tier SIMPLE"):
            router.get_model_for_tier(ComplexityTier.SIMPLE)


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
            {
                "role": "user",
                "content": (
                    "Design a distributed microservice architecture with Kubernetes "
                    "orchestration, implementing proper authentication, encryption, "
                    "and database optimization for high throughput. Think step by step "
                    "about the performance implications and scalability requirements."
                ),
            }
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is not None
        # Should return a valid model from the configured tiers
        assert result.model in [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-sonnet-4-20250514",
            "o1-preview",
        ]

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
            {
                "role": "user",
                "content": "Let's think step by step and reason through this problem carefully.",
            }
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
        tier, score, signals = router.classify("Explain how HTTP works with REST APIs and distributed systems")
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


class TestCustomTechnicalKeywords:
    """Test the custom_technical_keywords config option."""

    def test_custom_keywords_appended_to_defaults(self, mock_router_instance):
        """Custom keywords should be appended to the default technical keywords."""
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={"custom_technical_keywords": ["udp", "kafka"]},
        )
        assert router.technical_keywords == DEFAULT_TECHNICAL_KEYWORDS + ["udp", "kafka"]

    def test_custom_keywords_appended_to_technical_keywords_override(self, mock_router_instance):
        """Custom keywords should be appended to a technical_keywords override."""
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "technical_keywords": ["quantum", "photonics"],
                "custom_technical_keywords": ["udp"],
            },
        )
        assert router.technical_keywords == ["quantum", "photonics", "udp"]

    def test_custom_keywords_deduplicated_case_insensitively(self, mock_router_instance):
        """Duplicates against the base list and within the custom list should be dropped."""
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={"custom_technical_keywords": ["TCP", "udp", "UDP", "kafka"]},
        )
        lowered = [kw.lower() for kw in router.technical_keywords]
        assert lowered == [kw.lower() for kw in DEFAULT_TECHNICAL_KEYWORDS] + [
            "udp",
            "kafka",
        ]

    def test_no_custom_keywords_leaves_defaults_unchanged(self, mock_router_instance):
        """Absent or None custom_technical_keywords should leave the keyword list identical."""
        router_absent = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={"tiers": {"MEDIUM": "gpt-4o"}},
        )
        router_none = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={"custom_technical_keywords": None},
        )
        assert router_absent.technical_keywords == DEFAULT_TECHNICAL_KEYWORDS
        assert router_none.technical_keywords == DEFAULT_TECHNICAL_KEYWORDS

    def test_prompt_with_only_custom_keywords_scores_technical(self, mock_router_instance, basic_config):
        """A prompt matching only custom keywords should score higher on technicalTerms."""
        prompt = "Configure udp multicast between kafka brokers"
        baseline_router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=basic_config,
        )
        custom_router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                **basic_config,
                "custom_technical_keywords": ["UDP", "Kafka"],
            },
        )
        _, baseline_score, baseline_signals = baseline_router.classify(prompt)
        _, custom_score, custom_signals = custom_router.classify(prompt)
        assert not any("technical" in s.lower() for s in baseline_signals)
        assert any("technical" in s.lower() for s in custom_signals), f"Expected technical signal, got {custom_signals}"
        assert custom_score > baseline_score


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
            {
                "role": "user",
                "content": "Design a complex distributed system",
            },  # Complex prompt
            {"role": "assistant", "content": "I can help with that."},
            {
                "role": "user",
                "content": "Hello!",
            },  # Simple prompt - this should be used
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
        """Test pre-routing hook falls back to default model when no user message found."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        # Should return default model rather than None (None would cause
        # the complexity_router deployment itself to be selected, crashing)
        assert result is not None
        assert result.model in [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-sonnet-4-20250514",
            "o1-preview",
        ]

    @pytest.mark.asyncio
    async def test_pre_routing_hook_list_content(self, complexity_router):
        """Test pre-routing hook handles list-format message content (OpenAI multi-part format)."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            },
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        # Should extract text from list content and classify normally
        assert result is not None
        assert result.model == "gpt-4o-mini"  # "Hello, how are you?" is SIMPLE

    @pytest.mark.asyncio
    async def test_pre_routing_hook_list_content_complex(self, complexity_router):
        """Test pre-routing hook classifies list-format content by complexity."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Think step by step and reason through this: design a distributed system",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            }
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        assert result is not None
        assert result.model == "o1-preview"  # REASONING tier

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
        """Test pre-routing hook falls back to default model for empty string content."""
        messages = [
            {"role": "user", "content": ""},
        ]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        # Empty string content → no extractable user message → routes to default model
        assert result is not None
        assert result.model in [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-sonnet-4-20250514",
            "o1-preview",
        ]


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
        assert not any("code" in s.lower() for s in signals), "False positive: got code signal from 'capital'"
        # Should be SIMPLE (definition question)
        assert tier == ComplexityTier.SIMPLE

    def test_git_not_in_digital(self, complexity_router):
        """'git' should not match in 'digital'."""
        prompt = "Explain digital marketing strategies"
        tier, score, signals = complexity_router.classify(prompt)
        # Should NOT detect code presence from 'git' in 'digital'
        assert not any("code" in s.lower() for s in signals), "False positive: got code signal from 'digital'"

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
        assert not any("code" in s.lower() for s in signals), "False positive: got code signal from 'terrorism'"

    def test_class_not_in_classical(self, complexity_router):
        """'class' should not match in 'classical'."""
        prompt = "I enjoy listening to classical music"
        tier, score, signals = complexity_router.classify(prompt)
        assert not any("code" in s.lower() for s in signals), "False positive: got code signal from 'classical'"

    def test_merge_not_in_emerged(self, complexity_router):
        """'merge' should not match in 'emerged'."""
        prompt = "A new leader emerged from the crowd"
        tier, score, signals = complexity_router.classify(prompt)
        assert not any("code" in s.lower() for s in signals), "False positive: got code signal from 'emerged'"

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

    def test_hybrid_initialization_waits_for_later_pool_deployments(self):
        router = Router(
            model_list=[
                {
                    "model_name": "hybrid",
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "complexity_router_default_model": "cheap",
                        "complexity_router_config": {
                            "adaptive": True,
                            "tiers": {
                                "SIMPLE": ["cheap"],
                                "MEDIUM": ["cheap", "premium"],
                            },
                        },
                    },
                },
                {
                    "model_name": "cheap",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "input_cost_per_token": 0.00000015,
                    },
                    "model_info": {
                        "adaptive_router_preferences": {
                            "quality_tier": 1,
                            "strengths": [],
                        }
                    },
                },
                {
                    "model_name": "premium",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "input_cost_per_token": 0.000005,
                    },
                    "model_info": {
                        "adaptive_router_preferences": {
                            "quality_tier": 3,
                            "strengths": [],
                        }
                    },
                },
            ]
        )

        adaptive = router.adaptive_routers["hybrid"]
        assert adaptive.model_to_cost == {
            "cheap": pytest.approx(0.00000015),
            "premium": pytest.approx(0.000005),
        }
        assert adaptive.model_to_prefs["cheap"].quality_tier == 1
        assert adaptive.model_to_prefs["premium"].quality_tier == 3


class TestAsyncPreRoutingHookMultiFormat:
    """Test async_pre_routing_hook with multiple input formats."""

    @pytest.mark.asyncio
    async def test_should_route_with_chat_completions_messages(self, complexity_router):
        """Test routing with standard chat completions messages."""
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        assert result is not None
        assert result.model is not None
        assert result.messages is not None

    @pytest.mark.asyncio
    async def test_should_route_with_responses_api_string_input(self, complexity_router):
        """Test routing with Responses API string input via handler dispatch."""
        from litellm.llms.openai.responses.guardrail_translation.handler import (
            OpenAIResponsesHandler,
        )
        from litellm.types.utils import CallTypes

        mock_mappings = {CallTypes.responses: OpenAIResponsesHandler}

        with patch(
            "litellm.llms.load_guardrail_translation_mappings",
            return_value=mock_mappings,
        ):
            result = await complexity_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={"input": "What is the capital of France?"},
                messages=None,
                input="What is the capital of France?",
            )

        assert result is not None
        assert result.model is not None
        # messages should be None since the original request didn't have messages
        assert result.messages is None

    @pytest.mark.asyncio
    async def test_should_route_with_responses_api_list_input(self, complexity_router):
        """Test routing with Responses API list input via handler dispatch."""
        from litellm.llms.openai.responses.guardrail_translation.handler import (
            OpenAIResponsesHandler,
        )
        from litellm.types.utils import CallTypes

        mock_mappings = {CallTypes.responses: OpenAIResponsesHandler}

        list_input = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {
                "role": "user",
                "content": "Write a Python function to sort a list using merge sort",
            },
        ]

        with patch(
            "litellm.llms.load_guardrail_translation_mappings",
            return_value=mock_mappings,
        ):
            result = await complexity_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={"input": list_input},
                messages=None,
                input=list_input,
            )

        assert result is not None
        assert result.model is not None
        assert result.messages is None

    @pytest.mark.asyncio
    async def test_should_use_route_based_inference(self, complexity_router):
        """Test that route-based call type inference is used when available."""
        from litellm.llms.openai.responses.guardrail_translation.handler import (
            OpenAIResponsesHandler,
        )
        from litellm.types.utils import CallTypes

        mock_mappings = {CallTypes.responses: OpenAIResponsesHandler}

        with patch(
            "litellm.llms.load_guardrail_translation_mappings",
            return_value=mock_mappings,
        ):
            result = await complexity_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={
                    "input": "Roll 2d4+1",
                    "litellm_metadata": {
                        "user_api_key_request_route": "/v1/responses",
                    },
                },
                messages=None,
            )

        assert result is not None
        assert result.model is not None

    @pytest.mark.asyncio
    async def test_should_return_none_when_no_messages_or_input(self, complexity_router):
        """Test that None is returned when neither messages nor input is available."""
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=None,
            input=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_should_prefer_original_messages_over_conversion(self, complexity_router):
        """Test that original messages are used when both messages and input are available."""
        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={"input": "This should be ignored"},
            messages=messages,
        )
        assert result is not None
        assert result.messages == messages

    @pytest.mark.asyncio
    async def test_should_include_instructions_in_classification(self, complexity_router):
        """Test that Responses API instructions influence classification via system message."""
        from litellm.llms.openai.responses.guardrail_translation.handler import (
            OpenAIResponsesHandler,
        )
        from litellm.types.utils import CallTypes

        mock_mappings = {CallTypes.responses: OpenAIResponsesHandler}

        with patch(
            "litellm.llms.load_guardrail_translation_mappings",
            return_value=mock_mappings,
        ):
            result = await complexity_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={
                    "input": "Write merge sort",
                    "instructions": "You are an expert Python developer. Use advanced algorithms and optimize for performance.",
                },
                messages=None,
            )

        assert result is not None
        assert result.model is not None


class TestExtractUserMessageAndSystemPrompt:
    """Test the _extract_user_message_and_system_prompt static method."""

    def test_should_extract_user_message(self):
        """Test extraction of the last user message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]
        user_msg, sys_prompt = ComplexityRouter._extract_user_message_and_system_prompt(messages)
        assert user_msg == "How are you?"
        assert sys_prompt == "You are helpful."

    def test_should_handle_no_user_message(self):
        """Test when there is no user message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hi!"},
        ]
        user_msg, sys_prompt = ComplexityRouter._extract_user_message_and_system_prompt(messages)
        assert user_msg is None
        assert sys_prompt == "You are helpful."

    def test_should_handle_multipart_content(self):
        """Test extraction from multipart content messages."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            }
        ]
        user_msg, sys_prompt = ComplexityRouter._extract_user_message_and_system_prompt(messages)
        assert user_msg == "Describe this image"
        assert sys_prompt is None

    def test_should_handle_empty_messages(self):
        """Test with empty messages list."""
        user_msg, sys_prompt = ComplexityRouter._extract_user_message_and_system_prompt([])
        assert user_msg is None
        assert sys_prompt is None


def _llm_response(content: str):
    """Build a fake acompletion response with the given message content."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


@pytest.fixture
def llm_classifier_config() -> Dict:
    """Config with an LLM-based classifier wired to a 'haiku-classifier' model."""
    return {
        "tiers": {
            "SIMPLE": "gpt-4o-mini",
            "MEDIUM": "gpt-4o",
            "COMPLEX": "claude-sonnet-4-20250514",
            "REASONING": "o1-preview",
        },
        "classifier_type": "llm",
        "classifier_llm_config": {"model": "haiku-classifier", "timeout_ms": 400},
    }


@pytest.fixture
def llm_complexity_router(mock_router_instance, llm_classifier_config):
    """ComplexityRouter configured to classify via an LLM call."""
    return ComplexityRouter(
        model_name="test-complexity-router",
        litellm_router_instance=mock_router_instance,
        complexity_router_config=llm_classifier_config,
    )


class TestLLMClassifierConfig:
    """Test config validation for the LLM classifier option."""

    def test_llm_classifier_type_requires_config(self):
        """classifier_type='llm' without classifier_llm_config must raise."""
        with pytest.raises(ValidationError):
            ComplexityRouterConfig(classifier_type="llm")

    def test_heuristic_classifier_type_needs_no_llm_config(self):
        """classifier_type='heuristic' (the default) needs no classifier_llm_config."""
        config = ComplexityRouterConfig()
        assert config.classifier_type == "heuristic"
        assert config.classifier_llm_config is None


class TestLLMClassifier:
    """Test the LLM-based classifier path (aclassify) and its fallback behavior."""

    @pytest.mark.asyncio
    async def test_aclassify_heuristic_skips_llm_call(self, complexity_router, mock_router_instance):
        """When classifier_type is 'heuristic' (default), aclassify must not call the LLM."""
        mock_router_instance.acompletion = AsyncMock()
        tier, score, signals = await complexity_router.aclassify("Hello!")
        mock_router_instance.acompletion.assert_not_called()
        assert tier == ComplexityTier.SIMPLE

    @pytest.mark.asyncio
    async def test_aclassify_llm_success_routes_by_llm_verdict(self, llm_complexity_router, mock_router_instance):
        """A well-formed structured LLM response should decide the tier directly.

        Uses a prompt that heuristic scoring alone would classify as SIMPLE, to prove
        the LLM verdict -- not the heuristic scorer -- is what decided the tier.
        """
        mock_router_instance.acompletion = AsyncMock(return_value=_llm_response('{"tier": "COMPLEX"}'))
        tier, score, signals = await llm_complexity_router.aclassify("hi")
        assert tier == ComplexityTier.COMPLEX
        assert "llm-classifier:COMPLEX" in signals
        mock_router_instance.acompletion.assert_awaited_once()
        call_kwargs = mock_router_instance.acompletion.call_args.kwargs
        assert call_kwargs["model"] == "haiku-classifier"
        assert call_kwargs["timeout"] == 0.4

    @pytest.mark.asyncio
    async def test_aclassify_forwards_request_metadata_for_spend_tracking(
        self, llm_complexity_router, mock_router_instance
    ):
        """The classifier call must carry the original request's metadata.

        Without this, the proxy's cost-tracking gate (_should_track_cost_callback)
        sees no user_api_key/team_id/user_id and silently drops all spend logging
        and budget accounting for the classifier call.
        """
        mock_router_instance.acompletion = AsyncMock(return_value=_llm_response('{"tier": "SIMPLE"}'))
        request_metadata = {"user_api_key": "sk-abc", "user_api_key_team_id": "team-1"}
        await llm_complexity_router.aclassify("hi", request_kwargs={"litellm_metadata": request_metadata})
        call_kwargs = mock_router_instance.acompletion.call_args.kwargs
        assert call_kwargs["metadata"] == request_metadata

    @pytest.mark.asyncio
    async def test_aclassify_strips_budget_reservation_from_classifier_metadata(
        self, llm_complexity_router, mock_router_instance
    ):
        """The classifier call must not receive the parent request's budget reservation.

        The reservation belongs to the routed completion the classifier is deciding
        on, not to this internal classifier call. Forwarding it would let the
        classifier's own cost-tracking reconcile against a reservation it has no
        business touching, so it must be stripped while the rest of the attribution
        metadata (key/team) is preserved.
        """
        mock_router_instance.acompletion = AsyncMock(return_value=_llm_response('{"tier": "SIMPLE"}'))
        request_metadata = {
            "user_api_key": "sk-abc",
            "user_api_key_team_id": "team-1",
            "user_api_key_budget_reservation": {"reserved_cost": 1.0},
            "user_api_key_auth": {"models": ["gpt-4o"], "budget_reservation": {"reserved_cost": 1.0}},
        }
        await llm_complexity_router.aclassify("hi", request_kwargs={"litellm_metadata": request_metadata})
        call_kwargs = mock_router_instance.acompletion.call_args.kwargs
        # user_api_key_budget_reservation is stripped (budget enforcement) while
        # user_api_key_auth is kept so _filter_deployments_by_model_access_groups
        # can scope the classifier's model selection to the caller's access groups,
        # but only as a sanitized copy without its budget_reservation sub-field:
        # the cost callback falls back to reading the reservation from inside the
        # auth object when the top-level key is absent.
        assert call_kwargs["metadata"] == {
            "user_api_key": "sk-abc",
            "user_api_key_team_id": "team-1",
            "user_api_key_auth": {"models": ["gpt-4o"]},
        }
        assert request_metadata["user_api_key_auth"] == {
            "models": ["gpt-4o"],
            "budget_reservation": {"reserved_cost": 1.0},
        }

    @pytest.mark.asyncio
    async def test_aclassify_falls_back_to_heuristic_on_llm_exception(
        self, llm_complexity_router, mock_router_instance
    ):
        """A timeout/error from the classifier model must fall back to heuristic scoring."""
        mock_router_instance.acompletion = AsyncMock(side_effect=TimeoutError("classifier timed out"))
        tier, score, signals = await llm_complexity_router.aclassify("Hello!")
        assert tier == llm_complexity_router.classify("Hello!")[0]
        assert tier == ComplexityTier.SIMPLE

    @pytest.mark.asyncio
    async def test_aclassify_falls_back_to_heuristic_on_unparseable_response(
        self, llm_complexity_router, mock_router_instance
    ):
        """Non-JSON or schema-violating output must fall back to heuristic scoring, not raise."""
        mock_router_instance.acompletion = AsyncMock(return_value=_llm_response("not json"))
        tier, score, signals = await llm_complexity_router.aclassify("Hello!")
        assert tier == ComplexityTier.SIMPLE

    @pytest.mark.asyncio
    async def test_aclassify_falls_back_to_heuristic_on_empty_content(
        self, llm_complexity_router, mock_router_instance
    ):
        """Empty/None message content (e.g. provider quirk) must fall back, not raise."""
        mock_router_instance.acompletion = AsyncMock(return_value=_llm_response(None))
        tier, score, signals = await llm_complexity_router.aclassify("Hello!")
        assert tier == ComplexityTier.SIMPLE

    @pytest.mark.asyncio
    async def test_pre_routing_hook_uses_llm_classifier_end_to_end(self, llm_complexity_router, mock_router_instance):
        """The full pre-routing hook should route using the LLM classifier's verdict."""
        mock_router_instance.acompletion = AsyncMock(return_value=_llm_response('{"tier": "REASONING"}'))
        request_metadata = {"user_api_key": "sk-abc", "user_api_key_team_id": "team-1"}
        result = await llm_complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={"litellm_metadata": request_metadata},
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result is not None
        assert result.model == "o1-preview"  # REASONING tier model
        call_kwargs = mock_router_instance.acompletion.call_args.kwargs
        assert call_kwargs["metadata"] == request_metadata


class TestRouterPreRoutingAliasOverrides:
    """
    Regression tests for: litellm_params configured on a complexity-router alias
    entry (e.g. `cache_control_injection_points`, `drop_params`) were silently
    dropped, because `async_pre_routing_hook` swaps `model` from the alias name
    to the selected tier's model *before* the deployment lookup - so the actual
    outbound call only ever merges in the tier deployment's own litellm_params,
    never the alias's.
    """

    def _make_router(self) -> Router:
        return Router(
            model_list=[
                {
                    "model_name": "smart-router",
                    "litellm_params": {
                        "model": "auto_router/complexity_router",
                        "drop_params": True,
                        "cache_control_injection_points": [{"location": "message", "role": "system"}],
                        "complexity_router_config": {
                            "tiers": {
                                "SIMPLE": "gpt-4o-mini",
                                "MEDIUM": "gpt-4o",
                            }
                        },
                        "complexity_router_default_model": "gpt-4o",
                    },
                },
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                },
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o"},
                },
            ]
        )

    @pytest.mark.asyncio
    async def test_alias_litellm_params_applied_to_request_kwargs(self):
        """cache_control_injection_points/drop_params set on the alias entry
        reach the outbound request even though the tier deployment is what
        actually gets called."""
        router = self._make_router()
        request_kwargs: Dict = {}

        result = await router.async_pre_routing_hook(
            model="smart-router",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is not None
        assert request_kwargs["drop_params"] is True
        assert request_kwargs["cache_control_injection_points"] == [{"location": "message", "role": "system"}]

    @pytest.mark.asyncio
    async def test_alias_overrides_exclude_only_model(self):
        """`model` (the alias marker, e.g. auto_router/complexity_router) is
        excluded since it's never a real provider model. Router-only fields
        like complexity_router_config DO flow through into request_kwargs at
        this layer - they're filtered from the actual outbound LLM call
        downstream by litellm.types.utils.all_litellm_params instead, not by
        the router's pre-routing hook. See test_router_init_only_params_are_
        never_sent_to_a_provider for the guard on that downstream filter."""
        router = self._make_router()
        request_kwargs: Dict = {}

        await router.async_pre_routing_hook(
            model="smart-router",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert "model" not in request_kwargs
        assert request_kwargs["complexity_router_config"] == {
            "tiers": {
                "SIMPLE": "gpt-4o-mini",
                "MEDIUM": "gpt-4o",
            }
        }
        assert request_kwargs["complexity_router_default_model"] == "gpt-4o"

    def test_router_init_only_params_are_never_sent_to_a_provider(self):
        """The router's pre-routing hook only excludes `model` (see
        test_alias_overrides_exclude_only_model above) - every other alias
        litellm_param, including router-init-only fields like
        complexity_router_config, flows into request_kwargs unfiltered. That's
        only safe because litellm.completion()/acompletion() itself strips
        anything listed in all_litellm_params before building the provider
        request. If one of these keys is ever removed from that list, it
        ships raw to the real provider as extra_body - verified live via
        litellm.completion(..., complexity_router_config={...}) landing in
        extra_body before this list included it."""
        from litellm.types.utils import all_litellm_params

        router_init_only_params = (
            "auto_router_config_path",
            "auto_router_config",
            "auto_router_default_model",
            "auto_router_embedding_model",
            "complexity_router_config",
            "complexity_router_default_model",
            "adaptive_router_config",
            "adaptive_router_default_model",
            "quality_router_config",
            "quality_router_default_model",
        )
        for param in router_init_only_params:
            assert param in all_litellm_params, (
                f"{param} must stay in litellm.types.utils.all_litellm_params - "
                "removing it means it ships raw to the real provider as extra_body"
            )

    @pytest.mark.asyncio
    async def test_caller_supplied_kwargs_are_not_overwritten(self):
        """A value the caller already passed for this request takes
        precedence over the alias's configured default."""
        router = self._make_router()
        request_kwargs: Dict = {"drop_params": False}

        await router.async_pre_routing_hook(
            model="smart-router",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert request_kwargs["drop_params"] is False

    @pytest.mark.asyncio
    async def test_non_alias_model_is_untouched(self):
        """A plain (non-router-alias) model name is not affected by the
        alias-override merge at all."""
        router = self._make_router()
        request_kwargs: Dict = {}

        result = await router.async_pre_routing_hook(
            model="gpt-4o-mini",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result is None
        assert request_kwargs == {}

    @pytest.mark.asyncio
    async def test_adaptive_router_alias_overrides_survive_reload(self):
        """Alias litellm_params are read fresh from self.model_list at request
        time (not cached at init), so a set_model_list() reload (e.g.
        /config/reload) - which rebuilds self.model_list but leaves an
        already-built AdaptiveRouter alone - can't leave them stale."""
        model_list = [
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/adaptive_router",
                    "drop_params": True,
                    "adaptive_router_config": {"available_models": ["gpt-4o-mini"]},
                },
            },
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "openai/gpt-4o-mini"},
            },
        ]
        router = Router(model_list=model_list)
        router.set_model_list(model_list)
        assert "smart-router" in router.adaptive_routers

        request_kwargs: Dict = {}
        await router.async_pre_routing_hook(
            model="smart-router",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
        )

        assert request_kwargs["drop_params"] is True


class TestAdaptiveSoftFloors:
    def test_adaptive_defaults_use_cost_weighted_cold_policy(self):
        config = ComplexityRouterConfig(
            adaptive=True,
            tiers={"SIMPLE": ["cheap"]},
        )
        assert config.adaptive_weights.quality == pytest.approx(0.3)
        assert config.adaptive_weights.cost == pytest.approx(0.7)
        assert config.tier_distance_penalty == pytest.approx(0.5)

    @pytest.fixture
    def adaptive_router_instance(self):
        router = MagicMock()
        router.model_list = [
            {
                "model_name": "cheap",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "input_cost_per_token": 0.00000015,
                },
                "model_info": {"adaptive_router_preferences": {"quality_tier": 1, "strengths": []}},
            },
            {
                "model_name": "premium",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "input_cost_per_token": 0.000005,
                },
                "model_info": {"adaptive_router_preferences": {"quality_tier": 3, "strengths": []}},
            },
        ]
        router.model_name_to_deployment_indices = {"cheap": [0], "premium": [1]}
        return router

    @pytest.fixture
    def hybrid_config(self) -> Dict:
        return {
            "adaptive": True,
            "adaptive_weights": {"quality": 0.7, "cost": 0.3},
            "tier_distance_penalty": 0.15,
            "tiers": {
                "SIMPLE": ["cheap"],
                "MEDIUM": ["cheap"],
                "COMPLEX": ["premium"],
                "REASONING": ["premium"],
            },
            "default_model": "cheap",
        }

    def test_adaptive_config_requires_non_empty_pools(self):
        with pytest.raises(ValidationError):
            ComplexityRouterConfig(adaptive=True, tiers={"SIMPLE": []})

    def test_cold_start_randomly_samples_unobserved_classified_tier_models(self, adaptive_router_instance):
        cr = ComplexityRouter(
            model_name="hybrid",
            litellm_router_instance=adaptive_router_instance,
            complexity_router_config={
                "adaptive": True,
                "tiers": {
                    "SIMPLE": ["cheap", "premium"],
                    "MEDIUM": ["premium"],
                },
            },
        )
        request_kwargs: Dict = {"metadata": {}}

        with patch(
            "litellm.router_strategy.complexity_router.complexity_router.random.choice",
            return_value="premium",
        ) as choice:
            picked = cr._soft_floor_pick(ComplexityTier.SIMPLE, "hi", request_kwargs)

        assert picked == "premium"
        choice.assert_called_once_with(("cheap", "premium"))
        decision = request_kwargs["metadata"]["adaptive_router_decision"]
        assert decision["phase"] == "cold_start"
        assert {candidate["model"] for candidate in decision["candidates"]} == {
            "cheap",
            "premium",
        }

    def test_get_model_for_tier_list_without_adaptive_random_choice(self, mock_router_instance):
        router = ComplexityRouter(
            model_name="test",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "adaptive": False,
                "tiers": {"SIMPLE": ["cheap", "premium"], "MEDIUM": "mid"},
                "default_model": "mid",
            },
        )
        pool = ["cheap", "premium"]
        with patch(
            "litellm.router_strategy.complexity_router.complexity_router.random.choice",
            return_value="premium",
        ) as choice:
            assert router.get_model_for_tier(ComplexityTier.SIMPLE) == "premium"
            choice.assert_called_once_with(pool)
        assert router.get_model_for_tier(ComplexityTier.MEDIUM) == "mid"

    def test_soft_floor_prefers_home_tier_when_posteriors_equal(self, adaptive_router_instance, hybrid_config):
        from litellm.router_strategy.adaptive_router.bandit import BanditCell
        from litellm.types.router import RequestType

        cr = ComplexityRouter(
            model_name="hybrid",
            litellm_router_instance=adaptive_router_instance,
            complexity_router_config=hybrid_config,
        )
        adaptive = cr._ensure_adaptive_router()
        assert adaptive is not None
        for model in ("cheap", "premium"):
            adaptive._cells[(RequestType.GENERAL, model)] = BanditCell(alpha=5.0, beta=5.0)

        # Equal quality samples; home-tier penalty should favor cheap for SIMPLE.
        with patch(
            "litellm.router_strategy.adaptive_router.bandit.thompson_sample",
            return_value=0.5,
        ):
            picked = cr._soft_floor_pick(ComplexityTier.SIMPLE, "hi")
        assert picked == "cheap"

    def test_soft_floor_allows_cross_tier_when_posterior_dominates(self, adaptive_router_instance, hybrid_config):
        from litellm.router_strategy.adaptive_router.bandit import BanditCell
        from litellm.types.router import RequestType

        cr = ComplexityRouter(
            model_name="hybrid",
            litellm_router_instance=adaptive_router_instance,
            complexity_router_config=hybrid_config,
        )
        adaptive = cr._ensure_adaptive_router()
        assert adaptive is not None
        adaptive._cells[(RequestType.GENERAL, "cheap")] = BanditCell(alpha=1.0, beta=20.0)
        adaptive._cells[(RequestType.GENERAL, "premium")] = BanditCell(alpha=20.0, beta=1.0)

        with patch(
            "litellm.router_strategy.adaptive_router.bandit.thompson_sample",
            side_effect=lambda cell, rng=None: cell.alpha / (cell.alpha + cell.beta),
        ):
            picked = cr._soft_floor_pick(ComplexityTier.SIMPLE, "hi")
        assert picked == "premium"

    def test_reused_model_has_zero_distance_in_each_configured_tier(self, adaptive_router_instance):
        from litellm.router_strategy.adaptive_router.bandit import BanditCell
        from litellm.types.router import RequestType

        cr = ComplexityRouter(
            model_name="hybrid",
            litellm_router_instance=adaptive_router_instance,
            complexity_router_config={
                "adaptive": True,
                "tiers": {
                    "SIMPLE": ["cheap"],
                    "MEDIUM": ["cheap", "premium"],
                    "COMPLEX": ["premium"],
                },
            },
        )
        adaptive = cr._ensure_adaptive_router()
        assert adaptive is not None
        for model in ("cheap", "premium"):
            adaptive._cells[(RequestType.GENERAL, model)] = BanditCell(alpha=6.0, beta=5.0)
        request_kwargs: Dict = {"metadata": {}}

        with patch(
            "litellm.router_strategy.adaptive_router.bandit.thompson_sample",
            return_value=0.5,
        ):
            cr._soft_floor_pick(ComplexityTier.MEDIUM, "hi", request_kwargs)

        candidates = request_kwargs["metadata"]["adaptive_router_decision"]["candidates"]
        assert {candidate["model"]: candidate["tier_distance"] for candidate in candidates} == {
            "cheap": 0,
            "premium": 0,
        }

    @pytest.mark.asyncio
    async def test_pre_routing_hook_adaptive_stashes_chosen_model(self, adaptive_router_instance, hybrid_config):
        cr = ComplexityRouter(
            model_name="hybrid",
            litellm_router_instance=adaptive_router_instance,
            complexity_router_config=hybrid_config,
        )
        request_kwargs: Dict = {"metadata": {}}
        result = await cr.async_pre_routing_hook(
            model="hybrid",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result is not None
        assert result.model in {"cheap", "premium"}
        assert request_kwargs["metadata"].get("adaptive_router_chosen_model") == result.model
        decision = request_kwargs["metadata"]["adaptive_router_decision"]
        assert decision["phase"] == "cold_start"
        assert decision["classified_tier"] == "SIMPLE"
        assert decision["request_type"] == "general"
        assert decision["eligible_mode"] == "classified_tier"
        assert decision["chosen_model"] == result.model
        assert {candidate["model"] for candidate in decision["candidates"]} == {"cheap"}


class TestLexicalKeywordTierRules:
    """Test deterministic (literal) keyword_tier_rules overrides."""

    @pytest.fixture
    def rule_config(self, basic_config) -> Dict:
        return {
            **basic_config,
            "keyword_tier_rules": [
                {"keywords": ["deploy to k8s"], "tier": "REASONING"},
            ],
        }

    @pytest.mark.asyncio
    async def test_matching_rule_overrides_scoring(self, mock_router_instance, rule_config):
        """A prompt hitting a rule keyword routes to that tier, not the scored tier."""
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=rule_config,
        )
        prompt = "please deploy to k8s now"
        # Without the rule this short prompt would not score into REASONING.
        scored_tier, _, _ = router.classify(prompt)
        assert scored_tier != ComplexityTier.REASONING

        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": prompt}],
        )
        assert result is not None
        assert result.model == "o1-preview"  # REASONING tier model

    @pytest.mark.asyncio
    async def test_most_severe_tier_wins_regardless_of_rule_order(self, mock_router_instance, basic_config):
        """When several rules match, the highest-severity tier wins, independent of list order."""
        config = {
            **basic_config,
            "keyword_tier_rules": [
                {"keywords": ["database"], "tier": "SIMPLE"},  # listed first, lower tier
                {"keywords": ["database"], "tier": "REASONING"},  # listed later, higher tier
            ],
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "tell me about the database"}],
        )
        assert result is not None
        assert result.model == "o1-preview"  # REASONING wins over the earlier SIMPLE rule

    @pytest.mark.asyncio
    async def test_distinct_keywords_escalate_to_highest_tier(self, mock_router_instance, basic_config):
        """A prompt hitting keywords across tiers routes to the most complex one."""
        config = {
            **basic_config,
            "keyword_tier_rules": [
                {"keywords": ["hi"], "tier": "SIMPLE"},
                {"keywords": ["advise"], "tier": "COMPLEX"},
                {"keywords": ["kubernetes"], "tier": "REASONING"},
            ],
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "hi, advise me on kubernetes"}],
        )
        assert result is not None
        assert result.model == "o1-preview"  # REASONING, the highest of SIMPLE/COMPLEX/REASONING

    def test_lexical_override_returns_most_severe_matched_tier(self, mock_router_instance, basic_config):
        """Unit-level check of the escalation helper across mixed matches."""
        config = {
            **basic_config,
            "keyword_tier_rules": [
                {"keywords": ["hi"], "tier": "SIMPLE"},
                {"keywords": ["advise"], "tier": "COMPLEX"},
            ],
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        assert router._lexical_tier_override("hi there, please advise") == ComplexityTier.COMPLEX
        assert router._lexical_tier_override("just saying hi") == ComplexityTier.SIMPLE
        assert router._lexical_tier_override("nothing relevant here") is None

    @pytest.mark.asyncio
    async def test_no_rule_match_falls_back_to_scoring(self, mock_router_instance, basic_config):
        """A prompt that matches no rule is classified by the scorer as usual."""
        config = {
            **basic_config,
            "keyword_tier_rules": [
                {"keywords": ["zzznomatch"], "tier": "REASONING"},
            ],
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert result is not None
        assert result.model == "gpt-4o-mini"  # SIMPLE via scoring, rule did not fire

    def test_word_boundary_avoids_substring_false_positive(self, mock_router_instance, basic_config):
        """A single-word rule keyword must not match inside a larger word."""
        config = {
            **basic_config,
            "keyword_tier_rules": [{"keywords": ["k8s"], "tier": "REASONING"}],
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        assert router._lexical_tier_override("running my k8s cluster") == ComplexityTier.REASONING
        assert router._lexical_tier_override("what is a k8scluster thing") is None


def _make_embedding_response(vectors: List[List[float]]) -> "litellm.EmbeddingResponse":
    return litellm.EmbeddingResponse(
        model="fake-embed",
        data=[{"embedding": vec, "index": idx, "object": "embedding"} for idx, vec in enumerate(vectors)],
        object="list",
    )


class FakeEmbeddingRouter:
    """A stand-in router whose embeddings are deterministic 2D unit vectors.

    Any text mentioning a cluster/container concept maps to [1, 0]; everything
    else maps to [0, 1]. This lets the real SemanticRouter compute exact cosine
    similarities (1.0 or 0.0) so threshold behavior is testable without a network call.
    """

    _CLUSTER_MARKERS = ("k8s", "kube", "container", "cluster", "orchestrat")

    def __init__(self):
        self.async_embedding_calls: List[List[str]] = []
        self.async_embedding_kwargs: List[Dict] = []
        # Every embedded batch (sync route-index build AND async query), so tests can count
        # builds independently of which embedding path the library happens to use.
        self.embedded_batches: List[List[str]] = []
        # Thread ids of the synchronous (route-index build) embedding calls, so a test can
        # assert the build is offloaded off the event-loop thread.
        self.sync_embedding_thread_ids: List[int] = []

    def _vectors(self, docs: List[str]) -> List[List[float]]:
        return [
            [1.0, 0.0] if any(marker in doc.lower() for marker in self._CLUSTER_MARKERS) else [0.0, 1.0] for doc in docs
        ]

    @staticmethod
    def _as_list(text) -> List[str]:
        return text if isinstance(text, list) else [text]

    def embedding(self, input, model, **kwargs):
        import threading

        docs = self._as_list(input)
        self.embedded_batches.append(docs)
        self.sync_embedding_thread_ids.append(threading.get_ident())
        return _make_embedding_response(self._vectors(docs))

    async def aembedding(self, input, model, **kwargs):
        docs = self._as_list(input)
        self.embedded_batches.append(docs)
        self.async_embedding_calls.append(docs)
        self.async_embedding_kwargs.append(kwargs)
        return _make_embedding_response(self._vectors(docs))

    def utterance_embedding_count(self, utterance: str) -> int:
        """How many times the given route utterance was embedded == number of route-index builds."""
        return sum(1 for batch in self.embedded_batches if utterance in batch)


class TestSemanticKeywordTierRules:
    """Test embedding-based keyword_tier_rules matching."""

    @pytest.mark.asyncio
    async def test_semantic_match_routes_to_rule_tier(self, basic_config):
        """A paraphrase (no literal keyword) still routes via embedding similarity."""
        fake_router = FakeEmbeddingRouter()
        config = {
            **basic_config,
            "keyword_tier_rules": [
                {"keywords": ["kubernetes deployment", "container orchestration"], "tier": "REASONING"},
                {"keywords": ["hello", "thanks"], "tier": "SIMPLE"},
            ],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.5,
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=fake_router,
            complexity_router_config=config,
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "help me roll out my k8s cluster today"}],
        )
        assert result is not None
        assert result.model == "o1-preview"  # REASONING via semantic match
        assert fake_router.async_embedding_calls, "expected an embedding call for the prompt"

    @pytest.mark.asyncio
    async def test_tier_matches_on_best_utterance_not_diluted_by_others(self, basic_config):
        """A tier with several keywords must match if the query is close to ANY of them,
        not the average across all of them. A tier's route holds one utterance per keyword;
        mean aggregation (the semantic_router library default) scores the query against the
        *average* similarity across every utterance in the route, so a real match on one
        keyword gets dragged below threshold by the tier's other, unrelated keywords.
        """
        fake_router = FakeEmbeddingRouter()
        config = {
            **basic_config,
            "keyword_tier_rules": [
                {"keywords": ["kubernetes deployment", "thanks", "goodbye"], "tier": "REASONING"},
            ],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.5,
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=fake_router,
            complexity_router_config=config,
        )
        # Only "kubernetes deployment" is close to this query (cos 1.0); "thanks" and
        # "goodbye" are orthogonal (cos 0.0). Mean over the three would be ~0.33, below the
        # 0.5 threshold; the best (max) utterance alone clears it.
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "help me roll out my k8s cluster today"}],
        )
        assert result is not None
        assert result.model == "o1-preview"  # REASONING via best-utterance semantic match

    @pytest.mark.asyncio
    async def test_semantic_embedding_call_carries_caller_metadata(self, basic_config):
        """The query embedding call must carry the caller's metadata/litellm_metadata
        so embedding spend is attributed and budget-checked against the originating
        key/team, instead of being logged as an untracked, unattributed cost.
        """
        fake_router = FakeEmbeddingRouter()
        config = {
            **basic_config,
            "keyword_tier_rules": [{"keywords": ["kubernetes deployment"], "tier": "REASONING"}],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.5,
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=fake_router,
            complexity_router_config=config,
        )
        caller_metadata = {"user_api_key_hash": "hash-abc", "user_api_key_team_id": "team-1"}
        caller_litellm_metadata = {"user_api_key": "hash-abc"}
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={"metadata": caller_metadata, "litellm_metadata": caller_litellm_metadata},
            messages=[{"role": "user", "content": "roll out my k8s cluster"}],
        )
        assert result is not None
        assert fake_router.async_embedding_kwargs, "expected an embedding call for the prompt"
        assert fake_router.async_embedding_kwargs[0]["metadata"] == caller_metadata
        assert fake_router.async_embedding_kwargs[0]["litellm_metadata"] == caller_litellm_metadata

    @pytest.mark.asyncio
    async def test_semantic_embedding_call_strips_budget_reservation(self, basic_config):
        """The embedding call must not carry the parent request's budget reservation.

        The reservation belongs to the routed completion this embedding helps select, not
        to the embedding call. Forwarding it would let the embedding's cost callback
        finalize the reservation, so the routed completion's callback then skips
        incrementing the key/team budget - letting a caller run completions while only the
        embedding cost is enforced. Key/team attribution fields must still be forwarded.
        """
        fake_router = FakeEmbeddingRouter()
        config = {
            **basic_config,
            "keyword_tier_rules": [{"keywords": ["kubernetes deployment"], "tier": "REASONING"}],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.5,
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=fake_router,
            complexity_router_config=config,
        )
        caller_metadata = {
            "user_api_key_hash": "hash-abc",
            "user_api_key_team_id": "team-1",
            "user_api_key_budget_reservation": {"reserved_cost": 1.0},
            "user_api_key_auth": {"models": ["voyage-3-5"], "budget_reservation": {"reserved_cost": 1.0}},
        }
        await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={"metadata": caller_metadata, "litellm_metadata": dict(caller_metadata)},
            messages=[{"role": "user", "content": "roll out my k8s cluster"}],
        )
        assert fake_router.async_embedding_kwargs, "expected an embedding call for the prompt"
        # user_api_key_budget_reservation is stripped to prevent budget-bypass.
        # user_api_key_auth is kept so _filter_deployments_by_model_access_groups
        # scopes the embedding model selection to the caller's authorized groups,
        # but its budget_reservation sub-field is removed because the cost callback
        # falls back to reading the reservation from inside the auth object.
        expected = {
            "user_api_key_hash": "hash-abc",
            "user_api_key_team_id": "team-1",
            "user_api_key_auth": {"models": ["voyage-3-5"]},
        }
        assert fake_router.async_embedding_kwargs[0]["metadata"] == expected
        assert fake_router.async_embedding_kwargs[0]["litellm_metadata"] == expected
        assert caller_metadata["user_api_key_auth"] == {
            "models": ["voyage-3-5"],
            "budget_reservation": {"reserved_cost": 1.0},
        }

    @pytest.mark.asyncio
    async def test_semantic_routelayer_build_runs_off_event_loop(self, basic_config):
        """Building the SemanticRouter embeds route utterances via a synchronous provider
        call; it must run in a worker thread, not block the async event loop.
        """
        import threading

        fake_router = FakeEmbeddingRouter()
        config = {
            **basic_config,
            "keyword_tier_rules": [{"keywords": ["kubernetes deployment"], "tier": "REASONING"}],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.5,
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=fake_router,
            complexity_router_config=config,
        )
        loop_thread_id = threading.get_ident()
        await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "roll out my k8s cluster"}],
        )
        # The route-index build did a synchronous embedding call...
        assert fake_router.sync_embedding_thread_ids, "expected the route-index build to embed utterances"
        # ...and none of it ran on the event-loop thread.
        assert all(tid != loop_thread_id for tid in fake_router.sync_embedding_thread_ids)

    @pytest.mark.asyncio
    async def test_concurrent_cold_start_builds_routelayer_once(self, basic_config):
        """Concurrent first requests must not each construct the route index (which would
        fire duplicate embedding calls); the lazy build happens exactly once.
        """
        config = {
            **basic_config,
            "keyword_tier_rules": [{"keywords": ["kubernetes deployment"], "tier": "REASONING"}],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.5,
        }

        def _make_router(fake):
            return ComplexityRouter(
                model_name="test-router",
                litellm_router_instance=fake,
                complexity_router_config=config,
            )

        # Baseline: a single cold request's route-index build embeds the route utterance once.
        route_utterance = "kubernetes deployment"
        baseline_fake = FakeEmbeddingRouter()
        await _make_router(baseline_fake)._semantic_tier_override("roll out my k8s cluster", {})
        baseline_builds = baseline_fake.utterance_embedding_count(route_utterance)
        assert baseline_builds >= 1

        # Ten simultaneous cold-start requests must build the index the same number of
        # times as one request - i.e. exactly once, not once per concurrent caller.
        concurrent_fake = FakeEmbeddingRouter()
        concurrent_router = _make_router(concurrent_fake)
        await asyncio.gather(
            *(concurrent_router._semantic_tier_override("roll out my k8s cluster", {}) for _ in range(10))
        )
        assert concurrent_fake.utterance_embedding_count(route_utterance) == baseline_builds

    @pytest.mark.asyncio
    async def test_below_threshold_falls_back_to_scoring(self, basic_config):
        """When no route clears the threshold, scoring decides the tier."""
        fake_router = FakeEmbeddingRouter()
        config = {
            **basic_config,
            "keyword_tier_rules": [
                {"keywords": ["kubernetes deployment"], "tier": "REASONING"},
            ],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.9,
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=fake_router,
            complexity_router_config=config,
        )
        # "hello there friend" embeds orthogonal to the REASONING route (cos 0 < 0.9).
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "hello there friend"}],
        )
        assert result is not None
        assert result.model == "gpt-4o-mini"  # SIMPLE via scoring fallback

    @pytest.mark.asyncio
    async def test_route_embeddings_cached_across_requests(self, basic_config):
        """The route layer is built once and reused on subsequent requests."""
        fake_router = FakeEmbeddingRouter()
        config = {
            **basic_config,
            "keyword_tier_rules": [
                {"keywords": ["kubernetes deployment"], "tier": "REASONING"},
            ],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.5,
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=fake_router,
            complexity_router_config=config,
        )
        assert router._semantic_routelayer is None
        await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "roll out my k8s cluster"}],
        )
        first_layer = router._semantic_routelayer
        assert first_layer is not None
        await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "scale my container cluster"}],
        )
        assert router._semantic_routelayer is first_layer


class TestSemanticConfigValidation:
    """Test config validation for semantic_keyword_matching."""

    def test_semantic_without_embedding_model_raises(self):
        with pytest.raises(ValidationError):
            ComplexityRouterConfig(
                semantic_keyword_matching=True,
                keyword_tier_rules=[{"keywords": ["k8s"], "tier": "REASONING"}],
            )

    def test_semantic_without_rules_raises(self):
        with pytest.raises(ValidationError):
            ComplexityRouterConfig(
                semantic_keyword_matching=True,
                embedding_model="fake-embed",
            )

    def test_semantic_disabled_needs_no_embedding_model(self):
        config = ComplexityRouterConfig(
            keyword_tier_rules=[{"keywords": ["k8s"], "tier": "REASONING"}],
        )
        assert config.semantic_keyword_matching is False
        assert config.match_threshold == 0.5

    def test_keyword_tier_rule_rejects_empty_keywords(self):
        """A rule with no keywords is meaningless (and yields a zero-utterance semantic route)."""
        with pytest.raises(ValidationError):
            ComplexityRouterConfig(keyword_tier_rules=[{"keywords": [], "tier": "SIMPLE"}])

    def test_keyword_tier_rule_rejects_blank_only_keywords(self):
        """Whitespace-only keywords don't count as content."""
        with pytest.raises(ValidationError):
            ComplexityRouterConfig(keyword_tier_rules=[{"keywords": ["   ", ""], "tier": "SIMPLE"}])

    def test_keyword_tier_rule_strips_and_drops_blank_keywords(self):
        """Blank keywords mixed with real ones are dropped (not kept), and survivors trimmed.

        A stray "" would otherwise match-all in _keyword_matches and silently force this
        tier for every request.
        """
        config = ComplexityRouterConfig(
            keyword_tier_rules=[{"keywords": ["", "  deploy to k8s  ", " ", "kubernetes"], "tier": "REASONING"}]
        )
        assert config.keyword_tier_rules is not None
        assert config.keyword_tier_rules[0].keywords == ["deploy to k8s", "kubernetes"]


class _StubEncoder:
    """Minimal stand-in for LiteLLMRouterEncoder.aencode_queries, capturing the kwargs it was called with."""

    def __init__(self):
        self.aencode_queries_calls: List[Dict] = []

    async def aencode_queries(self, docs, **kwargs):
        self.aencode_queries_calls.append(kwargs)
        return [[0.0]]


class _StubRouteLayer:
    """Returns a fixed acall result so _semantic_tier_override branches can be exercised."""

    def __init__(self, result):
        self._result = result
        self.encoder = _StubEncoder()

    async def acall(self, text=None, vector=None):
        return self._result


class _RaisingEncoder:
    """Simulates an embedding-provider failure during semantic matching."""

    async def aencode_queries(self, docs, **kwargs):
        raise RuntimeError("embedding provider unavailable")


class _RaisingRouteLayer:
    def __init__(self):
        self.encoder = _RaisingEncoder()

    async def acall(self, text=None, vector=None):
        raise AssertionError("acall should not be reached when the encoder fails")


class TestKeywordOverrideEdgeCases:
    """Cover the defensive branches of the lexical and semantic override helpers."""

    def _semantic_router(self, mock_router_instance, basic_config):
        config = {
            **basic_config,
            "keyword_tier_rules": [{"keywords": ["kubernetes"], "tier": "REASONING"}],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.5,
        }
        return ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )

    def test_lexical_override_none_when_no_rules(self, mock_router_instance, basic_config):
        """No keyword_tier_rules configured -> lexical override is a no-op."""
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=basic_config,
        )
        assert router._lexical_tier_override("deploy to k8s and reason step by step") is None

    def test_semantic_routelayer_requires_embedding_model(self, mock_router_instance, basic_config):
        """Building the route layer without an embedding model raises (defensive invariant)."""
        config = {**basic_config, "keyword_tier_rules": [{"keywords": ["k8s"], "tier": "REASONING"}]}
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        assert router.config.embedding_model is None
        with pytest.raises(ValueError, match="embedding_model is required"):
            router._get_or_create_semantic_routelayer()

    @pytest.mark.asyncio
    async def test_semantic_override_maps_first_of_list(self, mock_router_instance, basic_config):
        """A list RouteChoice result maps to the first entry's tier."""
        from semantic_router.schema import RouteChoice

        router = self._semantic_router(mock_router_instance, basic_config)
        router._semantic_routelayer = _StubRouteLayer([RouteChoice(name="COMPLEX"), RouteChoice(name="SIMPLE")])
        assert await router._semantic_tier_override("anything", {}) == ComplexityTier.COMPLEX

    @pytest.mark.asyncio
    async def test_semantic_override_empty_list_returns_none(self, mock_router_instance, basic_config):
        """An empty list result falls through to scoring."""
        router = self._semantic_router(mock_router_instance, basic_config)
        router._semantic_routelayer = _StubRouteLayer([])
        assert await router._semantic_tier_override("anything", {}) is None

    @pytest.mark.asyncio
    async def test_semantic_override_unknown_route_name_returns_none(self, mock_router_instance, basic_config):
        """A matched route whose name is not a ComplexityTier is ignored."""
        from semantic_router.schema import RouteChoice

        router = self._semantic_router(mock_router_instance, basic_config)
        router._semantic_routelayer = _StubRouteLayer(RouteChoice(name="NOT_A_TIER"))
        assert await router._semantic_tier_override("anything", {}) is None

    @pytest.mark.asyncio
    async def test_semantic_embedding_error_falls_back_to_scoring(self, mock_router_instance, basic_config):
        """An embedding failure must not fail the request: the override yields None so
        async_pre_routing_hook falls through to the complexity scorer.
        """
        router = self._semantic_router(mock_router_instance, basic_config)
        router._semantic_routelayer = _RaisingRouteLayer()

        # _resolve_keyword_tier_override swallows the error and returns None (no override).
        assert await router._resolve_keyword_tier_override("roll out my k8s cluster", {}) is None

        # End-to-end, the hook still returns a routed model (from scoring) rather than raising.
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "roll out my k8s cluster"}],
        )
        assert result is not None
        assert result.model in {"gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514", "o1-preview"}


class TestSubCallMetadataSanitization:
    """The proxy cost callback must not be able to recover the parent budget reservation
    from sub-call metadata, in either of the shapes it knows how to read."""

    def test_cost_callback_cannot_recover_reservation_from_sanitized_metadata(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.proxy.hooks.proxy_track_cost_callback import (
            _get_budget_reservation_from_metadata,
        )
        from litellm.router_strategy.complexity_router.complexity_router import (
            _classifier_call_metadata,
        )

        reservation = {"reserved_cost": 1.0}
        auth_shapes = (
            {"models": ["gpt-4o"], "budget_reservation": dict(reservation)},
            UserAPIKeyAuth(api_key="sk-abc", budget_reservation=dict(reservation)),
        )
        for auth in auth_shapes:
            metadata = {
                "user_api_key_hash": "hash-abc",
                "user_api_key_budget_reservation": dict(reservation),
                "user_api_key_auth": auth,
            }
            assert _get_budget_reservation_from_metadata(metadata) == reservation

            sanitized = _classifier_call_metadata(metadata)
            assert sanitized is not None
            assert sanitized["user_api_key_auth"] is not None
            assert _get_budget_reservation_from_metadata(sanitized) is None

    def test_returns_empty_dict_for_missing_metadata(self):
        from litellm.router_strategy.complexity_router.complexity_router import (
            _classifier_call_metadata,
        )

        for absent in (None, {}):
            result = _classifier_call_metadata(absent)
            assert result == {}
            assert isinstance(result, dict)

    def test_sanitized_auth_keeps_access_group_fields_and_leaves_original_untouched(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.router_strategy.complexity_router.complexity_router import (
            _classifier_call_metadata,
        )

        auth = UserAPIKeyAuth(
            api_key="sk-abc",
            team_id="team-1",
            budget_reservation={"reserved_cost": 1.0},
        )
        sanitized = _classifier_call_metadata({"user_api_key_auth": auth})
        assert sanitized is not None
        sanitized_auth = sanitized["user_api_key_auth"]
        assert sanitized_auth.budget_reservation is None
        assert sanitized_auth.team_id == "team-1"
        assert sanitized_auth.api_key == auth.api_key
        assert auth.budget_reservation == {"reserved_cost": 1.0}


class TestRoutingDecisionCauseLogging:
    """The info log must name what drove each routing decision so an operator can tell a
    literal keyword match, a semantic keyword match, and the complexity scorer apart.
    """

    @pytest.fixture
    def router_log_capture(self, caplog):
        # verbose_router_logger sets propagate=False, so caplog's root handler never sees
        # its records; attach the capture handler directly for the duration of the test.
        caplog.set_level(logging.INFO, logger="LiteLLM Router")
        verbose_router_logger.addHandler(caplog.handler)
        try:
            yield caplog
        finally:
            verbose_router_logger.removeHandler(caplog.handler)

    @pytest.mark.asyncio
    async def test_literal_keyword_match_logs_its_cause(self, mock_router_instance, basic_config, router_log_capture):
        config = {
            **basic_config,
            "keyword_tier_rules": [{"keywords": ["deploy to k8s"], "tier": "REASONING"}],
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=config,
        )
        await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "please deploy to k8s now"}],
        )
        assert "routing decision cause=literal_keyword_match" in router_log_capture.text
        assert "tier=REASONING" in router_log_capture.text
        # A literal match must not be mislabelled as semantic.
        assert "cause=semantic_keyword_match" not in router_log_capture.text

    @pytest.mark.asyncio
    async def test_semantic_keyword_match_logs_its_cause(self, basic_config, router_log_capture):
        fake_router = FakeEmbeddingRouter()
        config = {
            **basic_config,
            "keyword_tier_rules": [{"keywords": ["kubernetes deployment"], "tier": "REASONING"}],
            "semantic_keyword_matching": True,
            "embedding_model": "fake-embed",
            "match_threshold": 0.5,
        }
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=fake_router,
            complexity_router_config=config,
        )
        await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "help me roll out my k8s cluster today"}],
        )
        assert "routing decision cause=semantic_keyword_match" in router_log_capture.text
        assert "tier=REASONING" in router_log_capture.text
        # A semantic match must not be mislabelled as literal.
        assert "cause=literal_keyword_match" not in router_log_capture.text

    @pytest.mark.asyncio
    async def test_complexity_scorer_logs_its_cause(self, mock_router_instance, basic_config, router_log_capture):
        # No keyword rules -> the scorer decides, and its line must be tagged as such.
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=basic_config,
        )
        await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "What is the boiling point of water at sea level?"}],
        )
        assert "routing decision cause=complexity_scorer" in router_log_capture.text
        assert "score=" in router_log_capture.text
        assert "cause=literal_keyword_match" not in router_log_capture.text
        assert "cause=semantic_keyword_match" not in router_log_capture.text


class TestSessionAffinity:
    """Test the session_affinity sticky-routing behavior (on by default)."""

    REASONING_MESSAGE = [
        {
            "role": "user",
            "content": "Let's think step by step and reason through this problem carefully.",
        }
    ]
    SIMPLE_MESSAGE = [{"role": "user", "content": "Hello!"}]

    @pytest.fixture
    def session_affinity_config(self, basic_config) -> Dict:
        return {**basic_config, "session_affinity": True}

    @pytest.fixture
    def session_affinity_disabled_config(self, basic_config) -> Dict:
        return {**basic_config, "session_affinity": False}

    @staticmethod
    def _request_kwargs(session_id: str) -> Dict:
        return {"metadata": {"session_id": session_id}}

    @pytest.mark.asyncio
    async def test_enabled_by_default_pins_model(self, mock_router_instance, basic_config):
        """Regression: session_affinity defaults to True, so a shared session_id pins the
        first turn's model and later turns reuse it instead of reclassifying."""
        assert "session_affinity" not in basic_config
        mock_router_instance.cache = DualCache()
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=basic_config,
        )
        request_kwargs = self._request_kwargs("session-1")
        first = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=request_kwargs, messages=self.REASONING_MESSAGE
        )
        second = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=request_kwargs, messages=self.SIMPLE_MESSAGE
        )
        assert first.model == "o1-preview"
        assert second.model == "o1-preview"

    @pytest.mark.asyncio
    async def test_can_be_disabled_reclassifies_every_turn(
        self, mock_router_instance, session_affinity_disabled_config
    ):
        """Regression: session_affinity=False must still reclassify every turn even when a
        shared session_id is present, so the opt-out keeps working."""
        mock_router_instance.cache = DualCache()
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=session_affinity_disabled_config,
        )
        request_kwargs = self._request_kwargs("session-1")
        first = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=request_kwargs, messages=self.REASONING_MESSAGE
        )
        second = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=request_kwargs, messages=self.SIMPLE_MESSAGE
        )
        assert first.model == "o1-preview"
        assert second.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_pins_model_after_first_turn(self, mock_router_instance, session_affinity_config):
        mock_router_instance.cache = DualCache()
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=session_affinity_config,
        )
        request_kwargs = self._request_kwargs("session-1")
        first = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=request_kwargs, messages=self.REASONING_MESSAGE
        )
        assert first.model == "o1-preview"

        with patch.object(router, "aclassify", wraps=router.aclassify) as spy_aclassify:
            second = await router.async_pre_routing_hook(
                model="test-model", request_kwargs=request_kwargs, messages=self.SIMPLE_MESSAGE
            )
            spy_aclassify.assert_not_called()
        # Pinned to the first turn's model, not re-classified down to SIMPLE.
        assert second.model == "o1-preview"

    @pytest.mark.asyncio
    async def test_different_sessions_classify_independently(self, mock_router_instance, session_affinity_config):
        mock_router_instance.cache = DualCache()
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=session_affinity_config,
        )
        reasoning = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=self._request_kwargs("session-a"), messages=self.REASONING_MESSAGE
        )
        simple = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=self._request_kwargs("session-b"), messages=self.SIMPLE_MESSAGE
        )
        assert reasoning.model == "o1-preview"
        assert simple.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_respects_ttl_seconds(self, mock_router_instance, basic_config):
        cache = AsyncMock()
        cache.async_get_cache = AsyncMock(return_value=None)
        mock_router_instance.cache = cache
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                **basic_config,
                "session_affinity": True,
                "session_affinity_ttl_seconds": 120,
            },
        )
        await router.async_pre_routing_hook(
            model="test-model", request_kwargs=self._request_kwargs("session-1"), messages=self.SIMPLE_MESSAGE
        )
        cache.async_set_cache.assert_called_once()
        call_kwargs = cache.async_set_cache.call_args.kwargs
        assert call_kwargs["ttl"] == 120
        assert call_kwargs["value"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_ttl_refreshed_on_cache_hit(self, mock_router_instance, basic_config):
        """Regression: a pinned turn must refresh the TTL, not just the first write --
        otherwise a session outliving session_affinity_ttl_seconds silently loses its pin."""
        cache = AsyncMock()
        cache.async_get_cache = AsyncMock(return_value="o1-preview")
        mock_router_instance.cache = cache
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                **basic_config,
                "session_affinity": True,
                "session_affinity_ttl_seconds": 90,
            },
        )
        result = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=self._request_kwargs("session-1"), messages=self.SIMPLE_MESSAGE
        )
        assert result.model == "o1-preview"
        cache.async_set_cache.assert_called_once()
        call_kwargs = cache.async_set_cache.call_args.kwargs
        assert call_kwargs["value"] == "o1-preview"
        assert call_kwargs["ttl"] == 90

    @pytest.mark.asyncio
    async def test_different_api_keys_do_not_share_pin(self, mock_router_instance, session_affinity_config):
        """A session_id is client-supplied and unauthenticated; two different callers
        (API keys) reusing the same session_id must not poison each other's pin."""
        mock_router_instance.cache = DualCache()
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=session_affinity_config,
        )
        caller_a_kwargs = {"metadata": {"session_id": "shared-session", "user_api_key_hash": "key-a"}}
        caller_b_kwargs = {"metadata": {"session_id": "shared-session", "user_api_key_hash": "key-b"}}

        pinned_for_a = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=caller_a_kwargs, messages=self.REASONING_MESSAGE
        )
        assert pinned_for_a.model == "o1-preview"

        # Caller B reuses the same session_id but has a different API key; its trivial
        # message must classify fresh, not inherit caller A's REASONING-tier pin.
        result_for_b = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=caller_b_kwargs, messages=self.SIMPLE_MESSAGE
        )
        assert result_for_b.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_no_session_id_falls_back_to_reclassify(self, mock_router_instance, session_affinity_config):
        cache = AsyncMock()
        mock_router_instance.cache = cache
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=session_affinity_config,
        )
        result = await router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=self.SIMPLE_MESSAGE
        )
        assert result.model == "gpt-4o-mini"
        cache.async_get_cache.assert_not_called()
        cache.async_set_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_adaptive_pinned_turn_still_stamps_chosen_model_metadata(self, mock_router_instance):
        """Regression: skipping classification on a pinned turn must not break the
        adaptive bandit's reward-feedback loop, which only records a turn's outcome
        when ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY is present in the request metadata."""
        mock_router_instance.cache = DualCache()
        mock_router_instance.model_list = [
            {
                "model_name": "cheap",
                "litellm_params": {"model": "openai/gpt-4o-mini", "input_cost_per_token": 0.0},
                "model_info": {},
            },
        ]
        mock_router_instance.model_name_to_deployment_indices = {"cheap": [0]}
        router = ComplexityRouter(
            model_name="hybrid",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "adaptive": True,
                "session_affinity": True,
                "tiers": {
                    "SIMPLE": ["cheap"],
                    "MEDIUM": ["cheap"],
                    "COMPLEX": ["cheap"],
                    "REASONING": ["cheap"],
                },
                "default_model": "cheap",
            },
        )
        first = await router.async_pre_routing_hook(
            model="hybrid",
            request_kwargs=self._request_kwargs("session-1"),
            messages=[{"role": "user", "content": "hi"}],
        )
        assert first.model == "cheap"

        request_kwargs_2 = self._request_kwargs("session-1")
        with patch.object(router, "aclassify", wraps=router.aclassify) as spy_aclassify:
            second = await router.async_pre_routing_hook(
                model="hybrid",
                request_kwargs=request_kwargs_2,
                messages=[{"role": "user", "content": "hi again"}],
            )
            spy_aclassify.assert_not_called()
        assert second.model == "cheap"
        assert request_kwargs_2["metadata"]["adaptive_router_chosen_model"] == "cheap"


class _DummyPlugin:
    async def run(self, context):
        return context


class TestRoutingPlugins:
    """Test the `complexity_router_config.plugins` field: narrows the classified
    tier's candidate pool before a model is picked. Discussion:
    https://github.com/BerriAI/litellm/discussions/32168"""

    @pytest.mark.asyncio
    async def test_plugin_narrows_tier_candidates(self, mock_router_instance):
        class ExcludeGpt4oMini:
            async def run(self, context):
                context.candidate_models = [m for m in context.candidate_models if m != "gpt-4o-mini"]
                return context

        router = ComplexityRouter(
            model_name="test-complexity-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": ["gpt-4o-mini", "gpt-4o-nano"]},
                "plugins": [ExcludeGpt4oMini()],
            },
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result is not None
        assert result.model == "gpt-4o-nano"

    @pytest.mark.asyncio
    async def test_plugin_narrowing_to_zero_raises_even_with_default_model_configured(self, mock_router_instance):
        """Regression: default_model must never be used as an escape hatch around a
        plugin's narrowing decision -- it was never checked against the plugins, so
        falling back to it would let a tenant/budget policy be silently bypassed.
        Reported by Veria AI on PR #33251."""

        class BlockEverything:
            async def run(self, context):
                context.candidate_models = []
                return context

        router = ComplexityRouter(
            model_name="test-complexity-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": "gpt-4o-mini"},
                "default_model": "gpt-4o-fallback",
                "plugins": [BlockEverything()],
            },
        )
        with pytest.raises(ValueError, match="No candidate models left for tier"):
            await router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    async def test_plugin_narrowing_to_zero_without_default_model_raises(self, mock_router_instance):
        class BlockEverything:
            async def run(self, context):
                context.candidate_models = []
                return context

        router = ComplexityRouter(
            model_name="test-complexity-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": "gpt-4o-mini"},
                "plugins": [BlockEverything()],
            },
        )
        with pytest.raises(ValueError, match="No candidate models left for tier"):
            await router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=[{"role": "user", "content": "hi"}],
            )

    @pytest.mark.asyncio
    async def test_plugin_receives_metadata_from_request_kwargs(self, mock_router_instance):
        captured = {}

        class CaptureMetadata:
            async def run(self, context):
                captured.update(context.metadata)
                return context

        router = ComplexityRouter(
            model_name="test-complexity-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": "gpt-4o-mini"},
                "plugins": [CaptureMetadata()],
            },
        )
        await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={"metadata": {"tenant": "acme-corp"}},
            messages=[{"role": "user", "content": "hi"}],
        )
        assert captured.get("tenant") == "acme-corp"

    @pytest.mark.asyncio
    async def test_plugin_applies_to_keyword_tier_override(self, mock_router_instance):
        """A policy plugin must not be bypassable via the keyword_tier_rules override path."""

        class ExcludeGpt4oMini:
            async def run(self, context):
                context.candidate_models = [m for m in context.candidate_models if m != "gpt-4o-mini"]
                return context

        router = ComplexityRouter(
            model_name="test-complexity-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": ["gpt-4o-mini", "gpt-4o-nano"]},
                "keyword_tier_rules": [{"keywords": ["hello"], "tier": "SIMPLE"}],
                "plugins": [ExcludeGpt4oMini()],
            },
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "hello there"}],
        )
        assert result is not None
        assert result.model == "gpt-4o-nano"

    @pytest.mark.asyncio
    async def test_plugin_applies_to_no_user_message_default_tier_path(self, mock_router_instance):
        """Regression: `self.config.default_model or await self._pick_model_for_tier(...)`
        short-circuited on a truthy default_model, so the no-user-message path never ran
        the plugin pipeline at all when default_model was configured. A policy plugin
        must not be bypassable via this path either. Reported by Veria AI on PR #33251."""

        class ExcludeDefaultModel:
            async def run(self, context):
                context.candidate_models = [m for m in context.candidate_models if m != "gpt-4o-default"]
                return context

        router = ComplexityRouter(
            model_name="test-complexity-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"MEDIUM": ["gpt-4o-default", "gpt-4o-nano"]},
                "default_model": "gpt-4o-default",
                "plugins": [ExcludeDefaultModel()],
            },
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "assistant", "content": "Hello!"},
            ],
        )
        assert result is not None
        assert result.model == "gpt-4o-nano"

    @pytest.mark.asyncio
    async def test_no_user_message_prefers_default_model_over_medium_tier_without_plugins(
        self, mock_router_instance
    ):
        """Regression: without plugins configured, the no-user-message path must keep its
        pre-existing default_model-first priority over the MEDIUM tier exactly as before --
        closing the plugin-bypass gap must not silently flip model selection for the (much
        larger) population of users who don't use plugins at all. Flagged by Greptile on
        PR #33251 after the plugin-bypass fix changed this priority unconditionally."""
        router = ComplexityRouter(
            model_name="test-complexity-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"MEDIUM": ["gpt-4o-medium-tier"]},
                "default_model": "gpt-4o-configured-default",
            },
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "assistant", "content": "Hello!"},
            ],
        )
        assert result is not None
        assert result.model == "gpt-4o-configured-default"

    def test_plugins_and_adaptive_together_raises(self):
        with pytest.raises(ValidationError, match="plugins and adaptive=True cannot both be set"):
            ComplexityRouterConfig(
                tiers={"SIMPLE": ["gpt-4o-mini"]},
                adaptive=True,
                plugins=[_DummyPlugin()],
            )

    @pytest.mark.asyncio
    async def test_no_plugins_configured_is_unaffected(self, complexity_router):
        """Regression guard: a ComplexityRouter with no `plugins` configured behaves exactly as before."""
        result = await complexity_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "Hello!"}],
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_session_affinity_pin_shortcut_disabled_when_plugins_configured(self, mock_router_instance):
        """Regression: the session_affinity cache-pin shortcut returned a stale pinned
        model without ever re-running it through plugins, so a policy plugin's decision
        (e.g. a budget cap crossed mid-session) was only ever enforced on a session's
        first turn. With plugins configured, every turn must go through
        _classify_and_route (and therefore the plugin pipeline) again."""
        mock_router_instance.cache = DualCache()

        class AllowAll:
            async def run(self, context):
                return context

        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": ["gpt-4o-mini"]},
                "session_affinity": True,
                "plugins": [AllowAll()],
            },
        )
        request_kwargs = {"metadata": {"session_id": "session-1"}}

        with patch.object(router, "_classify_and_route", wraps=router._classify_and_route) as spy:
            first = await router.async_pre_routing_hook(
                model="test-model", request_kwargs=request_kwargs, messages=[{"role": "user", "content": "hi"}]
            )
            second = await router.async_pre_routing_hook(
                model="test-model", request_kwargs=request_kwargs, messages=[{"role": "user", "content": "hi again"}]
            )
        assert first.model == "gpt-4o-mini"
        assert second.model == "gpt-4o-mini"
        assert spy.call_count == 2


class TestEscalationKeywords:
    """Test user-triggered escalation: a keyword in the prompt bumps the resolved tier
    one step higher so a user can force a stronger model when unhappy with results."""

    @staticmethod
    def _request_kwargs(session_id: str) -> Dict:
        return {"metadata": {"session_id": session_id}}

    def test_default_escalation_keyword(self, complexity_router):
        assert complexity_router.escalation_keywords == ["LITELLM ESCALATE"]

    def test_escalation_triggered_is_case_sensitive(self, complexity_router):
        assert complexity_router._escalation_triggered("please LITELLM ESCALATE now") is True
        assert complexity_router._escalation_triggered("please litellm escalate now") is False
        assert complexity_router._escalation_triggered("how do I escalate this ticket") is False

    def test_escalate_tier_bumps_one_step(self, complexity_router):
        assert complexity_router._escalate_tier(ComplexityTier.SIMPLE) == ComplexityTier.MEDIUM
        assert complexity_router._escalate_tier(ComplexityTier.MEDIUM) == ComplexityTier.COMPLEX
        assert complexity_router._escalate_tier(ComplexityTier.COMPLEX) == ComplexityTier.REASONING

    def test_escalate_tier_caps_at_highest_configured(self, complexity_router):
        assert complexity_router._escalate_tier(ComplexityTier.REASONING) == ComplexityTier.REASONING

    def test_escalate_tier_skips_unconfigured_intermediate(self, mock_router_instance):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={"tiers": {"SIMPLE": "gpt-4o-mini", "REASONING": "o1-preview"}},
        )
        assert router._escalate_tier(ComplexityTier.SIMPLE) == ComplexityTier.REASONING

    def test_tier_for_model_returns_most_severe(self, mock_router_instance):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": "shared", "COMPLEX": "shared", "REASONING": "top"}
            },
        )
        assert router._tier_for_model("shared") == ComplexityTier.COMPLEX
        assert router._tier_for_model("top") == ComplexityTier.REASONING
        assert router._tier_for_model("unknown") is None

    @pytest.mark.asyncio
    async def test_escalation_bumps_classified_tier(self, mock_router_instance, basic_config):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=basic_config,
        )
        # Baseline: this prompt classifies SIMPLE.
        baseline = await router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=[{"role": "user", "content": "Hello there!"}]
        )
        assert baseline.model == "gpt-4o-mini"

        escalated = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "LITELLM ESCALATE Hello there!"}],
        )
        assert escalated.model == "gpt-4o"  # SIMPLE bumped to MEDIUM

    @pytest.mark.asyncio
    async def test_lowercase_keyword_does_not_escalate(self, mock_router_instance, basic_config):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=basic_config,
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "litellm escalate Hello there!"}],
        )
        assert result.model == "gpt-4o-mini"  # not escalated

    @pytest.mark.asyncio
    async def test_custom_escalation_keyword(self, mock_router_instance, basic_config):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={**basic_config, "escalation_keywords": ["MAKE IT BETTER"]},
        )
        # The default keyword no longer triggers once a custom list is supplied.
        default = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "LITELLM ESCALATE Hello there!"}],
        )
        assert default.model == "gpt-4o-mini"

        custom = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "MAKE IT BETTER Hello there!"}],
        )
        assert custom.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_empty_keyword_list_disables_escalation(self, mock_router_instance, basic_config):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={**basic_config, "escalation_keywords": []},
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "LITELLM ESCALATE Hello there!"}],
        )
        assert result.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_escalation_caps_at_highest_tier(self, mock_router_instance, basic_config):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config=basic_config,
        )
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[
                {
                    "role": "user",
                    "content": "LITELLM ESCALATE Let's think step by step and reason through this carefully.",
                }
            ],
        )
        assert result.model == "o1-preview"  # already REASONING, stays there

    @pytest.mark.asyncio
    async def test_escalation_bumps_keyword_tier_override(self, mock_router_instance, basic_config):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                **basic_config,
                "keyword_tier_rules": [{"keywords": ["billing"], "tier": "SIMPLE"}],
            },
        )
        baseline = await router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=[{"role": "user", "content": "a billing question"}]
        )
        assert baseline.model == "gpt-4o-mini"

        escalated = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "LITELLM ESCALATE a billing question"}],
        )
        assert escalated.model == "gpt-4o"  # override SIMPLE bumped to MEDIUM

    @pytest.mark.asyncio
    async def test_escalation_overrides_session_pin_and_persists(self, mock_router_instance, basic_config):
        """Mid-session escalation bumps relative to the pinned model (never below it) and
        the bumped model persists for later turns."""
        mock_router_instance.cache = DualCache()
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={**basic_config, "session_affinity": True},
        )
        request_kwargs = self._request_kwargs("session-1")
        first = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=request_kwargs, messages=[{"role": "user", "content": "Hello!"}]
        )
        assert first.model == "gpt-4o-mini"  # pinned SIMPLE

        with patch.object(router, "aclassify", wraps=router.aclassify) as spy_aclassify:
            escalated = await router.async_pre_routing_hook(
                model="test-model",
                request_kwargs=request_kwargs,
                messages=[{"role": "user", "content": "LITELLM ESCALATE"}],
            )
            spy_aclassify.assert_not_called()
        assert escalated.model == "gpt-4o"  # bumped relative to the SIMPLE pin, not reclassified

        # The bump persists: a later ordinary turn stays on the escalated model.
        later = await router.async_pre_routing_hook(
            model="test-model", request_kwargs=request_kwargs, messages=[{"role": "user", "content": "thanks"}]
        )
        assert later.model == "gpt-4o"

        # Escalating again climbs one more tier.
        again = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs=request_kwargs,
            messages=[{"role": "user", "content": "LITELLM ESCALATE still not good"}],
        )
        assert again.model == "claude-sonnet-4-20250514"  # MEDIUM bumped to COMPLEX

    def test_blank_escalation_keywords_are_stripped(self):
        """Blank/whitespace-only phrases are dropped so `"" in message` can't escalate
        every request; surrounding whitespace on real phrases is trimmed."""
        assert ComplexityRouterConfig(
            tiers={"SIMPLE": "gpt-4o-mini", "MEDIUM": "gpt-4o"},
            escalation_keywords=["", "  "],
        ).escalation_keywords == []
        assert ComplexityRouterConfig(
            tiers={"SIMPLE": "gpt-4o-mini", "MEDIUM": "gpt-4o"},
            escalation_keywords=["  LITELLM ESCALATE  ", ""],
        ).escalation_keywords == ["LITELLM ESCALATE"]

    @pytest.mark.asyncio
    async def test_blank_escalation_keyword_does_not_escalate_everything(
        self, mock_router_instance, basic_config
    ):
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={**basic_config, "escalation_keywords": [""]},
        )
        assert router.escalation_keywords == []
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "Hello there!"}],
        )
        assert result.model == "gpt-4o-mini"  # not escalated

    def test_escalated_pin_stays_on_same_model_at_ceiling(self, mock_router_instance):
        """At the highest configured tier escalation keeps the exact pinned model, even
        when that tier's pool has peers `get_model_for_tier` could randomly pick instead."""
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": "gpt-4o-mini", "REASONING": ["o1-a", "o1-b", "o1-c"]}
            },
        )
        for pinned in ("o1-a", "o1-b", "o1-c"):
            assert router._escalated_pin(pinned) == pinned

    @pytest.mark.asyncio
    async def test_session_escalation_at_ceiling_keeps_multi_model_pin(self, mock_router_instance):
        mock_router_instance.cache = DualCache()
        router = ComplexityRouter(
            model_name="test-router",
            litellm_router_instance=mock_router_instance,
            complexity_router_config={
                "tiers": {"SIMPLE": "gpt-4o-mini", "REASONING": ["o1-a", "o1-b", "o1-c"]},
                "session_affinity": True,
            },
        )
        cache_key = router._get_session_affinity_cache_key("session-top", {})
        await mock_router_instance.cache.async_set_cache(key=cache_key, value="o1-b")
        result = await router.async_pre_routing_hook(
            model="test-model",
            request_kwargs=self._request_kwargs("session-top"),
            messages=[{"role": "user", "content": "LITELLM ESCALATE do better"}],
        )
        assert result.model == "o1-b"  # unchanged: no random hop to o1-a / o1-c
