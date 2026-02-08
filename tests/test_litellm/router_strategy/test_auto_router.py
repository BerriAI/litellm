"""Tests for the regex + tier-based auto-router.

Covers:
- classify_task() with representative inputs for each category
- load_routing_config() with YAML files
- SmartRouter.resolve_route() end-to-end
- AutoRouter.async_pre_routing_hook() integration
- _parse_inline_config() with JSON and YAML strings
"""

import json
import os
import sys
import tempfile
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.router_strategy.auto_router.classifier import (
    ClassificationRule,
    TaskCategory,
    classify_task,
)
from litellm.router_strategy.auto_router.tiers import (
    ModelTier,
    RoutingConfig,
    TierConfig,
    load_routing_config,
)
from litellm.router_strategy.auto_router.smart_router import (
    RouteDecision,
    SmartRouter,
)
from litellm.router_strategy.auto_router.auto_router import AutoRouter


# ─── classify_task tests ────────────────────────────────────────────


class TestClassifyTask:
    """Test classify_task() with representative inputs for each category."""

    def test_empty_message(self):
        result = classify_task("")
        assert result.category == TaskCategory.HEARTBEAT
        assert result.confidence >= 0.9

    def test_whitespace_only(self):
        result = classify_task("   ")
        assert result.category == TaskCategory.HEARTBEAT

    def test_heartbeat_greeting(self):
        for msg in ["hi", "hello", "ping", "hey", "yo"]:
            result = classify_task(msg)
            assert result.category == TaskCategory.HEARTBEAT, f"Failed for: {msg}"

    def test_heartbeat_system(self):
        result = classify_task("read heartbeat.md")
        assert result.category == TaskCategory.HEARTBEAT

    def test_reasoning_math(self):
        result = classify_task("prove that the square root of 2 is irrational")
        assert result.category == TaskCategory.REASONING

    def test_reasoning_step_by_step(self):
        result = classify_task("step-by-step reasoning about this problem")
        assert result.category == TaskCategory.REASONING

    def test_reasoning_theorem(self):
        result = classify_task("explain the Pythagorean theorem")
        assert result.category == TaskCategory.REASONING

    def test_analysis_compare(self):
        result = classify_task("compare React and Vue for a new project")
        assert result.category == TaskCategory.ANALYSIS

    def test_analysis_pros_cons(self):
        result = classify_task("what are the pros and cons of microservices?")
        assert result.category == TaskCategory.ANALYSIS

    def test_coding_function(self):
        result = classify_task("write a Python function to sort a list")
        assert result.category == TaskCategory.CODING

    def test_coding_backtick(self):
        result = classify_task("fix this code: ```def foo(): pass```")
        assert result.category == TaskCategory.CODING

    def test_coding_debug(self):
        result = classify_task("I'm getting a traceback error in my script")
        assert result.category == TaskCategory.CODING

    def test_coding_git(self):
        result = classify_task("git rebase my feature branch onto main")
        assert result.category == TaskCategory.CODING

    def test_translation(self):
        result = classify_task("translate this to Spanish")
        assert result.category == TaskCategory.TRANSLATION

    def test_summarization(self):
        result = classify_task("summarize this article for me")
        assert result.category == TaskCategory.SUMMARIZATION

    def test_summarization_tldr(self):
        result = classify_task("tldr of this document")
        assert result.category == TaskCategory.SUMMARIZATION

    def test_creative_story(self):
        result = classify_task("write me a short story about a dragon")
        assert result.category == TaskCategory.CREATIVE

    def test_creative_poem(self):
        result = classify_task("compose a haiku about the ocean")
        assert result.category == TaskCategory.CREATIVE

    def test_lookup_what_is(self):
        result = classify_task("what is the capital of France?")
        assert result.category == TaskCategory.LOOKUP

    def test_lookup_who_is(self):
        result = classify_task("who is Alan Turing?")
        assert result.category == TaskCategory.LOOKUP

    def test_simple_chat_short(self):
        result = classify_task("tell me something interesting")
        assert result.category == TaskCategory.SIMPLE_CHAT

    def test_default_fallback_long(self):
        # Long message that doesn't match any specific pattern
        result = classify_task("a" * 100)
        assert result.category == TaskCategory.SIMPLE_CHAT
        assert result.confidence == 0.40

    def test_custom_rules_override(self):
        import re

        custom = [
            ClassificationRule(
                pattern=re.compile(r"deploy", re.IGNORECASE),
                category=TaskCategory.CODING,
                confidence=0.95,
                description="custom deploy rule",
            )
        ]
        result = classify_task("deploy the application", custom)
        assert result.category == TaskCategory.CODING
        assert result.confidence == 0.95


# ─── load_routing_config tests ──────────────────────────────────────


class TestLoadRoutingConfig:
    """Test load_routing_config() with YAML files."""

    def test_default_config(self):
        config = load_routing_config(None)
        assert isinstance(config, RoutingConfig)
        assert ModelTier.LOW in config.tier_models
        assert ModelTier.MID in config.tier_models
        assert ModelTier.TOP in config.tier_models

    def test_missing_file_returns_defaults(self):
        config = load_routing_config("/nonexistent/path/config.yaml")
        assert isinstance(config, RoutingConfig)
        assert config.tier_models[ModelTier.LOW].model == "deepseek/deepseek-chat-v3-0324"

    def test_yaml_file_overrides(self):
        yaml_content = """
tiers:
  low:
    model: "custom/cheap-model"
    max_cost_per_m_tokens: 0.5
  top:
    model: "custom/expensive-model"
    max_cost_per_m_tokens: 50.0
routing:
  coding: top
custom_patterns:
  - pattern: "kubernetes"
    category: "coding"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            config = load_routing_config(f.name)

        os.unlink(f.name)

        assert config.tier_models[ModelTier.LOW].model == "custom/cheap-model"
        assert config.tier_models[ModelTier.TOP].model == "custom/expensive-model"
        # MID should still be default
        assert config.tier_models[ModelTier.MID].model == "zai/glm-4.7"
        assert config.category_routing[TaskCategory.CODING] == ModelTier.TOP
        assert len(config.custom_patterns) == 1

    def test_yaml_with_fallback_models(self):
        yaml_content = """
tiers:
  mid:
    model: "custom/mid-model"
    max_cost_per_m_tokens: 3.0
    fallback_models:
      - "fallback/model-a"
      - "fallback/model-b"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            config = load_routing_config(f.name)

        os.unlink(f.name)

        assert config.tier_models[ModelTier.MID].model == "custom/mid-model"
        assert config.tier_models[ModelTier.MID].fallback_models == (
            "fallback/model-a",
            "fallback/model-b",
        )


# ─── SmartRouter.resolve_route tests ────────────────────────────────


class TestSmartRouter:
    """Test SmartRouter.resolve_route() end-to-end."""

    def test_heartbeat_routes_to_low(self):
        router = SmartRouter()
        decision = router.resolve_route(
            [{"role": "user", "content": "hi"}]
        )
        assert decision.tier == ModelTier.LOW
        assert decision.category == TaskCategory.HEARTBEAT

    def test_coding_routes_to_mid(self):
        router = SmartRouter()
        decision = router.resolve_route(
            [{"role": "user", "content": "write a Python function to parse JSON"}]
        )
        assert decision.tier == ModelTier.MID
        assert decision.category == TaskCategory.CODING

    def test_reasoning_routes_to_top(self):
        router = SmartRouter()
        decision = router.resolve_route(
            [{"role": "user", "content": "prove that P = NP"}]
        )
        assert decision.tier == ModelTier.TOP
        assert decision.category == TaskCategory.REASONING

    def test_custom_config_changes_model(self):
        config = RoutingConfig()
        config.tier_models[ModelTier.LOW] = TierConfig(
            model="my-custom/cheap-model",
            max_cost_per_m_tokens=0.5,
        )
        router = SmartRouter(routing_config=config)
        decision = router.resolve_route(
            [{"role": "user", "content": "hello"}]
        )
        assert decision.model == "my-custom/cheap-model"

    def test_extract_user_text_from_last_message(self):
        router = SmartRouter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "write a Python function"},
        ]
        decision = router.resolve_route(messages)
        assert decision.category == TaskCategory.CODING

    def test_extract_user_text_multipart(self):
        router = SmartRouter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "write a Python function"},
                    {"type": "image_url", "image_url": {"url": "http://example.com"}},
                ],
            }
        ]
        decision = router.resolve_route(messages)
        assert decision.category == TaskCategory.CODING

    def test_empty_messages_returns_heartbeat(self):
        router = SmartRouter()
        decision = router.resolve_route([])
        assert decision.category == TaskCategory.HEARTBEAT

    def test_specific_model_passthrough(self):
        router = SmartRouter()
        # Request a model that's a configured tier model
        decision = router.resolve_route(
            [{"role": "user", "content": "hi"}],
            requested_model="deepseek/deepseek-chat-v3-0324",
        )
        assert decision.model == "deepseek/deepseek-chat-v3-0324"

    def test_custom_patterns_from_config(self):
        config = RoutingConfig()
        config.custom_patterns = [
            {"pattern": r"deploy\s+to\s+production", "category": "coding"}
        ]
        router = SmartRouter(routing_config=config)
        decision = router.resolve_route(
            [{"role": "user", "content": "deploy to production now"}]
        )
        assert decision.category == TaskCategory.CODING

    def test_normalize_model_claude(self):
        router = SmartRouter()
        assert router._normalize_model("claude-opus-4-6") == "anthropic/claude-opus-4"
        assert router._normalize_model("claude-sonnet-4-5") == "anthropic/claude-sonnet-4-5"
        assert router._normalize_model("anthropic/claude-sonnet-4-5") == "anthropic/claude-sonnet-4-5"
        assert router._normalize_model("unknown-model") is None

    def test_infer_tier_from_model(self):
        router = SmartRouter()
        assert router._infer_tier_from_model("anthropic/claude-opus-4") == ModelTier.TOP
        assert router._infer_tier_from_model("deepseek/deepseek-chat") == ModelTier.LOW
        assert router._infer_tier_from_model("some/random-model") == ModelTier.MID


# ─── AutoRouter integration tests ───────────────────────────────────


class TestAutoRouter:
    """Test AutoRouter.async_pre_routing_hook() integration."""

    @pytest.fixture
    def mock_router_instance(self):
        router = MagicMock()
        return router

    def test_init_with_defaults(self, mock_router_instance):
        auto_router = AutoRouter(
            model_name="test-auto-router",
            default_model="gpt-4o-mini",
            litellm_router_instance=mock_router_instance,
        )
        assert auto_router.model_name == "test-auto-router"
        assert auto_router.default_model == "gpt-4o-mini"
        assert isinstance(auto_router.smart_router, SmartRouter)

    @pytest.mark.asyncio
    async def test_pre_routing_hook_returns_model(self, mock_router_instance):
        auto_router = AutoRouter(
            model_name="test-auto-router",
            default_model="gpt-4o-mini",
            litellm_router_instance=mock_router_instance,
        )
        messages = [{"role": "user", "content": "prove that 2+2=4"}]
        result = await auto_router.async_pre_routing_hook(
            model="auto", request_kwargs={}, messages=messages
        )
        assert result is not None
        assert result.model is not None
        assert result.messages == messages

    @pytest.mark.asyncio
    async def test_pre_routing_hook_none_messages(self, mock_router_instance):
        auto_router = AutoRouter(
            model_name="test-auto-router",
            default_model="gpt-4o-mini",
            litellm_router_instance=mock_router_instance,
        )
        result = await auto_router.async_pre_routing_hook(
            model="auto", request_kwargs={}, messages=None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_routing_hook_coding(self, mock_router_instance):
        auto_router = AutoRouter(
            model_name="test-auto-router",
            default_model="gpt-4o-mini",
            litellm_router_instance=mock_router_instance,
        )
        messages = [{"role": "user", "content": "write a Python function to sort a list"}]
        result = await auto_router.async_pre_routing_hook(
            model="auto", request_kwargs={}, messages=messages
        )
        assert result is not None
        # Should route to MID tier model
        assert result.model == "zai/glm-4.7"

    @pytest.mark.asyncio
    async def test_pre_routing_hook_heartbeat(self, mock_router_instance):
        auto_router = AutoRouter(
            model_name="test-auto-router",
            default_model="gpt-4o-mini",
            litellm_router_instance=mock_router_instance,
        )
        messages = [{"role": "user", "content": "hi"}]
        result = await auto_router.async_pre_routing_hook(
            model="auto", request_kwargs={}, messages=messages
        )
        assert result is not None
        # Should route to LOW tier model
        assert result.model == "deepseek/deepseek-chat-v3-0324"


# ─── _parse_inline_config tests ─────────────────────────────────────


class TestParseInlineConfig:
    """Test AutoRouter._parse_inline_config() with JSON and YAML strings."""

    def test_json_config(self):
        config_json = json.dumps(
            {
                "tiers": {
                    "low": {"model": "custom/low-model", "max_cost_per_m_tokens": 0.5}
                },
                "routing": {"coding": "top"},
            }
        )
        config = AutoRouter._parse_inline_config(config_json)
        assert config.tier_models[ModelTier.LOW].model == "custom/low-model"
        assert config.category_routing[TaskCategory.CODING] == ModelTier.TOP

    def test_yaml_config(self):
        config_yaml = """
tiers:
  top:
    model: "custom/top-model"
    max_cost_per_m_tokens: 100.0
routing:
  heartbeat: low
"""
        config = AutoRouter._parse_inline_config(config_yaml)
        assert config.tier_models[ModelTier.TOP].model == "custom/top-model"

    def test_invalid_config_returns_defaults(self):
        config = AutoRouter._parse_inline_config("not valid json or yaml [[")
        assert isinstance(config, RoutingConfig)
        # Should have defaults
        assert ModelTier.LOW in config.tier_models

    def test_json_with_custom_patterns(self):
        config_json = json.dumps(
            {
                "custom_patterns": [
                    {"pattern": r"deploy", "category": "coding"},
                    {"pattern": r"security\s+audit", "category": "analysis"},
                ]
            }
        )
        config = AutoRouter._parse_inline_config(config_json)
        assert len(config.custom_patterns) == 2

    def test_empty_json_object(self):
        config = AutoRouter._parse_inline_config("{}")
        assert isinstance(config, RoutingConfig)
        # Should have all defaults
        assert len(config.tier_models) == 3
