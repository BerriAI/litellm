import asyncio
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.router_strategy.auto_router.auto_router import AutoRouter

pytestmark_skip_beta = pytest.mark.skip(
    reason="Skipping auto router tests - beta feature"
)


class TestExtractTextFromMessages:
    """Tests for AutoRouter._extract_text_from_messages (no semantic_router dependency)."""

    def test_should_extract_content_from_simple_user_message(self):
        messages = [{"role": "user", "content": "Hello world"}]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "Hello world"

    def test_should_extract_last_user_message_from_tool_call_conversation(self):
        messages = [
            {"role": "user", "content": "What's the weather in NYC?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "72°F and sunny",
            },
            {"role": "user", "content": "Now tell me about London"},
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "Now tell me about London"

    def test_should_find_user_message_when_last_message_is_assistant_with_tool_calls(
        self,
    ):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "What's the weather?"

    def test_should_find_user_message_when_last_message_is_tool_response(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "72°F and sunny",
            },
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "What's the weather?"

    def test_should_handle_multimodal_content_list(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            }
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "What's in this image?"

    def test_should_handle_multimodal_content_with_multiple_text_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            }
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "First part Second part"

    def test_should_return_empty_string_when_user_content_is_none(self):
        messages = [{"role": "user", "content": None}]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == ""

    def test_should_return_empty_string_when_no_user_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == ""

    def test_should_return_empty_string_for_empty_messages_list(self):
        result = AutoRouter._extract_text_from_messages([])
        assert result == ""


def _routes(*names):
    routes = []
    for name in names:
        route = MagicMock()
        route.name = name
        routes.append(route)
    return routes


def _configure_candidates(router, group_to_model):
    router.model_list = [
        {"model_name": group, "litellm_params": {"model": model}} for group, model in group_to_model.items()
    ]
    router.model_name_to_deployment_indices = {group: [i] for i, group in enumerate(group_to_model)}


@pytest.fixture
def mock_router_instance():
    """Create a mock LiteLLM Router instance."""
    router = MagicMock()
    router.acompletion = AsyncMock()
    return router


@pytest.fixture
def mock_semantic_router():
    """Create a mock SemanticRouter instance."""
    mock_router = MagicMock()
    mock_route = MagicMock()
    mock_route.name = "test-route"
    mock_router.routes = [mock_route]
    return mock_router


@pytest.fixture
def mock_route_choice():
    """Create a mock RouteChoice instance."""
    mock_choice = MagicMock()
    mock_choice.name = "test-model"
    return mock_choice


@pytestmark_skip_beta
class TestAutoRouter:
    """Test class for AutoRouter methods."""

    @patch("semantic_router.routers.SemanticRouter")
    def test_init(self, mock_semantic_router_class, mock_router_instance):
        """Test that AutoRouter initializes correctly with all required parameters."""
        # Arrange
        mock_semantic_router_class.from_json.return_value = mock_semantic_router_class

        model_name = "test-auto-router"
        router_config_path = "test/path/router.json"
        default_model = "gpt-4o-mini"
        embedding_model = "text-embedding-model"

        # Act
        auto_router = AutoRouter(
            model_name=model_name,
            auto_router_config_path=router_config_path,
            default_model=default_model,
            embedding_model=embedding_model,
            litellm_router_instance=mock_router_instance,
        )

        # Assert
        assert auto_router.auto_router_config_path == router_config_path
        assert auto_router.auto_sync_value == AutoRouter.DEFAULT_AUTO_SYNC_VALUE
        assert auto_router.default_model == default_model
        assert auto_router.embedding_model == embedding_model
        assert auto_router.litellm_router_instance == mock_router_instance
        assert auto_router.routelayer is None
        mock_semantic_router_class.from_json.assert_called_once_with(router_config_path)

    @pytest.mark.asyncio
    @patch("semantic_router.routers.SemanticRouter")
    @patch("litellm.router_strategy.auto_router.litellm_encoder.LiteLLMRouterEncoder")
    async def test_async_pre_routing_hook_with_route_choice(
        self,
        mock_encoder_class,
        mock_semantic_router_class,
        mock_router_instance,
        mock_route_choice,
    ):
        """Test async_pre_routing_hook returns correct model when route is found."""
        # Arrange
        mock_loaded_router = MagicMock()
        mock_loaded_router.routes = ["route1", "route2"]
        mock_semantic_router_class.from_json.return_value = mock_loaded_router

        mock_routelayer = MagicMock()
        mock_routelayer.return_value = mock_route_choice
        mock_semantic_router_class.return_value = mock_routelayer

        auto_router = AutoRouter(
            model_name="test-auto-router",
            auto_router_config_path="test/path/router.json",
            default_model="gpt-4o-mini",
            embedding_model="text-embedding-model",
            litellm_router_instance=mock_router_instance,
        )

        messages = [{"role": "user", "content": "test message"}]

        # Act
        result = await auto_router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=messages
        )

        # Assert
        assert result is not None
        assert result.model == "test-model"  # Should use the route choice name
        assert result.messages == messages
        mock_routelayer.assert_called_once_with(text="test message")

    @pytest.mark.asyncio
    @patch("semantic_router.routers.SemanticRouter")
    @patch("litellm.router_strategy.auto_router.litellm_encoder.LiteLLMRouterEncoder")
    async def test_async_pre_routing_hook_with_list_route_choice(
        self,
        mock_encoder_class,
        mock_semantic_router_class,
        mock_router_instance,
        mock_route_choice,
    ):
        """Test async_pre_routing_hook handles list of RouteChoice objects correctly."""
        # Arrange
        mock_loaded_router = MagicMock()
        mock_loaded_router.routes = ["route1", "route2"]
        mock_semantic_router_class.from_json.return_value = mock_loaded_router

        mock_routelayer = MagicMock()
        mock_routelayer.return_value = [mock_route_choice]  # Return list
        mock_semantic_router_class.return_value = mock_routelayer

        auto_router = AutoRouter(
            model_name="test-auto-router",
            auto_router_config_path="test/path/router.json",
            default_model="gpt-4o-mini",
            embedding_model="text-embedding-model",
            litellm_router_instance=mock_router_instance,
        )

        messages = [{"role": "user", "content": "test message"}]

        # Act
        result = await auto_router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=messages
        )

        # Assert
        assert result is not None
        assert result.model == "test-model"
        assert result.messages == messages

    @pytest.mark.asyncio
    async def test_async_pre_routing_hook_no_messages(self, mock_router_instance):
        """Test async_pre_routing_hook returns None when no messages provided."""
        # Arrange
        with patch("semantic_router.routers.SemanticRouter"):
            auto_router = AutoRouter(
                model_name="test-auto-router",
                auto_router_config_path="test/path/router.json",
                default_model="gpt-4o-mini",
                embedding_model="text-embedding-model",
                litellm_router_instance=mock_router_instance,
            )

        # Act
        result = await auto_router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=None
        )

        # Assert
        assert result is None

    @patch("semantic_router.routers.SemanticRouter")
    def test_init_honors_configured_savings_baseline_model(self, mock_semantic_router_class, mock_router_instance):
        """An operator-configured baseline overrides the flagship default."""
        mock_semantic_router_class.from_json.return_value = mock_semantic_router_class

        auto_router = AutoRouter(
            model_name="test-auto-router",
            auto_router_config_path="test/path/router.json",
            default_model="gpt-4o-mini",
            embedding_model="text-embedding-model",
            litellm_router_instance=mock_router_instance,
            savings_baseline_model="claude-sonnet-5",
        )

        assert auto_router.savings_baseline_model == "claude-sonnet-5"

    @pytest.mark.asyncio
    @patch("semantic_router.routers.SemanticRouter")
    @patch("litellm.router_strategy.auto_router.litellm_encoder.LiteLLMRouterEncoder")
    async def test_async_pre_routing_hook_carries_savings_baseline_model(
        self,
        mock_encoder_class,
        mock_semantic_router_class,
        mock_router_instance,
        mock_route_choice,
    ):
        """The hook response must carry the baseline so the spend writer can price
        auto-router savings; without it the dashboard's driver silently stays zero."""
        mock_loaded_router = MagicMock()
        mock_loaded_router.routes = ["route1", "route2"]
        mock_semantic_router_class.from_json.return_value = mock_loaded_router

        mock_routelayer = MagicMock()
        mock_routelayer.return_value = mock_route_choice
        mock_semantic_router_class.return_value = mock_routelayer

        auto_router = AutoRouter(
            model_name="test-auto-router",
            auto_router_config_path="test/path/router.json",
            default_model="gpt-4o-mini",
            embedding_model="text-embedding-model",
            litellm_router_instance=mock_router_instance,
            savings_baseline_model="claude-opus-5",
        )

        result = await auto_router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=[{"role": "user", "content": "hi"}]
        )

        assert result is not None
        assert result.savings_baseline_model == "claude-opus-5"


class TestSavingsBaselineModel:
    """The counterfactual the cost dashboard measures auto-router savings against.

    Constructed without __init__ on purpose: resolving the baseline touches only the
    router's own deployments, never semantic_router, so this runs wherever the rest of
    the beta suite is skipped.
    """

    @staticmethod
    def _auto_router(group_to_model: dict, route_names: list, default_model: str, configured=None) -> AutoRouter:
        parent = MagicMock()
        parent.model_list = [
            {"model_name": group, "litellm_params": dict(params) if isinstance(params, dict) else {"model": params}}
            for group, params in group_to_model.items()
        ]
        parent.model_name_to_deployment_indices = {group: [i] for i, group in enumerate(group_to_model)}

        auto_router = AutoRouter.__new__(AutoRouter)
        auto_router.loaded_routes = []
        for name in route_names:
            route = MagicMock()
            route.name = name
            auto_router.loaded_routes.append(route)
        auto_router.default_model = default_model
        auto_router.litellm_router_instance = parent
        auto_router.configured_savings_baseline_model = configured
        return auto_router

    def test_route_names_resolve_through_the_parent_router_to_pricable_models(self):
        """Routes name the router's own model groups, not models, so a group has to be
        resolved to the model it actually calls before anything can be priced."""
        auto_router = self._auto_router(
            {"cheap-tier": "anthropic/claude-haiku-4-5", "mid-tier": "anthropic/claude-sonnet-5"},
            ["cheap-tier", "mid-tier"],
            "cheap-tier",
        )
        from litellm.router_strategy.savings_baseline import models_for_group

        parent = auto_router.litellm_router_instance
        resolved = sorted(
            model for group in ("cheap-tier", "mid-tier") for model in models_for_group(parent, group)
        )
        assert resolved == ["anthropic/claude-haiku-4-5", "anthropic/claude-sonnet-5"]

    def test_baseline_is_the_priciest_model_this_router_could_have_picked(self):
        """Without the router a deployment picks one model that can carry the hardest
        request, so the counterfactual is this router's priciest candidate. A fixed
        flagship credits savings against a model the operator would never have run: a
        router choosing only between sonnet and haiku saved nobody the price of opus."""
        sonnet_only = self._auto_router(
            {"cheap-tier": "anthropic/claude-haiku-4-5", "mid-tier": "anthropic/claude-sonnet-5"},
            ["cheap-tier", "mid-tier"],
            "cheap-tier",
        )
        assert sonnet_only.savings_baseline_model == "anthropic/claude-sonnet-5"

        with_flagship = self._auto_router(
            {
                "cheap-tier": "anthropic/claude-haiku-4-5",
                "mid-tier": "anthropic/claude-sonnet-5",
                "big-tier": "anthropic/claude-opus-5",
            },
            ["cheap-tier", "mid-tier", "big-tier"],
            "cheap-tier",
        )
        assert with_flagship.savings_baseline_model == "anthropic/claude-opus-5"

    def test_the_default_model_counts_as_a_candidate(self):
        auto_router = self._auto_router(
            {"cheap-tier": "anthropic/claude-haiku-4-5", "fallback": "anthropic/claude-opus-5"},
            ["cheap-tier"],
            "fallback",
        )
        assert auto_router.savings_baseline_model == "anthropic/claude-opus-5"

    def test_an_explicit_baseline_overrides_the_derived_one(self):
        """And is qualified like a derived one: the baseline reaches the spend writer as
        a bare string with no provider beside it, so an operator who writes a name that
        another vendor also owns would otherwise be priced against that vendor."""
        auto_router = self._auto_router(
            {"cheap-tier": "anthropic/claude-haiku-4-5"},
            ["cheap-tier"],
            "cheap-tier",
            configured="claude-opus-5",
        )
        assert auto_router.savings_baseline_model == "anthropic/claude-opus-5"

    def test_an_unresolvable_explicit_baseline_disables_the_driver(self):
        auto_router = self._auto_router(
            {"cheap-tier": "anthropic/claude-haiku-4-5"},
            ["cheap-tier"],
            "cheap-tier",
            configured="no-such-provider-xyz/no-such-model",
        )
        assert auto_router.savings_baseline_model is None

    def test_nothing_priceable_disables_the_driver_rather_than_inventing_a_baseline(self):
        """A missing number beats a fabricated one."""
        auto_router = self._auto_router({}, ["not-a-real-model"], "also-not-real")
        assert auto_router.savings_baseline_model is None

    def test_the_baseline_follows_deployments_added_after_the_first_read(self):
        """The parent router adds and removes deployments while it runs. A baseline
        pinned on first use would keep naming a model the router no longer has, and a
        pricier one added later could never become the baseline."""
        auto_router = self._auto_router(
            {"cheap": {"model": "claude-haiku-4-5", "custom_llm_provider": "anthropic"}},
            ["cheap", "big"],
            "cheap",
        )
        assert auto_router.savings_baseline_model == "anthropic/claude-haiku-4-5"

        parent = auto_router.litellm_router_instance
        parent.model_list.append({"model_name": "big", "litellm_params": {"model": "anthropic/claude-opus-5"}})
        parent.model_name_to_deployment_indices["big"] = [len(parent.model_list) - 1]

        assert auto_router.savings_baseline_model == "anthropic/claude-opus-5"

    def test_the_baseline_drops_a_deployment_that_was_removed(self):
        auto_router = self._auto_router(
            {
                "cheap": {"model": "claude-haiku-4-5", "custom_llm_provider": "anthropic"},
                "big": {"model": "claude-opus-5", "custom_llm_provider": "anthropic"},
            },
            ["cheap", "big"],
            "cheap",
        )
        assert auto_router.savings_baseline_model == "anthropic/claude-opus-5"

        parent = auto_router.litellm_router_instance
        parent.model_name_to_deployment_indices.pop("big")
        auto_router.loaded_routes = [r for r in auto_router.loaded_routes if r.name != "big"]

        assert auto_router.savings_baseline_model == "anthropic/claude-haiku-4-5"

    def test_a_deployment_naming_its_provider_separately_is_still_priced(self):
        """A deployment may name its vendor in `custom_llm_provider` rather than in the
        model prefix. Pricing the bare name then resolves to a different vendor's rates
        or to nothing at all, so the candidate is mispriced or silently dropped and the
        derived baseline is wrong. Vertex prices this model at $0 without its provider
        and azure_ai raises outright, so neither could ever win as the priciest."""
        auto_router = self._auto_router(
            {
                "cheap": {"model": "claude-haiku-4-5", "custom_llm_provider": "anthropic"},
                "vertex-tier": {"model": "claude-sonnet-4@20250514", "custom_llm_provider": "vertex_ai"},
            },
            ["cheap", "vertex-tier"],
            "cheap",
        )
        assert auto_router.savings_baseline_model == "vertex_ai/claude-sonnet-4@20250514"

    def test_candidates_are_qualified_so_the_spend_writer_resolves_the_same_vendor(self):
        """The baseline travels to the spend writer as a bare string, so it has to carry
        its provider or the writer prices it under whichever vendor owns the bare name."""
        from litellm.proxy.spend_tracking.savings import _resolve_model

        auto_router = self._auto_router(
            {"azure-tier": {"model": "deepseek-r1", "custom_llm_provider": "azure_ai"}},
            ["azure-tier"],
            "azure-tier",
        )
        baseline = auto_router.savings_baseline_model
        assert baseline == "azure_ai/deepseek-r1"
        assert _resolve_model(baseline, None) == ("deepseek-r1", "azure_ai")

    def test_a_candidate_with_no_per_token_price_cannot_be_the_baseline(self):
        """A model that costs nothing per token cannot stand in for what the traffic
        would otherwise have cost. Left in, it would report the whole real spend as a
        loss the moment it won the priciest-candidate contest."""
        auto_router = self._auto_router(
            {"images": {"model": "dall-e-2", "custom_llm_provider": "openai"}},
            ["images"],
            "images",
        )
        assert auto_router.savings_baseline_model is None

    def test_a_priced_candidate_still_wins_over_an_unpriced_one(self):
        auto_router = self._auto_router(
            {
                "images": {"model": "dall-e-2", "custom_llm_provider": "openai"},
                "chat": {"model": "claude-haiku-4-5", "custom_llm_provider": "anthropic"},
            },
            ["images", "chat"],
            "chat",
        )
        assert auto_router.savings_baseline_model == "anthropic/claude-haiku-4-5"
