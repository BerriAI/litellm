import asyncio
import json
from typing import Any, Dict, Final, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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


semantic_router = pytest.importorskip("semantic_router", reason="auto-router needs the semantic-router extra")

ROUTER_CONFIG: Final = json.dumps(
    {
        "routes": [
            {
                "name": "code-model",
                "description": "coding asks",
                "utterances": ["fix this stack trace", "refactor this function"],
                "score_threshold": 0.5,
            }
        ]
    }
)


class FailingRouteLayer:
    """Route layer whose embedding call fails, as it does when the prompt exceeds the encoder's window."""

    def __call__(self, text: str) -> Any:
        raise ValueError(
            "Internal_litellm_router API call failed. Error: litellm.InternalServerError: "
            "input is too large to process. increase the physical batch size"
        )


class FixedRouteLayer:
    """Route layer that returns whatever the test tells it to, recording the text it was asked about."""

    def __init__(self, route_choice: Any) -> None:
        self.route_choice = route_choice
        self.seen_text: str | None = None

    def __call__(self, text: str) -> Any:
        self.seen_text = text
        return self.route_choice


class StubEmbeddingRouter:
    """Stands in for the LiteLLM Router when the route index has to be built for real."""

    def embedding(self, input: List[str], model: str, **kwargs: Any) -> Any:
        import litellm

        return litellm.EmbeddingResponse(
            data=[{"embedding": [0.1, 0.2], "index": i, "object": "embedding"} for i in range(len(input))]
        )


def _auto_router(routelayer: Any, litellm_router_instance: Any = None, **kwargs: Any) -> AutoRouter:
    auto_router: Final = AutoRouter(
        model_name="my-auto-router",
        auto_router_config=ROUTER_CONFIG,
        default_model="fallback-model",
        embedding_model="text-embedding-3-small",
        litellm_router_instance=litellm_router_instance or MagicMock(),
        **kwargs,
    )
    auto_router.routelayer = routelayer
    return auto_router


class TestAutoRouterAlwaysResolvesARoutableModel:
    """The hook returns the alias's default model instead of failing or leaking the alias downstream."""

    @pytest.mark.asyncio
    async def test_should_fall_back_to_default_model_when_the_embedding_call_fails(self):
        auto_router: Final = _auto_router(FailingRouteLayer())

        result: Final = await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={},
            messages=[{"role": "user", "content": "a" * 100_000}],
        )

        assert result is not None
        assert result.model == "fallback-model"

    @pytest.mark.asyncio
    async def test_should_fall_back_to_default_model_when_no_route_matches(self):
        auto_router: Final = _auto_router(FixedRouteLayer(None))

        result: Final = await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={},
            messages=[{"role": "user", "content": "nothing like any route"}],
        )

        assert result is not None
        # Leaving "my-auto-router" here fails downstream with "Unmapped LLM provider".
        assert result.model == "fallback-model"

    @pytest.mark.asyncio
    async def test_should_fall_back_to_default_model_when_the_route_layer_returns_an_empty_list(self):
        auto_router: Final = _auto_router(FixedRouteLayer([]))

        result: Final = await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={},
            messages=[{"role": "user", "content": "nothing like any route"}],
        )

        assert result is not None
        assert result.model == "fallback-model"

    @pytest.mark.asyncio
    async def test_should_route_to_the_first_choice_when_the_route_layer_returns_a_populated_list(self):
        from semantic_router.schema import RouteChoice

        auto_router: Final = _auto_router(
            FixedRouteLayer([RouteChoice(name="code-model"), RouteChoice(name="chat-model")])
        )

        result: Final = await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={},
            messages=[{"role": "user", "content": "fix this stack trace"}],
        )

        assert result is not None
        assert result.model == "code-model"

    @pytest.mark.asyncio
    async def test_should_still_route_to_the_matched_route_when_one_matches(self):
        from semantic_router.schema import RouteChoice

        layer: Final = FixedRouteLayer(RouteChoice(name="code-model"))
        auto_router: Final = _auto_router(layer)

        result: Final = await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={},
            messages=[{"role": "user", "content": "fix this stack trace"}],
        )

        assert result is not None
        assert result.model == "code-model"
        assert layer.seen_text == "fix this stack trace"


class TestAutoRouterEmbeddingInputCap:
    """The cap configured on the deployment is what the encoder enforces."""

    def test_should_default_the_cap_to_the_shared_constant(self):
        from litellm.constants import DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS

        assert _auto_router(FixedRouteLayer(None)).max_input_chars == DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS

    @pytest.mark.asyncio
    async def test_should_build_its_encoder_with_the_configured_cap(self):
        auto_router: Final = _auto_router(None, litellm_router_instance=StubEmbeddingRouter(), max_input_chars=777)

        await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={},
            messages=[{"role": "user", "content": "fix this stack trace"}],
        )

        assert auto_router.routelayer is not None
        assert auto_router.routelayer.encoder.max_input_chars == 777


class TestAutoRouterRoutesResponsesApiInput:
    """Responses API requests carry the prompt in `input`, not `messages`, and still have to reach the route layer."""

    @pytest.mark.asyncio
    async def test_should_route_a_string_input_when_messages_is_none(self):
        from semantic_router.schema import RouteChoice

        layer: Final = FixedRouteLayer(RouteChoice(name="code-model"))
        auto_router: Final = _auto_router(layer)

        result: Final = await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={
                "input": "fix this stack trace",
                "litellm_metadata": {"user_api_key_request_route": "/v1/responses"},
            },
            messages=None,
        )

        assert result is not None
        assert result.model == "code-model"
        assert result.messages is None
        assert layer.seen_text == "fix this stack trace"

    @pytest.mark.asyncio
    async def test_should_route_a_list_input_with_instructions_when_messages_is_none(self):
        from semantic_router.schema import RouteChoice

        layer: Final = FixedRouteLayer(RouteChoice(name="code-model"))
        auto_router: Final = _auto_router(layer)

        result: Final = await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={
                "instructions": "You are a coding agent.",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "fix this stack trace"}],
                    }
                ],
                "litellm_metadata": {"user_api_key_request_route": "/v1/responses"},
            },
            messages=None,
        )

        assert result is not None
        assert result.model == "code-model"
        assert layer.seen_text is not None
        assert "fix this stack trace" in layer.seen_text

    @pytest.mark.asyncio
    async def test_should_skip_routing_when_neither_messages_nor_input_is_present(self):
        layer: Final = FixedRouteLayer(None)
        auto_router: Final = _auto_router(layer)

        result: Final = await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={"litellm_metadata": {"user_api_key_request_route": "/v1/responses"}},
            messages=None,
        )

        assert result is None
        assert layer.seen_text is None

    @pytest.mark.asyncio
    async def test_should_keep_routing_an_empty_messages_list_to_the_default_model(self):
        layer: Final = FixedRouteLayer(None)
        auto_router: Final = _auto_router(layer)

        result: Final = await auto_router.async_pre_routing_hook(
            model="my-auto-router",
            request_kwargs={"messages": [], "litellm_metadata": {"user_api_key_request_route": "/v1/chat/completions"}},
            messages=[],
        )

        assert result is not None
        assert result.model == "fallback-model"
        assert layer.seen_text == ""
