"""
Tests for AutoRouter Responses API (input field) support.

These tests cover the new ``has_messages / has_input`` branching logic and the
empty-string guard in ``AutoRouter.async_pre_routing_hook``.  They do NOT
require ``semantic_router`` to be installed because they stub it out via
``sys.modules`` patching.
"""
import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

# semantic_router is an optional dependency (beta feature). Stub it out so
# these tests can run in environments where it is not installed.
class _FakeRouteChoice:
    """Minimal stand-in so ``isinstance(x, RouteChoice)`` is a valid check."""
    pass


_SEMANTIC_ROUTER_MOCK = MagicMock()
# RouteChoice must be a real class for isinstance() to work
_SEMANTIC_ROUTER_MOCK.schema.RouteChoice = _FakeRouteChoice

_LITELLM_ENCODER_MOCK = MagicMock()
_SEMANTIC_ROUTER_STUBS = {
    # semantic_router and every sub-module that async_pre_routing_hook imports
    "semantic_router": _SEMANTIC_ROUTER_MOCK,
    "semantic_router.routers": _SEMANTIC_ROUTER_MOCK.routers,
    "semantic_router.schema": _SEMANTIC_ROUTER_MOCK.schema,
    "semantic_router.routers.base": _SEMANTIC_ROUTER_MOCK.routers.base,
    "semantic_router.encoders": _SEMANTIC_ROUTER_MOCK.encoders,
    "semantic_router.encoders.base": _SEMANTIC_ROUTER_MOCK.encoders.base,
    # litellm_encoder imports semantic_router internally; stub the whole module
    # so tests that pre-set routelayer never instantiate LiteLLMRouterEncoder
    "litellm.router_strategy.auto_router.litellm_encoder": _LITELLM_ENCODER_MOCK,
}


@contextmanager
def _semantic_router_patched():
    """Context manager that stubs semantic_router for both init and method calls."""
    with patch.dict(sys.modules, _SEMANTIC_ROUTER_STUBS):
        yield


def _make_auto_router(default_model: str = "default-model"):
    """
    Create an AutoRouter instance without requiring semantic_router.

    Must be called inside a ``_semantic_router_patched()`` context when the
    returned instance will be used in a method call that reaches the
    ``from semantic_router...`` imports (i.e. non-early-return paths).
    """
    from litellm.router_strategy.auto_router.auto_router import AutoRouter

    with patch.object(AutoRouter, "_load_semantic_routing_routes", return_value=[]):
        return AutoRouter(
            model_name="test-auto-router",
            default_model=default_model,
            embedding_model="text-embedding-model",
            litellm_router_instance=MagicMock(),
        )


def _mock_routelayer(routed_model: str = "routed-model") -> MagicMock:
    """Return a mock routelayer that yields ``routed_model`` via a list RouteChoice."""
    mock_choice = MagicMock()
    mock_choice.name = routed_model
    return MagicMock(return_value=[mock_choice])


class TestAutoRouterResponsesAPIEarlyReturns:
    """
    Tests for async_pre_routing_hook paths that return before calling the
    semantic router (before ``from semantic_router...`` imports fire).
    """

    @pytest.mark.asyncio
    async def test_no_messages_no_input_returns_none(self):
        """When both messages and input are absent, the hook skips routing."""
        with _semantic_router_patched():
            auto_router = _make_auto_router()
            result = await auto_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=None,
                input=None,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_messages_no_input_returns_none(self):
        """An empty messages list with no input also skips routing."""
        with _semantic_router_patched():
            auto_router = _make_auto_router()
            result = await auto_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=[],
                input=None,
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_string_input_returns_default_model(self):
        """An empty string input falls back to the default model (not routelayer)."""
        with _semantic_router_patched():
            auto_router = _make_auto_router(default_model="my-default")
            result = await auto_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=None,
                input="",
            )
        assert result is not None
        assert result.model == "my-default"

    @pytest.mark.asyncio
    async def test_whitespace_only_input_returns_default_model(self):
        """A whitespace-only input is treated the same as empty — use default."""
        with _semantic_router_patched():
            auto_router = _make_auto_router(default_model="my-default")
            result = await auto_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=None,
                input="   ",
            )
        assert result is not None
        assert result.model == "my-default"

    @pytest.mark.asyncio
    async def test_empty_list_input_returns_default_model(self):
        """An empty list input falls back to the default model."""
        with _semantic_router_patched():
            auto_router = _make_auto_router(default_model="my-default")
            result = await auto_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=None,
                input=[],
            )
        assert result is not None
        assert result.model == "my-default"


class TestAutoRouterMessagesAndInputRouting:
    """
    Tests for the has_messages and else (input) branches that reach the
    semantic router.  routelayer is pre-set on the instance to skip the
    lazy-init block and avoid LiteLLMRouterEncoder instantiation.
    """

    @pytest.mark.asyncio
    async def test_messages_branch_passes_content_to_routelayer(self):
        """has_messages=True: last message content is passed to routelayer."""
        with _semantic_router_patched():
            auto_router = _make_auto_router(default_model="my-default")
            auto_router.routelayer = _mock_routelayer("chat-model")

            result = await auto_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=[{"role": "user", "content": "Hello world"}],
                input=None,
            )

        assert result is not None
        assert result.model == "chat-model"
        auto_router.routelayer.assert_called_once_with(text="Hello world")

    @pytest.mark.asyncio
    async def test_input_branch_passes_extracted_text_to_routelayer(self):
        """has_input=True, has_messages=False: extracted input text sent to routelayer."""
        with _semantic_router_patched():
            auto_router = _make_auto_router(default_model="my-default")
            auto_router.routelayer = _mock_routelayer("responses-model")

            result = await auto_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=None,
                input="Hello from Responses API",
            )

        assert result is not None
        assert result.model == "responses-model"
        auto_router.routelayer.assert_called_once_with(text="Hello from Responses API")

    @pytest.mark.asyncio
    async def test_input_list_branch_passes_extracted_text_to_routelayer(self):
        """Responses API list input: extracted text from input_text parts sent to routelayer."""
        with _semantic_router_patched():
            auto_router = _make_auto_router(default_model="my-default")
            auto_router.routelayer = _mock_routelayer("responses-model")

            result = await auto_router.async_pre_routing_hook(
                model="test-model",
                request_kwargs={},
                messages=None,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello from list"}],
                    }
                ],
            )

        assert result is not None
        assert result.model == "responses-model"
        auto_router.routelayer.assert_called_once_with(text="Hello from list")
