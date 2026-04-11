"""
Tests for AutoRouter Responses API (input field) support.

These tests cover the new ``has_messages / has_input`` branching logic and the
empty-string guard in ``AutoRouter.async_pre_routing_hook``.  They do NOT
require ``semantic_router`` to be installed because they only exercise code
paths that return before any SemanticRouter call.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

# semantic_router is an optional dependency (beta feature). Stub it out so
# these tests can run in environments where it is not installed.
_SEMANTIC_ROUTER_MOCK = MagicMock()
_SEMANTIC_ROUTER_STUBS = {
    "semantic_router": _SEMANTIC_ROUTER_MOCK,
    "semantic_router.routers": _SEMANTIC_ROUTER_MOCK.routers,
    "semantic_router.schema": _SEMANTIC_ROUTER_MOCK.schema,
    "semantic_router.routers.base": _SEMANTIC_ROUTER_MOCK.routers.base,
}


def _make_auto_router(default_model: str = "default-model") -> "AutoRouter":  # type: ignore[name-defined]
    """Create an AutoRouter instance without requiring semantic_router."""
    with patch.dict(sys.modules, _SEMANTIC_ROUTER_STUBS):
        from litellm.router_strategy.auto_router.auto_router import AutoRouter

        with patch.object(AutoRouter, "_load_semantic_routing_routes", return_value=[]):
            return AutoRouter(
                model_name="test-auto-router",
                default_model=default_model,
                embedding_model="text-embedding-model",
                litellm_router_instance=MagicMock(),
            )


class TestAutoRouterResponsesAPIEarlyReturns:
    """
    Tests for async_pre_routing_hook paths that return before calling the
    semantic router — safe to run without semantic_router installed.
    """

    @pytest.mark.asyncio
    async def test_no_messages_no_input_returns_none(self):
        """When both messages and input are absent, the hook skips routing."""
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
        auto_router = _make_auto_router(default_model="my-default")
        result = await auto_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=None,
            input=[],
        )
        assert result is not None
        assert result.model == "my-default"
