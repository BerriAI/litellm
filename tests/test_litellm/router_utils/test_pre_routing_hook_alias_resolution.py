"""
Tests that Router.async_pre_routing_hook resolves model_group_alias
before looking up router maps (complexity_routers, auto_routers, etc.).

Without alias resolution, calling completion(model="my-alias") when
"my-alias" maps to a complexity_router group causes an "Unmapped LLM
provider" 400 error because the alias name is not found in the router
maps, so the pre-routing hook returns None and the raw
auto_router/complexity_router deployment is selected as-is.

See: https://github.com/BerriAI/litellm/issues/27473
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm import Router
from litellm.types.router import PreRoutingHookResponse


def _make_router_with_alias(
    alias_name: str,
    target_group: str,
    router_type: str = "complexity",
) -> Router:
    """Build a Router with a model_group_alias and a registered router."""
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {"model": "openai/gpt-4o-mini"},
            }
        ],
        model_group_alias={alias_name: target_group},
    )

    mock_sub_router = MagicMock()
    mock_sub_router.async_pre_routing_hook = AsyncMock(
        return_value=PreRoutingHookResponse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
        )
    )

    if router_type == "complexity":
        router.complexity_routers[target_group] = mock_sub_router
    elif router_type == "auto":
        router.auto_routers[target_group] = mock_sub_router
    elif router_type == "quality":
        router.quality_routers[target_group] = mock_sub_router

    return router


class TestPreRoutingHookAliasResolution:
    """Router.async_pre_routing_hook must resolve model_group_alias."""

    @pytest.mark.asyncio
    async def test_alias_resolves_to_complexity_router(self):
        """An aliased model name should match the complexity_router for
        the resolved group name."""
        router = _make_router_with_alias(
            alias_name="my-alias",
            target_group="auto_router/complexity_router/my-router",
            router_type="complexity",
        )

        result = await router.async_pre_routing_hook(
            model="my-alias",
            request_kwargs={},
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result is not None
        assert result.model == "gpt-4o-mini"
        mock = router.complexity_routers["auto_router/complexity_router/my-router"]
        mock.async_pre_routing_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_alias_resolves_to_auto_router(self):
        """An aliased model name should match the auto_router for the
        resolved group name."""
        router = _make_router_with_alias(
            alias_name="smart",
            target_group="auto_router/my-auto-router",
            router_type="auto",
        )

        result = await router.async_pre_routing_hook(
            model="smart",
            request_kwargs={},
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result is not None
        mock = router.auto_routers["auto_router/my-auto-router"]
        mock.async_pre_routing_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_alias_resolves_to_quality_router(self):
        """An aliased model name should match the quality_router for the
        resolved group name."""
        router = _make_router_with_alias(
            alias_name="best",
            target_group="quality-router-group",
            router_type="quality",
        )

        result = await router.async_pre_routing_hook(
            model="best",
            request_kwargs={},
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result is not None
        mock = router.quality_routers["quality-router-group"]
        mock.async_pre_routing_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_alias_model_still_works(self):
        """A model name that is NOT an alias should still match the router
        map directly (no regression)."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ],
        )
        mock_sub_router = MagicMock()
        mock_sub_router.async_pre_routing_hook = AsyncMock(
            return_value=PreRoutingHookResponse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
            )
        )
        router.complexity_routers["my-router"] = mock_sub_router

        result = await router.async_pre_routing_hook(
            model="my-router",
            request_kwargs={},
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result is not None
        mock_sub_router.async_pre_routing_hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_alias_no_router_returns_none(self):
        """When the model is not an alias and not in any router map,
        async_pre_routing_hook should return None."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ],
        )

        result = await router.async_pre_routing_hook(
            model="some-unknown-model",
            request_kwargs={},
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_alias_dict_format_resolves(self):
        """model_group_alias supports dict format with a 'model' key."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini"},
                }
            ],
            model_group_alias={
                "my-alias": {
                    "model": "auto_router/complexity_router/cr",
                    "hidden": False,
                }
            },
        )
        mock_sub_router = MagicMock()
        mock_sub_router.async_pre_routing_hook = AsyncMock(
            return_value=PreRoutingHookResponse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
            )
        )
        router.complexity_routers["auto_router/complexity_router/cr"] = mock_sub_router

        result = await router.async_pre_routing_hook(
            model="my-alias",
            request_kwargs={},
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result is not None
        mock_sub_router.async_pre_routing_hook.assert_awaited_once()
