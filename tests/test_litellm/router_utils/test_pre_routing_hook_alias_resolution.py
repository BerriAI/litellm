"""Regression tests for issue #27473.

When a pre-routing router group (complexity / auto / adaptive / quality) is
exposed to clients through a ``model_group_alias``, ``Router.async_pre_routing_hook``
must resolve the alias to the real group name before looking the strategy up in
``self.complexity_routers`` / ``self.auto_routers`` / ``self.adaptive_routers`` /
``self.quality_routers``.

Before the fix the hook looked the raw alias up in those maps, found nothing,
returned ``None``, and the raw ``auto_router/complexity_router`` deployment was
dispatched as-is, producing
``BadRequestError: Unmapped LLM provider ... model=complexity_router, custom_llm_provider=auto_router``.
"""

from unittest.mock import AsyncMock

import pytest

from litellm import Router
from litellm.types.router import PreRoutingHookResponse


def _router_with_complexity_alias() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "internal-ai",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {
                            "SIMPLE": "internal-haiku",
                            "REASONING": "internal-sonnet",
                        },
                    },
                },
            },
            {
                "model_name": "internal-haiku",
                "litellm_params": {"model": "openai/gpt-4o-mini"},
            },
            {
                "model_name": "internal-sonnet",
                "litellm_params": {"model": "openai/gpt-4o"},
            },
        ],
        model_group_alias={"claude-sonnet-4-6": "internal-ai"},
    )


@pytest.mark.asyncio
async def test_alias_resolves_to_complexity_router():
    r = _router_with_complexity_alias()
    assert "internal-ai" in r.complexity_routers
    strategy = r.complexity_routers["internal-ai"][0].strategy
    messages = [{"role": "user", "content": "hi"}]
    strategy.async_pre_routing_hook = AsyncMock(  # type: ignore[method-assign]
        return_value=PreRoutingHookResponse(model="internal-haiku", messages=messages)
    )

    response = await r.async_pre_routing_hook(
        model="claude-sonnet-4-6",
        request_kwargs={},
        messages=messages,
    )

    assert response is not None
    assert response.model == "internal-haiku"
    strategy.async_pre_routing_hook.assert_awaited_once()  # type: ignore[union-attr]
    assert strategy.async_pre_routing_hook.await_args.kwargs["model"] == "internal-ai"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_direct_group_name_still_dispatches():
    """No regression: sending the real group name must still dispatch."""
    r = _router_with_complexity_alias()
    strategy = r.complexity_routers["internal-ai"][0].strategy
    messages = [{"role": "user", "content": "hi"}]
    strategy.async_pre_routing_hook = AsyncMock(  # type: ignore[method-assign]
        return_value=PreRoutingHookResponse(model="internal-sonnet", messages=messages)
    )

    response = await r.async_pre_routing_hook(
        model="internal-ai",
        request_kwargs={},
        messages=messages,
    )

    assert response is not None
    assert response.model == "internal-sonnet"
    assert strategy.async_pre_routing_hook.await_args.kwargs["model"] == "internal-ai"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_alias_to_plain_model_group_returns_none():
    """An alias that points at a non-pre-routing group must not dispatch a strategy."""
    r = _router_with_complexity_alias()
    r.model_group_alias["plain-alias"] = "internal-haiku"

    response = await r.async_pre_routing_hook(
        model="plain-alias",
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response is None


@pytest.mark.asyncio
async def test_alias_applies_router_group_litellm_params():
    """The resolved group's litellm_params (e.g. complexity_router_config) are
    applied off the real group name, not the alias, so the lookup must succeed."""
    r = _router_with_complexity_alias()
    strategy = r.complexity_routers["internal-ai"][0].strategy
    messages = [{"role": "user", "content": "hi"}]
    strategy.async_pre_routing_hook = AsyncMock(  # type: ignore[method-assign]
        return_value=PreRoutingHookResponse(model="internal-haiku", messages=messages)
    )

    request_kwargs: dict = {}
    await r.async_pre_routing_hook(
        model="claude-sonnet-4-6",
        request_kwargs=request_kwargs,
        messages=messages,
    )

    assert request_kwargs.get("complexity_router_config") is not None
