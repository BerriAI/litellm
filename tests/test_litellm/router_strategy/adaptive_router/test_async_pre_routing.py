"""Direct unit tests for AdaptiveRouter.async_pre_routing_hook.

The strategy method (newly extracted from `Router.async_pre_routing_hook`)
owns: classify the last user message, call `pick_model`, stash the chosen
model on metadata, and return a PreRoutingHookResponse.

Routing is stateless per-turn — `pick_model` does not take a session id.
"""

from unittest.mock import AsyncMock

import pytest

from litellm.router_strategy.adaptive_router.adaptive_router import AdaptiveRouter
from litellm.types.router import (
    AdaptiveRouterConfig,
    PreRoutingHookResponse,
    RequestType,
)


def _make_router() -> AdaptiveRouter:
    return AdaptiveRouter(
        router_name="smart-cheap-router",
        config=AdaptiveRouterConfig(available_models=["fast", "smart"]),
        model_to_prefs={},
        model_to_cost={"fast": 0.00000015, "smart": 0.0000050},
    )


@pytest.mark.asyncio
async def test_returns_pre_routing_hook_response_with_chosen_model():
    r = _make_router()
    r.pick_model = AsyncMock(return_value="smart")  # type: ignore[method-assign]

    response = await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={},
        messages=[{"role": "user", "content": "hello"}],
    )

    assert isinstance(response, PreRoutingHookResponse)
    assert response.model == "smart"


@pytest.mark.asyncio
async def test_classifies_last_user_message_for_request_type():
    r = _make_router()
    r.pick_model = AsyncMock(return_value="smart")  # type: ignore[method-assign]

    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={},
        messages=[{"role": "user", "content": "Write a Python function for fizzbuzz"}],
    )

    assert (
        r.pick_model.await_args.kwargs["request_type"]  # type: ignore[union-attr]
        == RequestType.CODE_GENERATION
    )


@pytest.mark.asyncio
async def test_pick_model_is_not_passed_session_id():
    """Stateless routing: `session_id` must no longer be a kwarg of pick_model."""
    r = _make_router()
    r.pick_model = AsyncMock(return_value="fast")  # type: ignore[method-assign]

    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={"metadata": {"litellm_session_id": "sess-A"}},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert "session_id" not in r.pick_model.await_args.kwargs  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_stashes_chosen_model_in_existing_metadata():
    r = _make_router()
    r.pick_model = AsyncMock(return_value="smart")  # type: ignore[method-assign]

    request_kwargs: dict = {"metadata": {"litellm_session_id": "sess-A"}}
    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs=request_kwargs,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert request_kwargs["metadata"]["adaptive_router_chosen_model"] == "smart"
    assert request_kwargs["metadata"]["litellm_session_id"] == "sess-A"


@pytest.mark.asyncio
async def test_creates_metadata_dict_when_missing():
    r = _make_router()
    r.pick_model = AsyncMock(return_value="fast")  # type: ignore[method-assign]

    request_kwargs: dict = {}
    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs=request_kwargs,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert request_kwargs["metadata"]["adaptive_router_chosen_model"] == "fast"


@pytest.mark.asyncio
async def test_handles_empty_messages():
    r = _make_router()
    r.pick_model = AsyncMock(return_value="fast")  # type: ignore[method-assign]

    response = await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={},
        messages=None,
    )

    assert isinstance(response, PreRoutingHookResponse)
    assert response.model == "fast"
    r.pick_model.assert_awaited_once()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_returns_messages_unchanged_in_response():
    r = _make_router()
    r.pick_model = AsyncMock(return_value="smart")  # type: ignore[method-assign]

    messages = [{"role": "user", "content": "hi"}]
    response = await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={},
        messages=messages,
    )

    assert response.messages == messages


# ---- min_quality_tier extraction ----------------------------------------


@pytest.mark.asyncio
async def test_min_quality_tier_from_header_is_forwarded_to_pick_model():
    """`x-litellm-min-quality-tier` header should reach pick_model."""
    r = _make_router()
    r.pick_model = AsyncMock(return_value="smart")  # type: ignore[method-assign]

    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={"headers": {"x-litellm-min-quality-tier": "3"}},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert (
        r.pick_model.await_args.kwargs["min_quality_tier"] == 3  # type: ignore[union-attr]
    )


@pytest.mark.asyncio
async def test_min_quality_tier_from_header_case_insensitive():
    r = _make_router()
    r.pick_model = AsyncMock(return_value="smart")  # type: ignore[method-assign]

    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={"headers": {"X-LiteLLM-Min-Quality-Tier": "2"}},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert (
        r.pick_model.await_args.kwargs["min_quality_tier"] == 2  # type: ignore[union-attr]
    )


@pytest.mark.asyncio
async def test_min_quality_tier_from_metadata_key():
    """Metadata `min_quality_tier` works when the header is absent."""
    r = _make_router()
    r.pick_model = AsyncMock(return_value="smart")  # type: ignore[method-assign]

    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={"metadata": {"min_quality_tier": 3}},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert (
        r.pick_model.await_args.kwargs["min_quality_tier"] == 3  # type: ignore[union-attr]
    )


@pytest.mark.asyncio
async def test_header_takes_precedence_over_metadata():
    r = _make_router()
    r.pick_model = AsyncMock(return_value="smart")  # type: ignore[method-assign]

    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={
            "headers": {"x-litellm-min-quality-tier": "3"},
            "metadata": {"min_quality_tier": 1},
        },
        messages=[{"role": "user", "content": "hi"}],
    )

    assert (
        r.pick_model.await_args.kwargs["min_quality_tier"] == 3  # type: ignore[union-attr]
    )


@pytest.mark.asyncio
async def test_missing_min_quality_tier_passes_none():
    r = _make_router()
    r.pick_model = AsyncMock(return_value="fast")  # type: ignore[method-assign]

    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert (
        r.pick_model.await_args.kwargs["min_quality_tier"] is None  # type: ignore[union-attr]
    )


@pytest.mark.asyncio
async def test_invalid_min_quality_tier_header_treated_as_none():
    """A garbage header value must not crash the request — treat as unset."""
    r = _make_router()
    r.pick_model = AsyncMock(return_value="fast")  # type: ignore[method-assign]

    await r.async_pre_routing_hook(
        model="smart-cheap-router",
        request_kwargs={"headers": {"x-litellm-min-quality-tier": "not-a-number"}},
        messages=[{"role": "user", "content": "hi"}],
    )

    assert (
        r.pick_model.await_args.kwargs["min_quality_tier"] is None  # type: ignore[union-attr]
    )
