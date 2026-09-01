from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from litellm.router import Router
from litellm.router_strategy.capability_router.capability_router import CapabilityRouter
from litellm.router_strategy.capability_router.config import (
    CapabilityClassifierVerdict,
    CapabilityRouterConfig,
)
from litellm.router_strategy.capability_router.policy import select_capability_model


def config() -> dict:
    return {
        "candidates": [
            {"model": "small", "description": "Reliable for short extraction tasks"},
            {"model": "frontier", "description": "Reliable for ambiguous multi-step tasks"},
        ],
        "classifier": {"model": "classifier"},
        "probability_threshold": 0.7,
        "fallback_model": "frontier",
        "cache_ttl_seconds": 60,
    }


def test_config_requires_unique_candidates_and_candidate_fallback() -> None:
    duplicate = config()
    duplicate["candidates"] = [
        {"model": "small", "description": "one"},
        {"model": "small", "description": "two"},
    ]
    with pytest.raises(ValidationError, match="unique"):
        CapabilityRouterConfig.model_validate(duplicate)

    missing_fallback = config()
    missing_fallback["fallback_model"] = "other"
    with pytest.raises(ValidationError, match="one of the candidate"):
        CapabilityRouterConfig.model_validate(missing_fallback)


def test_policy_selects_cheapest_model_above_global_threshold() -> None:
    parsed = CapabilityRouterConfig.model_validate(config())
    verdict = CapabilityClassifierVerdict.model_validate(
        {
            "candidates": [
                {"model": "small", "p_solve": 0.78, "reason": "clear bounded task"},
                {"model": "frontier", "p_solve": 0.95, "reason": "more capable"},
            ]
        }
    )

    decision = select_capability_model(parsed, verdict, {"small": 0.01, "frontier": 0.05})

    assert decision.selected_model == "small"
    assert decision.reason == "cheapest_qualified"
    assert [candidate.qualified for candidate in decision.candidates] == [True, True]


def test_policy_falls_back_if_no_model_qualifies_or_price_is_unknown() -> None:
    parsed = CapabilityRouterConfig.model_validate(config())
    verdict = CapabilityClassifierVerdict.model_validate(
        {
            "candidates": [
                {"model": "small", "p_solve": 0.4, "reason": "too hard"},
                {"model": "frontier", "p_solve": 0.6, "reason": "uncertain"},
            ]
        }
    )
    assert select_capability_model(parsed, verdict, {"small": 0.01, "frontier": 0.05}).reason == (
        "no_qualified_candidate"
    )

    qualified = verdict.model_copy(
        update={
            "candidates": tuple(
                candidate.model_copy(update={"p_solve": 0.9}) for candidate in verdict.candidates
            )
        }
    )
    assert select_capability_model(parsed, qualified, {"small": None, "frontier": 0.05}).reason == (
        "missing_candidate_price"
    )


def test_router_registers_capability_strategy() -> None:
    router = Router(
        model_list=[
            {"model_name": "small", "litellm_params": {"model": "openai/test-small"}},
            {"model_name": "frontier", "litellm_params": {"model": "openai/test-frontier"}},
            {"model_name": "classifier", "litellm_params": {"model": "openai/test-classifier"}},
            {
                "model_name": "cost-router",
                "litellm_params": {
                    "model": "auto_router/capability_router",
                    "capability_router_config": config(),
                },
            },
        ]
    )

    assert len(router.capability_routers["cost-router"]) == 1


@pytest.mark.asyncio
async def test_same_user_turn_reuses_cached_decision() -> None:
    router = Router(model_list=[])
    strategy = CapabilityRouter("cost-router", router, config())
    strategy._new_decision = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            select_capability_model(
                strategy.config,
                CapabilityClassifierVerdict.model_validate(
                    {
                        "candidates": [
                            {"model": "small", "p_solve": 0.9, "reason": "fits"},
                            {"model": "frontier", "p_solve": 0.95, "reason": "fits"},
                        ]
                    }
                ),
                {"small": 0.01, "frontier": 0.05},
            ),
            0.001,
        )
    )
    messages = [{"role": "user", "content": "Extract the invoice number"}]
    kwargs = {"messages": messages, "metadata": {"user_api_key_hash": "key", "session_id": "session"}}

    first = await strategy.async_pre_routing_hook("cost-router", kwargs, messages)
    second = await strategy.async_pre_routing_hook(
        "cost-router",
        {**kwargs, "messages": [*messages, {"role": "assistant", "content": "calling tool"}]},
        [*messages, {"role": "assistant", "content": "calling tool"}],
    )

    assert first.model == second.model == "small"
    assert first.routing_decision is not None and first.routing_decision["cached"] is False
    assert second.routing_decision is not None and second.routing_decision["cached"] is True
    strategy._new_decision.assert_awaited_once()
