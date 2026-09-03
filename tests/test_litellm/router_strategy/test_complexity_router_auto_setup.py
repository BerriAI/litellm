from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from litellm import Router
from litellm.router_strategy.complexity_router.auto_setup import (
    build_auto_setup_config,
    load_auto_router_snapshot,
)
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import (
    AutoSetupCandidate,
    ComplexityRouterConfig,
    ComplexityTier,
)
from litellm.router_strategy.complexity_router.response_latency import (
    record_runtime_response_latency,
    response_latency_sample,
    select_runtime_response_model,
)


def _available_refs(*model_names: str) -> dict[str, tuple[str, ...]]:
    return {model_name: (model_name,) for model_name in model_names}


def test_snapshot_is_verified_and_quality_gate_is_recomputed_for_available_models() -> None:
    snapshot = load_auto_router_snapshot()
    available = _available_refs("gpt-5.6-sol", "gpt-5.4-nano")

    economy = build_auto_setup_config(
        snapshot=snapshot,
        available_model_refs=available,
        quality_level="economy",
        optimize_for="cost",
    )
    maximum = build_auto_setup_config(
        snapshot=snapshot,
        available_model_refs=available,
        quality_level="max",
        optimize_for="cost",
    )
    nano_only = build_auto_setup_config(
        snapshot=snapshot,
        available_model_refs=_available_refs("gpt-5.4-nano"),
        quality_level="max",
        optimize_for="cost",
    )

    assert snapshot.artifact_sha256 == "d0ea7e584dbb9ea37eb15ce20ae342612e0dfaa5801b553e71fc76818d7cef64"
    assert economy.tiers["SIMPLE"] == ["gpt-5.4-nano", "gpt-5.6-sol"]
    assert maximum.tiers["SIMPLE"] == ["gpt-5.6-sol"]
    assert nano_only.tiers["SIMPLE"] == ["gpt-5.4-nano"]


def test_speed_only_changes_auto_generated_selection_policy() -> None:
    config = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_refs=_available_refs("gpt-5.6-sol", "gpt-5.4-nano"),
        quality_level="economy",
        optimize_for="task_completion_speed",
    )

    assert config.auto_setup is not None
    assert config.auto_setup.tier_policies["SIMPLE"].selection_mode == "runtime_response_latency"
    assert config.auto_setup.tier_policies["MEDIUM"].selection_mode == "runtime_response_latency"
    assert config.auto_setup.tier_policies["COMPLEX"].selection_mode == "snapshot_ranked"
    assert config.auto_setup.tier_policies["REASONING"].selection_mode == "snapshot_ranked"
    assert ComplexityRouterConfig(tiers=config.tiers).auto_setup is None


def test_auto_setup_cannot_change_adaptive_or_custom_tier_routers() -> None:
    generated = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_refs=_available_refs("gpt-5.6-sol"),
        quality_level="max",
        optimize_for="cost",
    )
    payload = generated.model_dump(mode="json")

    with pytest.raises(ValidationError, match="auto_setup and adaptive"):
        ComplexityRouterConfig.model_validate({**payload, "adaptive": True})
    with pytest.raises(ValidationError, match="built-in four-tier"):
        ComplexityRouterConfig.model_validate(
            {
                **payload,
                "tier_definitions": [
                    {"name": "fast", "description": "Fast"},
                    {"name": "strong", "description": "Strong"},
                ],
                "tiers": {"fast": ["gpt-5.6-sol"], "strong": ["gpt-5.6-sol"]},
                "fallback_tier": "fast",
                "classifier_type": "llm",
                "classifier_llm_config": {"model": "gpt-5.6-sol"},
            }
        )


class _Cache:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    async def async_get_cache(
        self,
        key: str,
        parent_otel_span: object | None = None,
        local_only: bool = False,
        **kwargs: object,
    ) -> object:
        return self.values.get(key)

    async def async_set_cache(
        self,
        key: str,
        value: object,
        local_only: bool = False,
        **kwargs: object,
    ) -> None:
        self.values[key] = value


def _candidate(model_name: str, cost: float) -> AutoSetupCandidate:
    return AutoSetupCandidate(
        model_name=model_name,
        benchmark_model_id=f"benchmark-{model_name}",
        quality_lower_bound=0.95,
        cost_per_completed_task_usd=cost,
    )


@pytest.mark.asyncio
async def test_easy_task_speed_learns_equal_output_response_time() -> None:
    cache = _Cache()
    candidates = (_candidate("cheap-slow", 0.01), _candidate("fast", 0.04))

    assert (
        await select_runtime_response_model(
            router_cache=cache,
            router_model_name="auto",
            tier="SIMPLE",
            candidates=candidates,
            available_models=("cheap-slow", "fast"),
            cold_start_model="cheap-slow",
            objective="task_completion_speed",
        )
        == "cheap-slow"
    )

    for model, end in (("cheap-slow", 0.30), ("cheap-slow", 0.30), ("fast", 0.15), ("fast", 0.15)):
        await record_runtime_response_latency(
            router_cache=cache,
            router_model_name="auto",
            tier="SIMPLE",
            routed_model=model,
            kwargs={"completion_start_time": 0.10},
            response_obj={
                "usage": {
                    "completion_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 10},
                }
            },
            start_time=0.0,
            end_time=end,
        )

    measured = response_latency_sample(
        {"completion_start_time": 0.10},
        {"usage": {"completion_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 10}}},
        0.0,
        0.30,
    )
    assert measured is not None and measured[1] == 10
    assert (
        await select_runtime_response_model(
            router_cache=cache,
            router_model_name="auto",
            tier="SIMPLE",
            candidates=candidates,
            available_models=("cheap-slow", "fast"),
            cold_start_model="cheap-slow",
            objective="task_completion_speed",
        )
        == "fast"
    )


@pytest.mark.asyncio
async def test_snapshot_policy_is_deterministic_but_manual_pool_keeps_random_selection() -> None:
    generated = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_refs=_available_refs("gpt-5.6-sol", "gpt-5.4-nano"),
        quality_level="economy",
        optimize_for="cost",
    )
    router_instance = MagicMock()
    router_instance.cache = AsyncMock()
    automatic = ComplexityRouter(
        model_name="auto",
        litellm_router_instance=router_instance,
        complexity_router_config=generated.model_dump(mode="json"),
    )
    manual = ComplexityRouter(
        model_name="manual",
        litellm_router_instance=router_instance,
        complexity_router_config={"tiers": {"SIMPLE": ["first", "second"]}},
    )

    assert await automatic._pick_model_for_tier(ComplexityTier.SIMPLE, None, None, {}) == "gpt-5.4-nano"
    assert (
        await automatic._pick_model_for_tier(
            ComplexityTier.SIMPLE,
            None,
            None,
            {},
            allowed_models=("gpt-5.6-sol",),
        )
        == "gpt-5.6-sol"
    )
    with patch(  # test-quality-ok: pinning random choice is required to prove manual routers retain their existing selection path
        "litellm.router_strategy.complexity_router.complexity_router.random.choice", return_value="second"
    ) as pick:
        assert await manual._pick_model_for_tier(ComplexityTier.SIMPLE, None, None, {}) == "second"
    pick.assert_called_once_with(("first", "second"))


@pytest.mark.asyncio
async def test_live_latency_telemetry_is_recorded_only_for_an_auto_runtime_policy() -> None:  # test-quality-ok: this callback's observable contract is whether telemetry is emitted
    generated = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_refs=_available_refs("gpt-5.6-sol", "gpt-5.4-nano"),
        quality_level="economy",
        optimize_for="task_completion_speed",
    )
    router = Router(
        model_list=[
            {
                "model_name": "smart",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": generated.model_dump(mode="json"),
                    "complexity_router_default_model": "gpt-5.6-sol",
                },
            },
            {"model_name": "gpt-5.6-sol", "litellm_params": {"model": "gpt-5.6-sol"}},
            {"model_name": "gpt-5.4-nano", "litellm_params": {"model": "gpt-5.4-nano"}},
        ]
    )
    metadata = {
        "routing_decision": {
            "router_model_name": "smart",
            "tier": "SIMPLE",
            "routed_model": "gpt-5.4-nano",
            "auto_setup_selection_mode": "runtime_response_latency",
        }
    }

    with patch(  # test-quality-ok: patching the telemetry boundary avoids mutating the shared runtime latency cache
        "litellm.router_strategy.complexity_router.response_latency.record_runtime_response_latency",
        new_callable=AsyncMock,
    ) as recorder:
        await router._record_auto_setup_response_latency(
            kwargs={"litellm_params": {"metadata": metadata}},
            completion_response={"usage": {"completion_tokens": 1}},
            start_time=0.0,
            end_time=0.1,
        )
        metadata["routing_decision"]["auto_setup_selection_mode"] = "snapshot_ranked"
        await router._record_auto_setup_response_latency(
            kwargs={"litellm_params": {"metadata": metadata}},
            completion_response={"usage": {"completion_tokens": 1}},
            start_time=0.0,
            end_time=0.1,
        )

    recorder.assert_awaited_once()
