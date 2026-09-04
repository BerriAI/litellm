import math
import random
from itertools import product
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from litellm import Router
from litellm.router_strategy.complexity_router.auto_setup import (
    AutoRouterSnapshot,
    AutoSetupDeployment,
    AutoSetupDeploymentPricing,
    _speed_completion_cost,  # pyright: ignore[reportPrivateUsage] -- regression test for multi-call trajectory pricing
    analyze_auto_setup_inventory,
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


def _available_refs(*model_names: str) -> dict[str, tuple[AutoSetupDeployment, ...]]:
    return {
        model_name: (
            AutoSetupDeployment(
                model_refs=(model_name,),
                pricing=(
                    AutoSetupDeploymentPricing(input_cost_per_token=0.0000002, output_cost_per_token=0.00000125)
                    if model_name == "gpt-5.4-nano"
                    else AutoSetupDeploymentPricing(input_cost_per_token=0.000005, output_cost_per_token=0.00003)
                ),
            ),
        )
        for model_name in model_names
    }


def _registry_deployment(model_ref: str) -> AutoSetupDeployment:
    from litellm.proxy.management_endpoints.auto_router_endpoints import (
        _deployment_pricing,  # pyright: ignore[reportPrivateUsage] -- exercises the endpoint's production registry resolver
    )

    raw = {"model_name": model_ref, "litellm_params": {"model": model_ref}, "model_info": {}}
    return AutoSetupDeployment(model_refs=(model_ref,), pricing=_deployment_pricing(raw))


def test_snapshot_is_verified_and_quality_gate_is_recomputed_for_available_models() -> None:
    snapshot = load_auto_router_snapshot()
    available = _available_refs("gpt-5.6-sol", "gpt-5.4-nano")

    economy = build_auto_setup_config(
        snapshot=snapshot,
        available_model_deployments=available,
        quality_level="economy",
        optimize_for="cost",
    )
    maximum = build_auto_setup_config(
        snapshot=snapshot,
        available_model_deployments=available,
        quality_level="max",
        optimize_for="cost",
    )
    nano_only = build_auto_setup_config(
        snapshot=snapshot,
        available_model_deployments=_available_refs("gpt-5.4-nano"),
        quality_level="max",
        optimize_for="cost",
    )

    assert snapshot.artifact_sha256 == "bab7488212d40d96c5e43f80893c5ca2961d0e25160cb1aebbef36a2422b8a2b"
    assert economy.tiers["SIMPLE"] == ["gpt-5.4-nano", "gpt-5.6-sol"]
    assert maximum.tiers["SIMPLE"] == ["gpt-5.6-sol"]
    assert nano_only.tiers["SIMPLE"] == ["gpt-5.4-nano"]


def test_speed_only_changes_auto_generated_selection_policy() -> None:
    config = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_deployments=_available_refs("gpt-5.6-sol", "gpt-5.4-nano"),
        quality_level="economy",
        optimize_for="task_completion_speed",
    )

    assert config.auto_setup is not None
    assert config.auto_setup.tier_policies["SIMPLE"].selection_mode == "runtime_response_latency"
    assert config.auto_setup.tier_policies["MEDIUM"].selection_mode == "runtime_response_latency"
    assert config.auto_setup.tier_policies["COMPLEX"].selection_mode == "snapshot_ranked"
    assert config.auto_setup.tier_policies["REASONING"].selection_mode == "snapshot_ranked"
    assert ComplexityRouterConfig(tiers=config.tiers).auto_setup is None


def test_every_priced_snapshot_alias_builds_all_twelve_auto_setups() -> None:
    snapshot = load_auto_router_snapshot()
    aliases = tuple(
        alias
        for model in snapshot.models.values()
        if model.identity.is_routable
        for alias in model.identity.litellm_model_keys
    )
    assert len(aliases) == 283
    priced = {alias: _registry_deployment(alias) for alias in aliases}
    unsupported = tuple(alias for alias, deployment in priced.items() if deployment.pricing is None)
    assert unsupported == (
        "perplexity/anthropic/claude-opus-4-6",
        "perplexity/anthropic/claude-opus-4-7",
        "chatgpt/gpt-5.2-codex",
        "chatgpt/gpt-5.4",
    )

    for alias, deployment in priced.items():
        if deployment.pricing is None:
            eligible, excluded = analyze_auto_setup_inventory(snapshot, {alias: (deployment,)})
            assert not eligible
            assert [(item.model_group, item.reason) for item in excluded] == [(alias, "pricing_unavailable")]
            continue
        for quality_level, objective in product(
            ("economy", "balanced", "high", "max"),
            ("cost", "task_completion_speed", "balanced"),
        ):
            config = build_auto_setup_config(
                snapshot=snapshot,
                available_model_deployments={alias: (deployment,)},
                quality_level=quality_level,
                optimize_for=objective,
            )
            assert set(config.tiers) == {"SIMPLE", "MEDIUM", "COMPLEX", "REASONING"}
            assert all(models == [alias] for models in config.tiers.values())
            assert config.auto_setup is not None
            assert all(policy.candidates[0].model_name == alias for policy in config.auto_setup.tier_policies.values())


def test_full_priced_inventory_builds_deterministically_for_every_choice() -> None:
    snapshot = load_auto_router_snapshot()
    deployments = {
        f"group-{index:03d}": (_registry_deployment(alias),)
        for index, alias in enumerate(
            dict.fromkeys(
                alias
                for model in snapshot.models.values()
                if model.identity.is_routable
                for alias in model.identity.litellm_model_keys
            )
        )
        if _registry_deployment(alias).pricing is not None
    }
    assert len(deployments) == 279

    cost_tiers: dict[str, dict[str, set[str]]] = {}
    for quality_level, objective in product(
        ("economy", "balanced", "high", "max"),
        ("cost", "task_completion_speed", "balanced"),
    ):
        config = build_auto_setup_config(
            snapshot=snapshot,
            available_model_deployments=deployments,
            quality_level=quality_level,
            optimize_for=objective,
        )
        repeated = build_auto_setup_config(
            snapshot=snapshot,
            available_model_deployments=dict(reversed(tuple(deployments.items()))),
            quality_level=quality_level,
            optimize_for=objective,
        )
        assert config.model_dump(mode="json") == repeated.model_dump(mode="json")
        assert all(models and len(models) == len(set(models)) for models in config.tiers.values())
        assert all(set(models).issubset(deployments) for models in config.tiers.values())
        if objective == "cost":
            cost_tiers[quality_level] = {tier: set(models) for tier, models in config.tiers.items()}

    for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING"):
        assert cost_tiers["max"][tier] <= cost_tiers["high"][tier]
        assert cost_tiers["high"][tier] <= cost_tiers["balanced"][tier]
        assert cost_tiers["balanced"][tier] <= cost_tiers["economy"][tier]


def test_seeded_mixed_inventories_never_select_unknown_or_unsafe_groups() -> None:
    snapshot = load_auto_router_snapshot()
    priced_aliases = [
        alias
        for model in snapshot.models.values()
        if model.identity.is_routable
        for alias in model.identity.litellm_model_keys
        if _registry_deployment(alias).pricing is not None
    ]
    rng = random.Random(20260903)
    for inventory_index in range(100):
        chosen = rng.sample(priced_aliases, rng.randint(1, min(30, len(priced_aliases))))
        inventory = {
            f"known-{inventory_index}-{index}": (_registry_deployment(alias),) for index, alias in enumerate(chosen)
        }
        inventory.update(
            {
                f"unknown-{inventory_index}-{index}": (
                    AutoSetupDeployment(
                        model_refs=(f"unknown/provider-{inventory_index}-{index}",),
                        pricing=AutoSetupDeploymentPricing(
                            input_cost_per_token=0.000001,
                            output_cost_per_token=0.000002,
                        ),
                    ),
                )
                for index in range(rng.randint(0, 10))
            }
        )
        for quality_level, objective in product(
            ("economy", "balanced", "high", "max"),
            ("cost", "task_completion_speed", "balanced"),
        ):
            config = build_auto_setup_config(
                snapshot=snapshot,
                available_model_deployments=inventory,
                quality_level=quality_level,
                optimize_for=objective,
            )
            selected = {model_name for models in config.tiers.values() for model_name in models}
            assert selected
            assert all(model_name.startswith("known-") for model_name in selected)


def test_mixed_or_partially_unmatched_model_groups_fail_closed() -> None:
    snapshot = load_auto_router_snapshot()
    pricing = AutoSetupDeploymentPricing(input_cost_per_token=0.000001, output_cost_per_token=0.000002)
    inventory = {
        "mixed": (
            AutoSetupDeployment(model_refs=("gpt-5.6-sol",), pricing=pricing),
            AutoSetupDeployment(model_refs=("gpt-5.4-nano",), pricing=pricing),
        ),
        "partial": (
            AutoSetupDeployment(model_refs=("gpt-5.6-sol",), pricing=pricing),
            AutoSetupDeployment(model_refs=("unknown/model",), pricing=pricing),
        ),
        "ambiguous-single-deployment": (
            AutoSetupDeployment(model_refs=("gpt-5.6-sol", "gpt-5.4-nano"), pricing=pricing),
        ),
        "safe": (AutoSetupDeployment(model_refs=("gpt-5.6-sol",), pricing=pricing),),
    }

    eligible, excluded = analyze_auto_setup_inventory(snapshot, inventory)

    assert eligible == ("safe",)
    assert {(item.model_group, item.reason) for item in excluded} == {
        ("mixed", "mixed_model_group"),
        ("partial", "no_benchmark_match"),
        ("ambiguous-single-deployment", "mixed_model_group"),
    }
    config = build_auto_setup_config(
        snapshot=snapshot,
        available_model_deployments=inventory,
        quality_level="economy",
        optimize_for="cost",
    )
    assert all(models == ["safe"] for models in config.tiers.values())


def test_schema_valid_missing_cost_and_zero_completion_evidence_fail_closed() -> None:
    raw = load_auto_router_snapshot().model_dump(mode="json")
    quality_profiles = raw["models"]["gpt-5.4-nano-xhigh"]["quality_by_complexity"].values()
    for profile in quality_profiles:
        profile["mean_request_cost_usd"] = None
        profile["cost_per_completed_task_usd"] = None
    raw["models"]["gpt-5.4-nano-xhigh"]["quality_by_complexity"]["trivial"]["completion_probability"] = 0

    snapshot = AutoRouterSnapshot.model_validate(raw)
    eligible, excluded = analyze_auto_setup_inventory(snapshot, _available_refs("gpt-5.4-nano"))

    assert eligible == ()
    assert [(item.model_group, item.reason) for item in excluded] == [("gpt-5.4-nano", "pricing_unavailable")]


def test_provider_repricing_and_multi_deployment_groups_are_conservative() -> None:
    snapshot = load_auto_router_snapshot()
    cheap = AutoSetupDeployment(
        model_refs=("gpt-5.6-sol",),
        pricing=AutoSetupDeploymentPricing(input_cost_per_token=0.000001, output_cost_per_token=0.000002),
    )
    expensive = AutoSetupDeployment(
        model_refs=("gpt-5.6-sol",),
        pricing=AutoSetupDeploymentPricing(input_cost_per_token=0.00001, output_cost_per_token=0.00002),
    )
    config = build_auto_setup_config(
        snapshot=snapshot,
        available_model_deployments={"cheap": (cheap,), "expensive": (expensive,), "mixed-price": (cheap, expensive)},
        quality_level="economy",
        optimize_for="cost",
    )
    assert config.auto_setup is not None
    candidates = config.auto_setup.tier_policies["SIMPLE"].candidates
    by_name = {candidate.model_name: candidate.cost_per_completed_task_usd for candidate in candidates}
    assert tuple(candidate.model_name for candidate in candidates) == ("cheap", "expensive", "mixed-price")
    assert math.isclose(by_name["expensive"], by_name["cheap"] * 10)
    assert math.isclose(by_name["mixed-price"], by_name["expensive"])


def test_multi_call_trajectory_cost_does_not_apply_single_request_context_thresholds() -> None:
    snapshot = load_auto_router_snapshot()
    speed = snapshot.models["gpt-5.6-sol-max"].task_completion_speed_by_complexity["complex"][0]
    pricing = AutoSetupDeploymentPricing(
        input_cost_per_token=0.000001,
        output_cost_per_token=0.000002,
        cache_read_input_token_cost=0.0000001,
        input_cost_per_token_above_200k_tokens=0.001,
        output_cost_per_token_above_200k_tokens=0.002,
        cache_read_input_token_cost_above_200k_tokens=0.0001,
    )
    deployment = AutoSetupDeployment(model_refs=("gpt-5.6-sol",), pricing=pricing)

    actual = _speed_completion_cost(speed, (deployment,))
    expected_attempt_cost = (
        (speed.mean_uncached_input_tokens or 0) * pricing.input_cost_per_token
        + (speed.mean_cache_read_input_tokens or 0) * (pricing.cache_read_input_token_cost or 0)
        + (speed.mean_output_tokens or 0) * pricing.output_cost_per_token
    )

    assert actual is not None
    assert math.isclose(actual, expected_attempt_cost / speed.success_probability)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("missing_complexity", "lacks complete quality evidence"),
        ("identity_mismatch", "model identity mismatch"),
        ("routability_mismatch", "routability mismatch"),
        ("quality_mismatch", "quality identity mismatch"),
    ),
)
def test_snapshot_semantic_validation_rejects_inconsistent_artifacts(mutation: str, error: str) -> None:
    raw = load_auto_router_snapshot().model_dump(mode="json")
    model_id = "claude-fable-5-1-max-effort"
    model = raw["models"][model_id]
    if mutation == "missing_complexity":
        del model["quality_by_complexity"]["trivial"]
    elif mutation == "identity_mismatch":
        model["benchmark_model_id"] = "wrong"
    elif mutation == "routability_mismatch":
        model["identity"]["is_routable"] = not model["identity"]["is_routable"]
    else:
        model["quality_by_complexity"]["trivial"]["benchmark_model_id"] = "wrong"

    with pytest.raises(ValidationError, match=error):
        AutoRouterSnapshot.model_validate(raw)


def test_auto_setup_cannot_change_adaptive_or_custom_tier_routers() -> None:
    generated = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_deployments=_available_refs("gpt-5.6-sol"),
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
async def test_easy_task_balanced_uses_both_live_response_time_and_expected_cost() -> None:
    cache = _Cache()
    candidates = (
        _candidate("fast-expensive", 1.0),
        _candidate("middle", 0.02),
        _candidate("cheap-slow", 0.01),
    )
    for model, end in (("fast-expensive", 0.10), ("middle", 0.12), ("cheap-slow", 0.30)):
        for _ in range(2):
            await record_runtime_response_latency(
                router_cache=cache,
                router_model_name="auto",
                tier="SIMPLE",
                routed_model=model,
                kwargs={},
                response_obj={"usage": {"completion_tokens": 10}},
                start_time=0.0,
                end_time=end,
            )

    common = {
        "router_cache": cache,
        "router_model_name": "auto",
        "tier": "SIMPLE",
        "candidates": candidates,
        "available_models": tuple(candidate.model_name for candidate in candidates),
        "cold_start_model": "cheap-slow",
    }
    assert await select_runtime_response_model(**common, objective="task_completion_speed") == "fast-expensive"
    assert await select_runtime_response_model(**common, objective="balanced") == "middle"


@pytest.mark.asyncio
async def test_snapshot_policy_is_deterministic_but_manual_pool_keeps_random_selection() -> None:
    generated = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_deployments=_available_refs("gpt-5.6-sol", "gpt-5.4-nano"),
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
async def test_live_latency_telemetry_is_recorded_only_for_an_auto_runtime_policy() -> None:
    # test-quality-ok: this callback's observable contract is whether telemetry is emitted
    generated = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_deployments=_available_refs("gpt-5.6-sol", "gpt-5.4-nano"),
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
