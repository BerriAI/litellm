import math
import random
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from litellm.router_strategy.complexity_router.auto_setup import (
    AutoRouterSnapshot,
    AutoSetupDeployment,
    AutoSetupDeploymentPricing,
    _quality_completion_cost,  # pyright: ignore[reportPrivateUsage] -- regression test for deployment-aware setup ranking
    analyze_auto_setup_inventory,
    build_auto_setup_config,
    load_auto_router_snapshot,
)
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import ComplexityTier


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
    )
    maximum = build_auto_setup_config(
        snapshot=snapshot,
        available_model_deployments=available,
        quality_level="max",
    )
    nano_only = build_auto_setup_config(
        snapshot=snapshot,
        available_model_deployments=_available_refs("gpt-5.4-nano"),
        quality_level="max",
    )

    assert snapshot.artifact_sha256 == "e191b5f7225e4da7a178eb5fe8780bcea24fb0166be23e2069338c6fe5522cd5"
    assert economy.tiers["SIMPLE"] == ["gpt-5.4-nano"]
    assert maximum.tiers["SIMPLE"] == ["gpt-5.6-sol"]
    assert nano_only.tiers["SIMPLE"] == ["gpt-5.4-nano"]


def test_auto_setup_emits_a_plain_static_complexity_router_config() -> None:
    config = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_deployments=_available_refs("gpt-5.6-sol", "gpt-5.4-nano"),
        quality_level="economy",
    )

    payload = config.model_dump(mode="json", exclude_none=True)
    assert payload["classifier_type"] == "heuristic_v2"
    assert "auto_setup" not in payload
    assert all(len(models) == 1 for models in config.tiers.values())


def test_every_snapshot_alias_builds_all_four_quality_setups_even_without_price() -> None:
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
        eligible, excluded = analyze_auto_setup_inventory(snapshot, {alias: (deployment,)})
        assert eligible == (alias,)
        assert excluded == ()
        for quality_level in ("economy", "balanced", "high", "max"):
            config = build_auto_setup_config(
                snapshot=snapshot,
                available_model_deployments={alias: (deployment,)},
                quality_level=quality_level,
            )
            assert set(config.tiers) == {"SIMPLE", "MEDIUM", "COMPLEX", "REASONING"}
            assert all(models == [alias] for models in config.tiers.values())


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

    selected_tiers: dict[str, dict[str, set[str]]] = {}
    for quality_level in ("economy", "balanced", "high", "max"):
        config = build_auto_setup_config(
            snapshot=snapshot,
            available_model_deployments=deployments,
            quality_level=quality_level,
        )
        repeated = build_auto_setup_config(
            snapshot=snapshot,
            available_model_deployments=dict(reversed(tuple(deployments.items()))),
            quality_level=quality_level,
        )
        assert config.model_dump(mode="json") == repeated.model_dump(mode="json")
        assert all(len(models) == 1 for models in config.tiers.values())
        assert all(set(models).issubset(deployments) for models in config.tiers.values())
        selected_tiers[quality_level] = {tier: set(models) for tier, models in config.tiers.items()}

    assert set(selected_tiers) == {"economy", "balanced", "high", "max"}


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
        for quality_level in ("economy", "balanced", "high", "max"):
            config = build_auto_setup_config(
                snapshot=snapshot,
                available_model_deployments=inventory,
                quality_level=quality_level,
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
    )
    assert all(models == ["safe"] for models in config.tiers.values())


def test_missing_cost_falls_back_to_quality_instead_of_blocking_setup() -> None:
    raw = load_auto_router_snapshot().model_dump(mode="json")
    quality_profiles = raw["models"]["gpt-5.4-nano-xhigh"]["quality_by_complexity"].values()
    for profile in quality_profiles:
        profile["mean_request_cost_usd"] = None
        profile["cost_per_completed_task_usd"] = None
    raw["models"]["gpt-5.4-nano-xhigh"]["quality_by_complexity"]["trivial"]["completion_probability"] = 0

    snapshot = AutoRouterSnapshot.model_validate(raw)
    eligible, excluded = analyze_auto_setup_inventory(snapshot, _available_refs("gpt-5.4-nano"))

    assert eligible == ("gpt-5.4-nano",)
    assert excluded == ()
    generated = build_auto_setup_config(
        snapshot=snapshot,
        available_model_deployments=_available_refs("gpt-5.4-nano"),
        quality_level="max",
    )
    assert all(models == ["gpt-5.4-nano"] for models in generated.tiers.values())


def test_multiple_unpriced_models_choose_the_highest_quality_inside_the_gate() -> None:
    inventory = {
        model_name: (AutoSetupDeployment(model_refs=(model_name,), pricing=None),)
        for model_name in ("gpt-5.6-sol", "gpt-5.4-nano")
    }

    generated = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_deployments=inventory,
        quality_level="max",
    )

    assert generated.tiers["SIMPLE"] == ["gpt-5.6-sol"]


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
    profile = snapshot.models["gpt-5.6-sol-max"].quality_by_complexity["trivial"]
    cheap_cost = _quality_completion_cost(profile, (cheap,))
    expensive_cost = _quality_completion_cost(profile, (expensive,))
    mixed_cost = _quality_completion_cost(profile, (cheap, expensive))

    assert cheap_cost is not None
    assert expensive_cost is not None
    assert mixed_cost is not None
    assert math.isclose(expensive_cost, cheap_cost * 10)
    assert math.isclose(mixed_cost, expensive_cost)

    config = build_auto_setup_config(
        snapshot=snapshot,
        available_model_deployments={"cheap": (cheap,), "expensive": (expensive,), "mixed-price": (cheap, expensive)},
        quality_level="economy",
    )
    assert config.tiers["SIMPLE"] == ["cheap"]


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


def test_auto_setup_result_has_no_runtime_only_configuration() -> None:
    generated = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_deployments=_available_refs("gpt-5.6-sol"),
        quality_level="max",
    )
    payload = generated.model_dump(mode="json", exclude_none=True)

    assert "auto_setup" not in payload
    assert payload["adaptive"] is False
    assert payload["classifier_type"] == "heuristic_v2"


@pytest.mark.asyncio
async def test_generated_single_model_tiers_use_the_existing_router_selection_path() -> None:
    generated = build_auto_setup_config(
        snapshot=load_auto_router_snapshot(),
        available_model_deployments=_available_refs("gpt-5.6-sol", "gpt-5.4-nano"),
        quality_level="economy",
    )
    router_instance = MagicMock()
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
    with patch(  # test-quality-ok: pinning random choice is required to prove manual routers retain their existing selection path
        "litellm.router_strategy.complexity_router.complexity_router.random.choice", return_value="second"
    ) as pick:
        assert await manual._pick_model_for_tier(ComplexityTier.SIMPLE, None, None, {}) == "second"
    pick.assert_called_once_with(["first", "second"])
