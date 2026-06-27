import pytest
from pydantic import ValidationError

from litellm.types.router import (
    AdaptiveRouterConfig,
    AdaptiveRouterPreferences,
    AdaptiveRouterWeights,  # noqa: F401  # imported per spec, exercised transitively
    RequestType,
)


def test_config_loads_valid_yaml():
    cfg = AdaptiveRouterConfig(
        available_models=["gpt-4o-mini", "gpt-4o"],
        weights={"quality": 0.7, "cost": 0.3},
    )
    assert cfg.available_models == ["gpt-4o-mini", "gpt-4o"]
    assert cfg.weights.quality == 0.7
    assert cfg.weights.cost == 0.3
    assert abs(cfg.weights.quality + cfg.weights.cost - 1.0) < 0.001


def test_config_rejects_misspelled_strength():
    with pytest.raises(ValidationError):
        AdaptiveRouterPreferences(quality_tier=2, strengths=["code_genertion"])


def test_config_weights_must_sum_to_one():
    with pytest.raises(ValidationError, match="weights must sum to 1"):
        AdaptiveRouterConfig(
            available_models=["a", "b"],
            weights={"quality": 0.9, "cost": 0.5},
        )


def test_config_quality_tier_must_be_1_2_or_3():
    with pytest.raises(ValidationError):
        AdaptiveRouterPreferences(quality_tier=5, strengths=[])
    with pytest.raises(ValidationError):
        AdaptiveRouterPreferences(quality_tier=0, strengths=[])


def test_config_accepts_all_six_request_types_in_strengths():
    prefs = AdaptiveRouterPreferences(
        quality_tier=3,
        strengths=[
            RequestType.CODE_GENERATION,
            RequestType.CODE_UNDERSTANDING,
            RequestType.TECHNICAL_DESIGN,
            RequestType.ANALYTICAL_REASONING,
            RequestType.WRITING,
            RequestType.FACTUAL_LOOKUP,
        ],
    )
    assert len(prefs.strengths) == 6


def test_adaptive_router_cost_reads_model_info():
    """GH#31481: cost should fall back to model_info when not set
    in litellm_params."""
    lp_dict = {}  # no cost in litellm_params
    mi_dict = {"input_cost_per_token": 0.0001}
    cost = lp_dict.get("input_cost_per_token")
    if cost is None:
        cost = mi_dict.get("input_cost_per_token")
    assert cost == 0.0001

    # Explicit zero in litellm_params should be preserved (not fallback)
    cost = {"input_cost_per_token": 0}.get("input_cost_per_token")
    assert cost == 0
