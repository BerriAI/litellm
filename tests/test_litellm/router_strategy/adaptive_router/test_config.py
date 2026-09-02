import pytest
from pydantic import ValidationError

from litellm.types.router import (
    AdaptiveRouterConfig,
    AdaptiveRouterEvaluationPrior,
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


def test_config_accepts_evaluation_priors():
    cfg = AdaptiveRouterConfig(
        available_models=["fast", "smart"],
        evaluation_priors=(
            AdaptiveRouterEvaluationPrior(
                request_type=RequestType.CODE_GENERATION,
                model="fast",
                successes=80,
                failures=20,
            ),
        ),
    )

    assert cfg.evaluation_priors[0].successes == 80


def test_config_accepts_bounded_exploration_rate():
    cfg = AdaptiveRouterConfig(available_models=["fast", "smart"], exploration_rate=0.05)

    assert cfg.exploration_rate == 0.05

    with pytest.raises(ValidationError):
        AdaptiveRouterConfig(available_models=["fast", "smart"], exploration_rate=1.1)


def test_config_rejects_evaluation_prior_for_unavailable_model():
    with pytest.raises(ValidationError, match="unavailable models"):
        AdaptiveRouterConfig.model_validate(
            {
                "available_models": ["fast"],
                "evaluation_priors": [
                    {
                        "request_type": "code_generation",
                        "model": "smart",
                        "successes": 1,
                        "failures": 1,
                    }
                ],
            }
        )
