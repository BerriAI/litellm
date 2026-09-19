import math
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.router_strategy.complexity_router.selective_policy import SelectiveHead, SelectivePolicy, verdict_features


def test_paired_policy_distinguishes_strong_only_from_cheap_only_success() -> None:
    rescue: Final = SelectivePolicy(
        version="test",
        feature_schema="v2-v1",
        target="paired",
        threshold=0.05,
        heads=(SelectiveHead(constant=1),),
    )
    cheap_only: Final = rescue.model_copy(update={"heads": (SelectiveHead(constant=2),)})
    assert not rescue.evaluate({}, 0.9, 0.9).use_efficient
    assert cheap_only.evaluate({}, 0.1, 0.9).use_efficient


def test_multiclass_head_maps_missing_outcomes_and_handles_large_logits() -> None:
    head: Final = SelectiveHead(
        features=("p_e",),
        coefficients=((0.0,), (0.0,), (0.0,)),
        intercept=(1000.0, 1000.0, 1000.0),
        classes=(0, 1, 3),
    )
    assert head.probabilities({"p_e": 0.5}) == pytest.approx((1 / 3, 1 / 3, 0, 1 / 3))


def test_scalar_calibration_clips_extreme_probabilities() -> None:
    head: Final = SelectiveHead(features=("logit_p",), coefficients=((1.0,),), intercept=(0.0,), classes=(0, 1))
    policy: Final = SelectivePolicy(
        version="test",
        feature_schema="v2-v1",
        target="scalar_calibration",
        threshold=0.05,
        heads=(head, head),
    )
    decision: Final = policy.evaluate({}, 0.0, 1.0)
    assert decision.score == pytest.approx(0.99998)
    assert not decision.use_efficient


def test_cost_policy_uses_incremental_dollars_and_inclusive_boundary() -> None:
    cheap: Final = SelectiveHead(features=("p_e",), coefficients=((0.0,),), intercept=(0.0,))
    strong: Final = SelectiveHead(features=("p_e",), coefficients=((0.0,),), intercept=(math.log(3.0),))
    policy: Final = SelectivePolicy(
        version="test",
        feature_schema="cap-v1",
        target="benefit_per_dollar",
        threshold=0.5,
        heads=(SelectiveHead(constant=1),),
        costs=(cheap, strong),
    )
    assert policy.evaluate({}, 0.9, 0.9).use_efficient
    assert not policy.model_copy(update={"threshold": 0.49}).evaluate({}, 0.9, 0.9).use_efficient


def test_overflow_falls_back_to_the_capable_model() -> None:
    cost: Final = SelectiveHead(features=("p_e",), coefficients=((0.0,),), intercept=(1000.0,))
    policy: Final = SelectivePolicy(
        version="test",
        feature_schema="cap-v1",
        target="benefit_per_dollar",
        threshold=1.0,
        heads=(SelectiveHead(constant=1),),
        costs=(cost, cost),
    )
    assert not policy.evaluate({}, 0.5, 0.9).use_efficient


@pytest.mark.parametrize("coefficient", [1e308, -1e308])
def test_overflowing_binary_logits_cannot_hide_as_equal_probabilities(coefficient: float) -> None:
    head: Final = SelectiveHead(
        features=("logit_p",), coefficients=((coefficient,),), intercept=(0.0,), classes=(0, 1)
    )
    policy: Final = SelectivePolicy(
        version="test",
        feature_schema="v2-v1",
        target="scalar_calibration",
        threshold=0.05,
        heads=(head, head),
    )
    assert not policy.evaluate({}, 0.99999, 0.99999).use_efficient


def test_per_model_policy_uses_both_fitted_heads() -> None:
    efficient: Final = SelectiveHead(
        features=("p_e",), coefficients=((0.0,),), intercept=(math.log(4.0),), classes=(0, 1)
    )
    capable: Final = SelectiveHead(
        features=("p_s",), coefficients=((0.0,),), intercept=(math.log(9.0),), classes=(0, 1)
    )
    policy: Final = SelectivePolicy(
        version="test",
        feature_schema="v2-v1",
        target="per_model",
        threshold=0.05,
        heads=(efficient, capable),
    )
    decision: Final = policy.evaluate({"p_e": 0.99, "p_s": 0.5}, 0.99, 0.5)
    assert decision.score == pytest.approx(0.1)
    assert not decision.use_efficient


@pytest.mark.parametrize(
    "values",
    [
        {"target": "per_model", "heads": [{"constant": 1}]},
        {"target": "rescue", "heads": [{"constant": 3}]},
        {"target": "rescue", "heads": [{"features": ["x"], "coefficients": [[1.0]], "intercept": [0.0]}]},
        {"target": "benefit_per_dollar", "heads": [{"constant": 1}], "costs": [{"constant": 0}, {"constant": 1}]},
    ],
)
def test_policy_rejects_incompatible_quality_and_cost_heads(values: object) -> None:
    from pydantic import TypeAdapter
    from collections.abc import Mapping

    fields: Final = TypeAdapter(Mapping[str, object]).validate_python(values)
    with pytest.raises(ValidationError):
        SelectivePolicy.model_validate({"version": "test", "feature_schema": "v2-v1", "threshold": 0.05, **fields})


@pytest.mark.parametrize(
    "invalid",
    [
        {"features": ["x"], "coefficients": [[1.0, 2.0]], "intercept": [0.0], "classes": [0, 1]},
        {"features": ["x"], "coefficients": [[1.0]], "intercept": [0.0], "classes": [1, 1]},
        {"constant": 1, "features": ["x"]},
        {"features": ["x"], "coefficients": [[float("nan")]], "intercept": [0.0], "classes": [0, 1]},
    ],
)
def test_invalid_coefficients_are_rejected_before_routing(invalid: object) -> None:
    with pytest.raises(ValidationError):
        SelectiveHead.model_validate(invalid)


def test_feature_contract_preserves_unknown_categories_without_task_identifiers() -> None:
    values: Final = verdict_features("A recursive Unicode parser", {"p_e": 0.7, "reasoning:unknown": 1.0})
    assert values["concept:recursive"] == values["concept:unicode"] == values["concept:parser"] == 1.0
    assert values["reasoning:unknown"] == 1.0
    assert "task_id" not in values
