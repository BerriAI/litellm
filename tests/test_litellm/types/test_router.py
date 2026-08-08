import pytest

from litellm.types.router import (
    SPECIAL_MODEL_INFO_PARAMS,
    Deployment,
    LiteLLM_Params,
    ModelInfo,
)
from litellm.types.utils import CustomPricingLiteLLMParams, MirroredPricingParams


def test_model_info_declares_mirrored_pricing_fields():
    """The pricing keys Deployment mirrors onto model_info must be declared fields, not
    extras that only survive because ModelInfo sets extra="allow"."""
    for field in SPECIAL_MODEL_INFO_PARAMS:
        assert field in ModelInfo.model_fields

    info = ModelInfo(id="x", input_cost_per_token=1e-06)
    assert info.__pydantic_extra__ == {}
    assert info.input_cost_per_token == 1e-06


def test_special_model_info_params_cannot_drift_from_the_mirror():
    assert SPECIAL_MODEL_INFO_PARAMS == tuple(MirroredPricingParams.model_fields)
    assert set(SPECIAL_MODEL_INFO_PARAMS) <= set(CustomPricingLiteLLMParams.model_fields)
    assert set(SPECIAL_MODEL_INFO_PARAMS) <= set(LiteLLM_Params.model_fields)


def test_custom_pricing_params_keeps_every_field_it_had():
    """The mirrored fields moved to a base class; none of them may go missing from
    CustomPricingLiteLLMParams, whose model_fields drive custom-pricing detection."""
    for field in (
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_character",
        "output_cost_per_character",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
        "input_cost_per_second",
        "cache_read_input_token_cost_flex",
        "input_cost_per_character_above_128k_tokens",
        "output_cost_per_audio_token",
    ):
        assert field in CustomPricingLiteLLMParams.model_fields


@pytest.mark.parametrize("field", SPECIAL_MODEL_INFO_PARAMS)
def test_deployment_mirrors_pricing_from_litellm_params_onto_model_info(field):
    deployment = Deployment(
        model_name="my-model",
        litellm_params=LiteLLM_Params(model="gpt-4o", **{field: 3e-06}),
    )
    assert getattr(deployment.model_info, field) == 3e-06
    assert deployment.model_info.model_dump(exclude_none=True)[field] == 3e-06


def test_unset_pricing_is_still_absent_from_dumps():
    """/model/info responses and DB writes dump model_info with exclude_none=True, so
    declaring the pricing fields must not start emitting ~6 null keys per deployment."""
    dumped = ModelInfo(id="x").model_dump(exclude_none=True)
    assert [field for field in SPECIAL_MODEL_INFO_PARAMS if field in dumped] == []


def test_pricing_strings_are_coerced_to_float():
    """Cost values arrive from the DB and the Admin UI as strings; they must land as
    floats so cost calculation doesn't multiply a str."""
    info = ModelInfo(id="x", output_cost_per_token="0.000002")
    assert info.output_cost_per_token == 2e-06


def test_invalid_pricing_is_rejected():
    with pytest.raises(ValueError):
        ModelInfo(id="x", input_cost_per_token="free")
