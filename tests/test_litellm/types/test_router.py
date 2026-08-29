import pytest

from litellm.types.router import (
    SPECIAL_MODEL_INFO_PARAMS,
    CredentialLiteLLMParams,
    Deployment,
    LiteLLM_Params,
    ModelInfo,
    anthropic_wif_fields_named,
    anthropic_wif_fields_present,
    holds_secret_pointer,
)
from litellm.types.utils import (
    CustomPricingLiteLLMParams,
    MirroredPricingParams,
    anthropic_wif_litellm_params,
)


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
    value = [{"range": [0, 128000], "input_cost_per_token": 3e-06}] if field == "tiered_pricing" else 3e-06
    deployment = Deployment(
        model_name="my-model",
        litellm_params=LiteLLM_Params(model="gpt-4o", **{field: value}),
    )
    assert getattr(deployment.model_info, field) == value
    assert deployment.model_info.model_dump(exclude_none=True)[field] == value


def test_deployment_mirrors_tiered_pricing_onto_model_info():
    """
    Regression: tiered_pricing set under a deployment's litellm_params was silently
    ignored at cost time because the Deployment mirror excluded it, so the logging
    path never flagged the deployment as custom-priced.
    """
    tiers = [
        {"range": [0, 3000], "input_cost_per_token": 3.25e-07, "output_cost_per_token": 1.95e-06},
        {"range": [3000, 128000], "input_cost_per_token": 6.5e-07, "output_cost_per_token": 3.9e-06},
    ]
    deployment = Deployment(
        model_name="my-model",
        litellm_params=LiteLLM_Params(model="anthropic/claude-haiku-4-5", tiered_pricing=tiers),
    )
    assert deployment.model_info.tiered_pricing == tiers


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
    with pytest.raises(ValueError, match="validation error for ModelInfo"):
        ModelInfo(id="x", input_cost_per_token="free")


def test_credential_litellm_params_declares_every_anthropic_wif_field():
    """Without these, get_deployment_credentials_with_provider round-trips litellm_params
    through a strict Pydantic dump and silently drops every WIF field before files/batches/
    passthrough callers see it -- the same #30235-shaped gap azure_ad_token closed above."""
    for field in anthropic_wif_litellm_params:
        assert field in CredentialLiteLLMParams.model_fields, field


def test_anthropic_wif_fields_round_trip_through_model_dump():
    values = {field: f"value-for-{field}" for field in anthropic_wif_litellm_params}
    values["anthropic_issuer_ttl_seconds"] = 300
    values["anthropic_disable_workload_identity_federation"] = True

    dumped = CredentialLiteLLMParams(**values).model_dump(exclude_none=True)

    for field, value in values.items():
        assert dumped[field] == value, field


def test_anthropic_wif_fields_present_reports_only_set_fields():
    assert anthropic_wif_fields_present({}) == ()
    assert anthropic_wif_fields_present({"model": "gpt-4o"}) == ()
    assert anthropic_wif_fields_present(
        {"anthropic_keycloak_token_url": "https://idp.example/token", "model": "gpt-4o"}
    ) == ("anthropic_keycloak_token_url",)


def test_anthropic_wif_fields_present_is_derived_from_the_shared_list():
    """A non-admin persistence gate built on this must automatically cover a field added
    later to anthropic_wif_litellm_params, not just the fields known when the gate was
    written -- so this must read the shared list rather than a hand-copied one."""
    values = {field: "set" for field in anthropic_wif_litellm_params}
    assert set(anthropic_wif_fields_present(values)) == set(anthropic_wif_litellm_params)


def test_anthropic_wif_fields_named_reports_keys_whatever_their_value():
    """The credential write gates must see a key a caller sets to ``None``: the federation
    resolver reacts to the key's presence, not its value, so ``{"anthropic_issuer_url": None}``
    wedges every deployment referencing the credential once persisted."""
    assert anthropic_wif_fields_named({}) == ()
    assert anthropic_wif_fields_named({"model": "gpt-4o"}) == ()
    assert anthropic_wif_fields_named({"anthropic_issuer_url": None}) == ("anthropic_issuer_url",)
    assert anthropic_wif_fields_present({"anthropic_issuer_url": None}) == ()
    assert anthropic_wif_fields_named(("anthropic_keycloak_token_url", "api_key")) == ("anthropic_keycloak_token_url",)


def test_anthropic_wif_fields_named_is_derived_from_the_shared_list():
    assert set(anthropic_wif_fields_named(frozenset(anthropic_wif_litellm_params))) == set(anthropic_wif_litellm_params)


@pytest.mark.parametrize("param_name", ["anthropic_issuer_signing_key_ref", "anthropic_keycloak_client_secret_ref"])
def test_wif_ref_fields_hold_secret_pointers(param_name: str):
    assert holds_secret_pointer(param_name)


@pytest.mark.parametrize("param_name", ["api_key", "anthropic_federation_rule_id", "anthropic_identity_token"])
def test_dereferenced_fields_do_not_hold_secret_pointers(param_name: str):
    assert not holds_secret_pointer(param_name)
