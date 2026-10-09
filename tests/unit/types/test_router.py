import logging

import pytest
from pydantic import ValidationError

from litellm.types.router import (
    SPECIAL_MODEL_INFO_PARAMS,
    CredentialLiteLLMParams,
    Deployment,
    GenericLiteLLMParams,
    LiteLLM_Params,
    ModelInfo,
    holds_secret_pointer,
    reject_server_owned_wif_params,
    server_owned_wif_fields_named,
    server_owned_wif_fields_present,
)
from litellm.types.utils import (
    CustomPricingLiteLLMParams,
    MirroredPricingParams,
    anthropic_wif_litellm_params,
    github_copilot_oauth_litellm_params,
    oauth_token_exchange_litellm_params,
    openai_wif_litellm_params,
    server_owned_wif_litellm_params,
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
        "cost_per_second",
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


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        ("true", True),
        (" False ", False),
        ("yes", True),
        (None, None),
        ("os.environ/DROP_PARAMS", "os.environ/DROP_PARAMS"),
        ("v2:gcm:ciphertext-from-a-pre-fix-row", "v2:gcm:ciphertext-from-a-pre-fix-row"),
    ],
)
def test_drop_params_coerces_flags_and_keeps_unresolved_strings(value, expected):
    assert GenericLiteLLMParams(drop_params=value).drop_params == expected


@pytest.mark.parametrize("value", [2, 2.5, [], {}])
def test_drop_params_ignores_non_flag_non_string_values_with_a_warning(value, caplog):
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        assert GenericLiteLLMParams(drop_params=value).drop_params is None
    assert f"drop_params={value!r} is not a flag value" in caplog.text


@pytest.mark.parametrize(
    "value", [True, "true", None, "os.environ/DROP_PARAMS", "v2:gcm:ciphertext-from-a-pre-fix-row"]
)
def test_drop_params_flags_and_strings_log_nothing(value, caplog):
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        GenericLiteLLMParams(drop_params=value)
    assert caplog.text == ""


def test_aws_session_tags_round_trip_as_sts_shaped_pairs():
    """The deployment field keeps the exact Key/Value shape STS AssumeRole expects."""
    params = LiteLLM_Params(
        model="bedrock/anthropic.claude-opus-5",
        aws_session_tags=[{"Key": "team", "Value": "genai"}, {"Key": "env", "Value": "prod"}],
    )

    assert params.model_dump(exclude_none=True)["aws_session_tags"] == [
        {"Key": "team", "Value": "genai"},
        {"Key": "env", "Value": "prod"},
    ]


@pytest.mark.parametrize(
    "aws_session_tags",
    ["team=genai", {"team": "genai"}, [{"key": "team", "value": "genai"}], [{"Key": "team"}]],
    ids=["string", "flat-dict", "lowercase-keys", "missing-value"],
)
def test_aws_session_tags_reject_shapes_sts_would_refuse(aws_session_tags):
    with pytest.raises(ValidationError, match="aws_session_tags"):
        LiteLLM_Params(model="bedrock/anthropic.claude-opus-5", aws_session_tags=aws_session_tags)


def test_provider_affinity_header_is_normalized():
    params = LiteLLM_Params(
        model="openai/gpt-4o-mini",
        provider_affinity_header="X-Conversation-Id",
    )

    assert params.provider_affinity_header == "X-Conversation-Id"
    assert params.model_dump(exclude_none=True)["provider_affinity_header"] == "X-Conversation-Id"


@pytest.mark.parametrize(
    "header",
    [
        "Authorization",
        "Proxy-Authorization",
        "Cookie",
        "Set-Cookie",
        "Host",
        "Content-Length",
        "Content-Type",
        "X-API-Key",
    ],
)
def test_provider_affinity_header_rejects_sensitive_or_transport_headers(header: str):
    with pytest.raises(ValueError, match="provider_affinity_header"):
        LiteLLM_Params(
            model="openai/gpt-4o-mini",
            provider_affinity_header=header,
        )


@pytest.mark.parametrize("header", ["", "X Conversation Id", "X-Conversation-Id\r\nInjected: true"])
def test_provider_affinity_header_rejects_invalid_header_names(header: str):
    with pytest.raises(ValueError, match="provider_affinity_header"):
        LiteLLM_Params(
            model="openai/gpt-4o-mini",
            provider_affinity_header=header,
        )


def test_model_info_parses_access_windows_time_strings():
    import datetime

    info = ModelInfo(
        id="x",
        access_windows=[
            {
                "start": "22:00",
                "end": "06:00",
                "timezone": "America/New_York",
                "team_ids": ["team-nightly"],
            }
        ],
    )
    window = info.access_windows[0]
    assert window.start == datetime.time(22, 0)
    assert window.end == datetime.time(6, 0)
    assert window.timezone == "America/New_York"
    assert window.team_ids == ("team-nightly",)


@pytest.mark.parametrize(
    "access_windows",
    [
        [{"start": "25:00", "end": "06:00", "timezone": "UTC", "team_ids": ["t"]}],
        [{"start": "22:00", "end": "06:00", "timezone": "Mars/Olympus", "team_ids": ["t"]}],
        [{"start": "22:00", "end": "06:00", "timezone": "UTC", "team_ids": []}],
    ],
    ids=["invalid-time", "unknown-timezone", "empty-team-ids"],
)
def test_model_info_rejects_invalid_access_windows(access_windows):
    with pytest.raises(ValidationError):
        ModelInfo(id="x", access_windows=access_windows)


def test_model_info_rejects_offset_aware_access_window_times():
    with pytest.raises(ValidationError):
        ModelInfo(
            id="x",
            access_windows=[{"start": "22:00+05:00", "end": "06:00", "timezone": "UTC", "team_ids": ["t"]}],
        )


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


def test_server_owned_wif_fields_present_reports_only_set_fields():
    assert server_owned_wif_fields_present({}) == ()
    assert server_owned_wif_fields_present({"model": "gpt-4o"}) == ()
    assert server_owned_wif_fields_present(
        {"anthropic_keycloak_token_url": "https://idp.example/token", "model": "gpt-4o"}
    ) == ("anthropic_keycloak_token_url",)


def test_server_owned_wif_fields_present_is_derived_from_the_shared_list():
    """A non-admin persistence gate built on this must automatically cover a field added
    later to server_owned_wif_litellm_params, not just the fields known when the gate was
    written -- so this must read the shared list rather than a hand-copied one."""
    values = {field: "set" for field in server_owned_wif_litellm_params}
    assert set(server_owned_wif_fields_present(values)) == set(server_owned_wif_litellm_params)


def test_server_owned_wif_fields_named_reports_keys_whatever_their_value():
    """The credential write gates must see a key a caller sets to ``None``: the federation
    resolver reacts to the key's presence, not its value, so ``{"anthropic_issuer_url": None}``
    wedges every deployment referencing the credential once persisted."""
    assert server_owned_wif_fields_named({}) == ()
    assert server_owned_wif_fields_named({"model": "gpt-4o"}) == ()
    assert server_owned_wif_fields_named({"anthropic_issuer_url": None}) == ("anthropic_issuer_url",)
    assert server_owned_wif_fields_present({"anthropic_issuer_url": None}) == ()
    assert server_owned_wif_fields_named(("anthropic_keycloak_token_url", "api_key")) == (
        "anthropic_keycloak_token_url",
    )


def test_server_owned_wif_fields_named_is_derived_from_the_shared_list():
    assert set(server_owned_wif_fields_named(frozenset(server_owned_wif_litellm_params))) == set(
        server_owned_wif_litellm_params
    )


@pytest.mark.parametrize("param_name", ["anthropic_issuer_signing_key_ref", "anthropic_keycloak_client_secret_ref"])
def test_wif_ref_fields_hold_secret_pointers(param_name: str):
    assert holds_secret_pointer(param_name)


@pytest.mark.parametrize("param_name", ["api_key", "anthropic_federation_rule_id", "anthropic_identity_token"])
def test_dereferenced_fields_do_not_hold_secret_pointers(param_name: str):
    assert not holds_secret_pointer(param_name)


def test_credential_litellm_params_declares_every_openai_wif_field():
    for field in openai_wif_litellm_params:
        assert field in CredentialLiteLLMParams.model_fields, field


def test_openai_wif_fields_round_trip_through_model_dump():
    values = {field: f"value-for-{field}" for field in openai_wif_litellm_params}

    dumped = CredentialLiteLLMParams(**values).model_dump(exclude_none=True)

    for field, value in values.items():
        assert dumped[field] == value, field


def test_server_owned_registry_includes_anthropic_openai_and_oauth_token_exchange():
    assert oauth_token_exchange_litellm_params == (
        "token_exchange_audience",
        "token_exchange_endpoint",
        "token_exchange_profile",
        "token_exchange_scope",
    )
    assert github_copilot_oauth_litellm_params == ("github_copilot_auth_type",)
    assert server_owned_wif_litellm_params == (
        anthropic_wif_litellm_params
        + openai_wif_litellm_params
        + oauth_token_exchange_litellm_params
        + github_copilot_oauth_litellm_params
    )
    assert set(openai_wif_litellm_params) == {
        "openai_identity_provider_id",
        "openai_service_account_id",
        "openai_identity_token_file",
    }


def test_credential_litellm_params_declares_each_oauth_token_exchange_field():
    for field in oauth_token_exchange_litellm_params:
        assert field in CredentialLiteLLMParams.model_fields, field


def test_server_owned_wif_fields_present_reports_openai_fields():
    assert server_owned_wif_fields_present(
        {"openai_identity_token_file": "/var/run/secrets/tokens/openai", "model": "gpt-4o"}
    ) == ("openai_identity_token_file",)
    assert server_owned_wif_fields_named({"openai_service_account_id": None}) == ("openai_service_account_id",)


@pytest.mark.parametrize("param_name", openai_wif_litellm_params)
def test_reject_server_owned_wif_params_names_each_openai_field(param_name: str):
    with pytest.raises(ValueError, match=param_name):
        reject_server_owned_wif_params({param_name: "client-supplied"})
