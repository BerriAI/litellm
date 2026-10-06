"""
Tests for get_litellm_params and related helpers.

Ensures backward compatibility after sparse kwargs extraction optimization.
"""

from typing import Final

import pytest
from pydantic import ValidationError

from litellm.constants import CONTROL_OPTIONS_KEY
from litellm.litellm_core_utils.get_litellm_params import (
    _OPTIONAL_KWARGS_KEYS,
    InvalidControlOption,
    _get_base_model_from_litellm_call_metadata,
    get_litellm_params,
    parse_control_options,
    stored_control_options,
)
from litellm.types.litellm_params import ControlOptions


def _funnel_kwargs_completion_forwards(monkeypatch, wif_kwargs: dict[str, object]) -> dict[str, object]:
    """completion() names its get_litellm_params arguments one by one, so a key the funnel knows
    is still dropped unless that call site forwards it from its own kwargs."""
    from unittest.mock import MagicMock

    import litellm
    import litellm.main as litellm_main

    spy = MagicMock(wraps=litellm_main.get_litellm_params)
    monkeypatch.setattr(  # test-quality-ok: completion() has no injection seam for its kwargs funnel
        litellm_main, "get_litellm_params", spy
    )
    litellm.completion(
        model="anthropic/claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="ok",
        **wif_kwargs,
    )
    return spy.call_args.kwargs


NAMED_PRICE_PARAMS: Final = frozenset(
    {
        "input_cost_per_token",
        "output_cost_per_token",
        "cost_per_second",
        "input_cost_per_second",
        "output_cost_per_second",
    }
)


class TestGetBaseModelFromLitellmCallMetadata:
    def test_none_metadata_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata(None) is None

    def test_empty_metadata_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({}) is None

    def test_missing_model_info_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({"foo": "bar"}) is None

    def test_model_info_none_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({"model_info": None}) is None

    def test_model_info_empty_dict_returns_none(self):
        assert _get_base_model_from_litellm_call_metadata({"model_info": {}}) is None

    def test_returns_base_model(self):
        result = _get_base_model_from_litellm_call_metadata(
            {"model_info": {"base_model": "gpt-4"}}
        )
        assert result == "gpt-4"


class TestGetLitellmParamsKwargsExtraction:
    """Verify that optional kwargs are correctly extracted via sparse extraction."""

    def test_no_kwargs_omits_optional_keys(self):
        """When no kwargs passed, optional keys are absent; the named price params are present as None."""
        result = get_litellm_params(api_key="test-key")
        for key in _OPTIONAL_KWARGS_KEYS - NAMED_PRICE_PARAMS:
            assert key not in result
        for key in NAMED_PRICE_PARAMS:
            assert result[key] is None

    def test_custom_pricing_kwargs_are_extracted(self) -> None:
        from litellm.litellm_core_utils.litellm_logging import use_custom_pricing_for_model
        from litellm.types.router import CustomPricingLiteLLMParams

        assert set(CustomPricingLiteLLMParams.model_fields) <= _OPTIONAL_KWARGS_KEYS

        result = get_litellm_params(output_cost_per_image=0.08, input_cost_per_audio_token=1e-6)
        assert result["output_cost_per_image"] == 0.08
        assert result["input_cost_per_audio_token"] == 1e-6
        assert use_custom_pricing_for_model(result) is True

        result_without_prices = get_litellm_params()
        assert "output_cost_per_image" not in result_without_prices
        assert use_custom_pricing_for_model(result_without_prices) is False

    def test_present_kwargs_are_extracted(self):
        result = get_litellm_params(
            aws_region_name="us-east-1",
            timeout=30,
            rpm=100,
        )
        assert result["aws_region_name"] == "us-east-1"
        assert result["timeout"] == 30
        assert result["rpm"] == 100

    def test_s3_endpoint_kwargs_are_extracted_when_provided(self):
        result = get_litellm_params(
            s3_endpoint_url="https://bucket.vpce-abc.s3.us-east-1.vpce.amazonaws.com",
            s3_region_name="us-east-1",
        )
        assert result["s3_endpoint_url"] == "https://bucket.vpce-abc.s3.us-east-1.vpce.amazonaws.com"
        assert result["s3_region_name"] == "us-east-1"

        result_without_s3_kwargs = get_litellm_params()
        assert "s3_endpoint_url" not in result_without_s3_kwargs
        assert "s3_region_name" not in result_without_s3_kwargs

    def test_a_caller_supplied_control_options_key_is_not_carried(self) -> None:
        assert CONTROL_OPTIONS_KEY not in get_litellm_params(**{CONTROL_OPTIONS_KEY: {"stream_chunk_size": 64}})

    def test_s3_credential_kwargs_are_forwarded_for_s3_signing(self):
        result = get_litellm_params(s3_access_key_id="s3-key", s3_secret_access_key="s3-secret")
        assert result["s3_access_key_id"] == "s3-key"
        assert result["s3_secret_access_key"] == "s3-secret"

        result_without_s3_kwargs = get_litellm_params()
        assert "s3_access_key_id" not in result_without_s3_kwargs
        assert "s3_secret_access_key" not in result_without_s3_kwargs

    def test_subset_of_kwargs_only_includes_provided(self):
        """Only provided kwargs appear, others remain absent."""
        result = get_litellm_params(azure_ad_token="token123")
        assert result["azure_ad_token"] == "token123"
        assert "aws_region_name" not in result
        assert "timeout" not in result

    def test_unknown_kwargs_are_ignored(self):
        result = get_litellm_params(some_random_kwarg="value")
        assert "some_random_kwarg" not in result

    def test_all_optional_kwargs_extractable(self):
        """Every key in _OPTIONAL_KWARGS_KEYS can be extracted."""
        kwargs = {key: f"val_{key}" for key in _OPTIONAL_KWARGS_KEYS}
        result = get_litellm_params(**kwargs)
        for key in _OPTIONAL_KWARGS_KEYS:
            assert result[key] == f"val_{key}"


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"stream_chunk_size": 64, "temperature": 0.2}, ControlOptions(stream_chunk_size=64)),
        ({"stream_chunk_size": "64"}, ControlOptions(stream_chunk_size=64)),
        ({"stream_chunk_size": None}, ControlOptions()),
        ({"temperature": 0.2}, ControlOptions()),
    ],
)
def test_control_options_are_read_from_the_request_kwargs(kwargs: dict[str, object], expected: ControlOptions) -> None:
    assert parse_control_options(kwargs) == expected


@pytest.mark.parametrize(
    "raw,shown",
    [
        ("sixty-four", "'sixty-four'"),
        (" 64", "' 64'"),
        ("-1", "'-1'"),
        ("\uff16\uff14", "'\uff16\uff14'"),
        ("x" * 500, "'xxxxxxxxxxxx...xxxxxxxxxxxxx'"),
        pytest.param(-(10**5000), "<int of 16610 bits>", id="huge_negative_int"),
        pytest.param(-(2**64 - 1), "-18446744073709551615", id="64_bit_negative_int"),
        pytest.param(-(2**64), "<int of 65 bits>", id="65_bit_negative_int"),
        pytest.param([-(10**5000)], "[<int of 16610 bits>]", id="nested_huge_int"),
        pytest.param(10**18, "1000000000000000000", id="19_digit_int"),
        pytest.param("1" + "0" * 18, "'1000000000000000000'", id="19_digit_string"),
        pytest.param("9" * 5000, "'999999999999...9999999999999'", id="5000_digit_string"),
        pytest.param("0" * 18 + "1", "'0000000000000000001'", id="19_digit_string_with_leading_zeros"),
        (64.0, "64.0"),
        (True, "True"),
        (0, "0"),
        ("0", "'0'"),
        (-1, "-1"),
    ],
)
def test_control_options_reject_a_stream_chunk_size_that_is_not_a_positive_int(raw: object, shown: str) -> None:
    assert parse_control_options({"stream_chunk_size": raw}) == InvalidControlOption(
        param="stream_chunk_size",
        message=f"Invalid stream_chunk_size={shown}: expected a positive integer of at most 18 digits",
    )


@pytest.mark.parametrize("raw", [10**18 - 1, "9" * 18], ids=["int", "digit_string"])
def test_control_options_accept_the_largest_18_digit_value(raw: object) -> None:
    assert parse_control_options({"stream_chunk_size": raw}) == ControlOptions(stream_chunk_size=10**18 - 1)


def test_control_options_accept_an_18_digit_string_with_leading_zeros() -> None:
    assert parse_control_options({"stream_chunk_size": "0" * 17 + "1"}) == ControlOptions(stream_chunk_size=1)


@pytest.mark.parametrize("raw", [0, -1, "sixty-four", 64.0, True])
def test_control_options_enforce_their_rule_at_construction(raw: object) -> None:
    with pytest.raises(ValidationError):
        ControlOptions(stream_chunk_size=raw)  # pyright: ignore[reportArgumentType]  # the invalid type is the input


@pytest.mark.parametrize(
    "litellm_params,expected",
    [
        ({CONTROL_OPTIONS_KEY: ControlOptions(stream_chunk_size=64)}, ControlOptions(stream_chunk_size=64)),
        ({}, ControlOptions()),
        ({CONTROL_OPTIONS_KEY: {"stream_chunk_size": 64}}, ControlOptions()),
        ({"stream_chunk_size": 64}, ControlOptions()),
    ],
)
def test_stored_control_options_reads_only_the_validated_options(
    litellm_params: dict[str, object], expected: ControlOptions
) -> None:
    assert stored_control_options(litellm_params) == expected


class TestGetLitellmParamsBaseModel:
    """Verify base_model resolution precedence."""

    def test_explicit_base_model_takes_precedence(self):
        result = get_litellm_params(
            base_model="explicit",
            metadata={"model_info": {"base_model": "from-metadata"}},
        )
        assert result["base_model"] == "explicit"

    def test_falls_back_to_metadata(self):
        result = get_litellm_params(
            metadata={"model_info": {"base_model": "from-metadata"}}
        )
        assert result["base_model"] == "from-metadata"

    def test_none_when_no_source(self):
        result = get_litellm_params()
        assert result["base_model"] is None


class TestGetLitellmParamsExplicitFields:
    """Verify explicit parameters are always present in the result."""

    def test_explicit_params_always_present(self):
        result = get_litellm_params()
        # Spot-check a few explicit keys that should always be in the dict
        expected_keys = [
            "acompletion",
            "api_key",
            "force_timeout",
            "verbose",
            "custom_llm_provider",
            "api_base",
            "metadata",
            "model_info",
            "max_retries",
            "ssl_verify",
            "api_version",
        ]
        for key in expected_keys:
            assert key in result

    def test_no_log_from_kwargs(self):
        """no-log can come via **kwargs as well as the explicit param."""
        result = get_litellm_params(**{"no-log": True})
        assert result["no-log"] is True

    def test_no_log_from_explicit_param(self):
        result = get_litellm_params(no_log=True)
        assert result["no-log"] is True


class TestGetLitellmParamsDataResidency:
    """Verify that data_residency is inferred from OpenAI regional api_base."""

    def test_eu_host_resolves_to_eu(self):
        result = get_litellm_params(
            custom_llm_provider="openai",
            api_base="https://eu.api.openai.com/v1",
        )
        assert result["data_residency"] == "eu"

    def test_us_host_resolves_to_us(self):
        result = get_litellm_params(
            custom_llm_provider="openai",
            api_base="https://us.api.openai.com/v1",
        )
        assert result["data_residency"] == "us"

    def test_global_host_resolves_to_none(self):
        result = get_litellm_params(
            custom_llm_provider="openai",
            api_base="https://api.openai.com/v1",
        )
        assert result["data_residency"] is None

    def test_no_api_base_is_none(self):
        result = get_litellm_params(custom_llm_provider="openai")
        assert result["data_residency"] is None

    def test_non_openai_provider_does_not_resolve(self):
        """Regional OpenAI host doesn't apply to other providers."""
        result = get_litellm_params(
            custom_llm_provider="anthropic",
            api_base="https://eu.api.openai.com/v1",
        )
        assert result["data_residency"] is None


class TestMetadataFallsBackToLitellmMetadata:
    def test_metadata_falls_back_to_litellm_metadata_when_absent(self):
        result = get_litellm_params(litellm_metadata={"trace_id": "trace-1"})
        assert result["metadata"] == {"trace_id": "trace-1"}
        assert result["litellm_metadata"] == {"trace_id": "trace-1"}

    def test_empty_metadata_falls_back_to_litellm_metadata(self):
        result = get_litellm_params(metadata={}, litellm_metadata={"trace_id": "trace-1"})
        assert result["metadata"] == {"trace_id": "trace-1"}

    def test_metadata_wins_when_both_present(self):
        result = get_litellm_params(
            metadata={"trace_id": "from-metadata"},
            litellm_metadata={"trace_id": "from-litellm-metadata"},
        )
        assert result["metadata"] == {"trace_id": "from-metadata"}

    @pytest.mark.parametrize("bad_value", ["not-json-a-string", 12345, ["a"], True])
    def test_non_dict_litellm_metadata_is_ignored(self, bad_value):
        result = get_litellm_params(litellm_metadata=bad_value)
        assert result["metadata"] is None

    def test_metadata_stays_none_without_litellm_metadata(self):
        result = get_litellm_params(api_key="test-key")
        assert result["metadata"] is None

    def test_session_and_trace_id_derived_from_litellm_metadata(self):
        result = get_litellm_params(
            litellm_metadata={"trace_id": "trace-1", "session_id": "session-1"},
        )
        assert result["litellm_session_id"] == "session-1"
        assert result["litellm_trace_id"] == "trace-1"

    def test_explicit_session_and_trace_id_are_not_overridden(self):
        result = get_litellm_params(
            litellm_session_id="explicit-session",
            litellm_trace_id="explicit-trace",
            litellm_metadata={"trace_id": "trace-1", "session_id": "session-1"},
        )
        assert result["litellm_session_id"] == "explicit-session"
        assert result["litellm_trace_id"] == "explicit-trace"

    def test_litellm_metadata_fallback_is_copied_not_aliased(self):
        litellm_metadata = {"trace_id": "trace-1"}

        result = get_litellm_params(litellm_metadata=litellm_metadata)

        assert result["metadata"] == litellm_metadata
        assert result["metadata"] is not litellm_metadata
        result["metadata"].pop("trace_id")
        assert litellm_metadata == {"trace_id": "trace-1"}


@pytest.mark.parametrize(
    "value, expected",
    [("true", True), ("false", False), (" TRUE ", True), (True, True), (None, None), ("os.environ/DROP_PARAMS", None)],
)
def test_drop_params_strings_reach_litellm_params_as_flags(
    value: str | bool | None, expected: bool | None
) -> None:
    assert get_litellm_params(drop_params=value)["drop_params"] is expected


class TestAnthropicWifKeys:
    """The six anthropic_* WIF keys need dual registration: carried by the kwargs
    funnel into litellm_params (where the Anthropic auth tier reads them) AND
    listed in all_litellm_params (so the extra_body sweep never sends them to
    /v1/messages)."""

    SIX_KEYS = {
        "anthropic_federation_rule_id": "fdrl_1",
        "anthropic_organization_id": "org-1",
        "anthropic_service_account_id": "svcacct_1",
        "anthropic_federation_workspace_id": "wrkspc_1",
        "anthropic_identity_token_file": "/var/run/secrets/tok",
        "anthropic_identity_token": "oidc/env/TOK",
    }

    def test_keys_survive_into_litellm_params(self):
        params = get_litellm_params(**self.SIX_KEYS)
        for key, value in self.SIX_KEYS.items():
            assert params[key] == value

    def test_keys_are_forwarded_from_completion_kwargs(self, monkeypatch):
        forwarded = _funnel_kwargs_completion_forwards(monkeypatch, self.SIX_KEYS)
        assert {key: forwarded[key] for key in self.SIX_KEYS} == self.SIX_KEYS

    def test_keys_stay_out_of_the_provider_body(self):
        from litellm.types.utils import all_litellm_params

        for key in self.SIX_KEYS:
            assert key in all_litellm_params

    def test_keys_absent_when_not_configured(self):
        params = get_litellm_params()
        for key in self.SIX_KEYS:
            assert key not in params


class TestAnthropicWifIdentitySourceKeys:
    """Phase 1 adds 11 more anthropic_* WIF keys (the anthropic_identity_source discriminator
    plus the internal_issuer/keycloak identity-source fields) that need the same dual
    registration as the original six tested above."""

    NEW_KEYS = {
        "anthropic_identity_source": "keycloak",
        "anthropic_issuer_url": "https://issuer.example",
        "anthropic_issuer_subject": "svc-account",
        "anthropic_issuer_audience": "https://api.anthropic.com",
        "anthropic_issuer_ttl_seconds": "300",
        "anthropic_issuer_signing_key_ref": "oidc/env/ISSUER_KEY",
        "anthropic_keycloak_token_url": "https://kc.example/realms/r/protocol/openid-connect/token",
        "anthropic_keycloak_client_id": "litellm",
        "anthropic_keycloak_auth_method": "client_secret_basic",
        "anthropic_keycloak_client_secret_ref": "oidc/env/KC_SECRET",
        "anthropic_keycloak_scope": "anthropic-wif",
        # Server-set when a client redirects api_base; carried here so it is not dropped in transit
        "anthropic_disable_workload_identity_federation": True,
    }

    def test_new_keys_are_exactly_the_non_legacy_registered_set(self):
        """Fails the moment a key is added to ANTHROPIC_WIF_KWARGS_KEYS without a matching entry
        here (or vice versa), catching drift between what wif.py dispatches on and what this
        test (and the funnel/provider-body tests below) actually exercises."""
        from litellm.types.workload_identity import ANTHROPIC_WIF_KWARGS_KEYS

        assert set(self.NEW_KEYS) == ANTHROPIC_WIF_KWARGS_KEYS - set(TestAnthropicWifKeys.SIX_KEYS)

    def test_keys_survive_into_litellm_params(self):
        params = get_litellm_params(**self.NEW_KEYS)
        for key, value in self.NEW_KEYS.items():
            assert params[key] == value

    def test_keys_are_forwarded_from_completion_kwargs(self, monkeypatch):
        forwarded = _funnel_kwargs_completion_forwards(monkeypatch, self.NEW_KEYS)
        assert {key: forwarded[key] for key in self.NEW_KEYS} == self.NEW_KEYS

    def test_keys_stay_out_of_the_provider_body(self):
        from litellm.types.utils import all_litellm_params

        for key in self.NEW_KEYS:
            assert key in all_litellm_params

    def test_keys_absent_when_not_configured(self):
        params = get_litellm_params()
        for key in self.NEW_KEYS:
            assert key not in params


class TestOpenAIWifKeys:
    """The three openai_* WIF keys carry a deployment's federation identity through the kwargs
    funnel into litellm_params (where the OpenAI client factory reads them) and stay out of the
    provider body, exactly like the anthropic_* keys above."""

    THREE_KEYS = {
        "openai_identity_provider_id": "idp_1",
        "openai_service_account_id": "user-1",
        "openai_identity_token_file": "/var/run/secrets/tokens/openai",
    }

    def test_keys_are_exactly_the_registered_set(self):
        from litellm.types.workload_identity import OPENAI_WIF_KWARGS_KEYS

        assert set(self.THREE_KEYS) == OPENAI_WIF_KWARGS_KEYS

    def test_keys_survive_into_litellm_params(self):
        params = get_litellm_params(**self.THREE_KEYS)
        for key, value in self.THREE_KEYS.items():
            assert params[key] == value

    def test_keys_are_forwarded_from_completion_kwargs(self, monkeypatch):
        forwarded = _funnel_kwargs_completion_forwards(monkeypatch, self.THREE_KEYS)
        assert {key: forwarded[key] for key in self.THREE_KEYS} == self.THREE_KEYS

    def test_keys_stay_out_of_the_provider_body(self):
        from litellm.types.utils import all_litellm_params

        for key in self.THREE_KEYS:
            assert key in all_litellm_params

    def test_keys_absent_when_not_configured(self):
        params = get_litellm_params()
        for key in self.THREE_KEYS:
            assert key not in params
