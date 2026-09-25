"""
Test case normalization in LitellmParams for all guardrail types
"""

from typing import Literal

import pytest
from pydantic import ValidationError

from litellm.types.guardrails import BaseLitellmParams, LitellmParams, runtime_stream_scope


class TestLitellmParamsCaseNormalization:
    """Test that LitellmParams normalizes case for all guardrail types"""

    def test_presidio_guardrail_with_capitalized_default_action(self):
        """Test Presidio guardrail with capitalized default_action"""
        params = LitellmParams(
            guardrail="presidio",
            mode="post_call",
            default_action="Deny",  # Capitalized
        )
        assert params.default_action == "deny"

    def test_azure_guardrail_with_capitalized_default_action(self):
        """Test Azure guardrail with capitalized default_action"""
        params = LitellmParams(
            guardrail="azure/text_moderations",
            mode="pre_call",
            default_action="Allow",  # Capitalized
        )
        assert params.default_action == "allow"

    def test_tool_permission_with_capitalized_fields(self):
        """Test tool_permission with capitalized fields"""
        params = LitellmParams(
            guardrail="tool_permission",
            mode="post_call",
            default_action="DENY",  # Uppercase
            on_disallowed_action="BLOCK",  # Uppercase
        )
        assert params.default_action == "deny"
        assert params.on_disallowed_action == "block"

    def test_lakera_with_capitalized_default_action(self):
        """Test Lakera guardrail with capitalized default_action"""
        params = LitellmParams(
            guardrail="lakera_v2",
            mode="pre_call",
            default_action="Deny",  # Capitalized
        )
        assert params.default_action == "deny"

    def test_bedrock_with_capitalized_default_action(self):
        """Test Bedrock guardrail with capitalized default_action"""
        params = LitellmParams(
            guardrail="bedrock",
            mode="pre_call",
            default_action="Allow",  # Capitalized
        )
        assert params.default_action == "allow"

    def test_multiple_guardrails_all_normalized(self):
        """Test that all guardrail types benefit from normalization"""
        test_cases = [
            ("presidio", "Deny"),
            ("azure/text_moderations", "Allow"),
            ("tool_permission", "DENY"),
            ("lakera_v2", "allow"),  # Already lowercase - should still work
            ("bedrock", "Deny"),
        ]

        for guardrail_type, default_action_input in test_cases:
            params = LitellmParams(
                guardrail=guardrail_type,
                mode="pre_call",
                default_action=default_action_input,
            )
            # Should always be lowercase
            assert params.default_action.lower() == params.default_action
            # Should match the expected lowercase value
            assert params.default_action in ["allow", "deny"]

    def test_on_disallowed_action_all_cases(self):
        """Test on_disallowed_action normalization across all cases"""
        test_cases = ["block", "Block", "BLOCK", "rewrite", "Rewrite", "REWRITE"]

        for action in test_cases:
            params = LitellmParams(
                guardrail="tool_permission",
                mode="post_call",
                on_disallowed_action=action,
            )
            assert params.on_disallowed_action in ["block", "rewrite"]
            assert params.on_disallowed_action.islower()


class TestOnViolationAcceptedValues:
    """on_violation is shared by /v1/realtime guardrails and the mcp_security guardrail"""

    @pytest.mark.parametrize("action", ["block", "alert"])
    def test_mcp_security_policy_template_on_violation_is_accepted(self, action: Literal["block", "alert"]):
        params = LitellmParams(
            guardrail="mcp_security",
            mode="pre_call",
            default_on=True,
            on_violation=action,
        )
        assert params.on_violation == action

    @pytest.mark.parametrize("action", ["warn", "end_session"])
    def test_realtime_on_violation_still_accepted(self, action: Literal["warn", "end_session"]):
        params = LitellmParams(guardrail="presidio", mode="pre_call", on_violation=action)
        assert params.on_violation == action

    @pytest.mark.parametrize("action", ["block", "alert"])
    def test_mcp_only_on_violation_is_rejected_for_other_guardrails(self, action: Literal["block", "alert"]):
        with pytest.raises(ValidationError, match="only supported by guardrail='mcp_security'"):
            LitellmParams(guardrail="presidio", mode="pre_call", on_violation=action)

    def test_unknown_on_violation_is_rejected(self):
        with pytest.raises(ValidationError):
            LitellmParams(guardrail="mcp_security", mode="pre_call", on_violation="ignore")


class TestSensitiveDataRoutingValidation:
    """on_sensitive_data='route' requires a target model to be set"""

    def test_route_with_target_model_is_valid(self):
        params = LitellmParams(
            guardrail="presidio",
            mode="pre_call",
            on_sensitive_data="route",
            sensitive_data_route_to_model="on-prem-model",
        )
        assert params.on_sensitive_data == "route"
        assert params.sensitive_data_route_to_model == "on-prem-model"

    def test_route_without_target_model_raises(self):
        with pytest.raises(ValidationError, match="sensitive_data_route_to_model"):
            LitellmParams(
                guardrail="presidio",
                mode="pre_call",
                on_sensitive_data="route",
            )

    def test_base_params_route_without_target_model_raises(self):
        with pytest.raises(ValidationError, match="sensitive_data_route_to_model"):
            BaseLitellmParams(on_sensitive_data="route")

    def test_base_params_normalize_on_sensitive_data_case(self):
        params = BaseLitellmParams(
            on_sensitive_data="Route",
            sensitive_data_route_to_model="on-prem-model",
        )
        assert params.on_sensitive_data == "route"

    def test_base_params_capitalized_route_without_target_model_raises(self):
        with pytest.raises(ValidationError, match="sensitive_data_route_to_model"):
            BaseLitellmParams(on_sensitive_data="ROUTE")

    def test_block_without_target_model_is_valid(self):
        params = LitellmParams(
            guardrail="presidio",
            mode="pre_call",
            on_sensitive_data="block",
        )
        assert params.on_sensitive_data == "block"
        assert params.sensitive_data_route_to_model is None

    def test_on_sensitive_data_is_case_normalized(self):
        params = LitellmParams(
            guardrail="presidio",
            mode="pre_call",
            on_sensitive_data="Route",
            sensitive_data_route_to_model="on-prem-model",
        )
        assert params.on_sensitive_data == "route"

    def test_on_sensitive_data_uppercase_block_normalized(self):
        params = LitellmParams(
            guardrail="presidio",
            mode="pre_call",
            on_sensitive_data="BLOCK",
        )
        assert params.on_sensitive_data == "block"


class TestStreamScopeValidation:
    def test_scalar_is_case_normalized(self):
        params = LitellmParams(guardrail="bedrock", mode="post_call", stream_scope="Streaming")
        assert params.stream_scope == "streaming"

    def test_map_keys_and_values_are_normalized(self):
        params = LitellmParams(
            guardrail="bedrock",
            mode=["pre_call", "post_call"],
            stream_scope={"Pre_Call": "Both", "POST_CALL": "Non_Streaming"},
        )
        assert params.stream_scope == {"pre_call": "both", "post_call": "non_streaming"}

    def test_invalid_scalar_is_rejected(self):
        with pytest.raises(ValidationError, match="stream_scope must be one of"):
            LitellmParams(guardrail="bedrock", mode="post_call", stream_scope="chunks")

    def test_invalid_map_key_is_rejected(self):
        with pytest.raises(ValidationError, match="stream_scope keys must be guardrail modes"):
            LitellmParams(guardrail="bedrock", mode="post_call", stream_scope={"not_a_mode": "both"})

    def test_invalid_map_value_is_rejected(self):
        with pytest.raises(ValidationError, match="stream_scope must be one of"):
            LitellmParams(guardrail="bedrock", mode="post_call", stream_scope={"post_call": "sometimes"})

    def test_runtime_stream_scope_normalizes_direct_constructor_maps(self):
        default, by_hook = runtime_stream_scope({"Pre_Call": "streaming"})
        assert default == "both"
        assert dict(by_hook) == {"pre_call": "streaming"}

    def test_runtime_stream_scope_rejects_invalid_direct_input(self):
        with pytest.raises(ValueError, match="stream_scope must be one of"):
            runtime_stream_scope("chunks")
