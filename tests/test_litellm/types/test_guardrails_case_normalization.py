"""
Test case normalization in LitellmParams for all guardrail types
"""
import pytest
from litellm.types.guardrails import LitellmParams


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
