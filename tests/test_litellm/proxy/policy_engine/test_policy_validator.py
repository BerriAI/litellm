"""
Unit tests for PolicyValidator - tests policy configuration validation.

Tests validation of:
- Inheritance chains (parent exists, no circular deps)
- Guardrail names exist in registry
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.policy_engine.policy_validator import PolicyValidator
from litellm.types.proxy.policy_engine import (
    Policy,
    PolicyGuardrails,
    PolicyValidationErrorType,
)


class TestPolicyValidator:
    """Test policy validation logic."""

    @pytest.mark.asyncio
    async def test_validate_missing_parent_policy(self):
        """Test that referencing non-existent parent policy fails."""
        policies = {
            "child": Policy(
                inherit="nonexistent-parent",
                guardrails=PolicyGuardrails(add=["hipaa_audit"]),
            ),
        }

        validator = PolicyValidator(prisma_client=None)
        result = await validator.validate_policies(policies=policies, validate_db=False)

        assert result.valid is False
        assert any(
            e.error_type == PolicyValidationErrorType.INVALID_INHERITANCE
            for e in result.errors
        )

    @pytest.mark.asyncio
    async def test_validate_invalid_guardrail(self):
        """Test that referencing non-existent guardrail fails."""
        policies = {
            "test-policy": Policy(
                guardrails=PolicyGuardrails(add=["nonexistent_guardrail"]),
            ),
        }

        validator = PolicyValidator(prisma_client=None)
        with patch.object(
            validator, "get_available_guardrails", return_value={"pii_blocker", "toxicity_filter"}
        ):
            result = await validator.validate_policies(policies=policies, validate_db=False)

        assert result.valid is False
        assert any(
            e.error_type == PolicyValidationErrorType.INVALID_GUARDRAIL
            and e.value == "nonexistent_guardrail"
            for e in result.errors
        )

    @pytest.mark.asyncio
    async def test_validate_valid_policy(self):
        """Test that a valid policy passes validation."""
        policies = {
            "base": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker"]),
            ),
            "child": Policy(
                inherit="base",
                guardrails=PolicyGuardrails(add=["toxicity_filter"]),
            ),
        }

        validator = PolicyValidator(prisma_client=None)
        with patch.object(
            validator, "get_available_guardrails", return_value={"pii_blocker", "toxicity_filter"}
        ):
            result = await validator.validate_policies(policies=policies, validate_db=False)

        assert result.valid is True
        assert len(result.errors) == 0
