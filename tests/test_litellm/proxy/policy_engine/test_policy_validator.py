"""
Unit tests for PolicyValidator - tests policy configuration validation.

Tests validation of:
- Inheritance chains (parent exists, no circular deps)
- Guardrail names exist in registry
- Model names exist in router
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.policy_engine.policy_validator import PolicyValidator
from litellm.types.proxy.policy_engine import (
    Policy,
    PolicyGuardrails,
    PolicyScope,
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
                scope=PolicyScope(teams=["healthcare-team"]),
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
                scope=PolicyScope(teams=["*"]),
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
    async def test_validate_invalid_model(self):
        """Test that referencing non-existent model warns."""
        policies = {
            "test-policy": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker"]),
                scope=PolicyScope(models=["nonexistent-model"]),
            ),
        }

        # Mock the router with known model names
        mock_router = MagicMock()
        mock_router.model_names = {"gpt-4", "gpt-3.5-turbo"}
        # Mock pattern_router to return empty list (no pattern matches)
        mock_router.pattern_router.get_deployments_by_pattern.return_value = []

        validator = PolicyValidator(prisma_client=None, llm_router=mock_router)
        with patch.object(validator, "get_available_guardrails", return_value={"pii_blocker"}):
            result = await validator.validate_policies(policies=policies, validate_db=False)

        # Model validation is a warning, not an error
        assert any(
            w.error_type == PolicyValidationErrorType.INVALID_MODEL
            and w.value == "nonexistent-model"
            for w in result.warnings
        )
