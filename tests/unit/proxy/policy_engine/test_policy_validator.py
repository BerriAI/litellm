"""
Unit tests for PolicyValidator - tests policy configuration validation.

Tests validation of:
- Inheritance chains (parent exists, no circular deps)
- Guardrail names exist in registry
"""

from typing import Optional, Set
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy.policy_engine.policy_validator import PolicyValidator
from litellm.types.proxy.policy_engine import (
    Policy,
    PolicyGuardrails,
    PolicyValidationErrorType,
)


class _FakeTable:
    """Minimal stand-in for a Prisma table whose find_first matches on one field."""

    def __init__(self, existing: Set[str], match_field: str):
        self._existing = existing
        self._match_field = match_field

    async def find_first(self, where: dict) -> Optional[object]:
        return object() if where.get(self._match_field) in self._existing else None


class _FakeDB:
    def __init__(self, teams: Set[str], keys: Set[str]):
        self.litellm_teamtable = _FakeTable(teams, "team_alias")
        self.litellm_verificationtoken = _FakeTable(keys, "key_alias")


class _FakePrisma:
    """Injected prisma client so PolicyValidator can be unit-tested without a DB."""

    def __init__(self, teams: Set[str] = frozenset(), keys: Set[str] = frozenset()):
        self.db = _FakeDB(teams, keys)


class _FakeRouter:
    """Injected router exposing only what check_model_exists reads."""

    def __init__(self, model_names: Set[str]):
        self.model_names = list(model_names)


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
            validator,
            "get_available_guardrails",
            return_value={"pii_blocker", "toxicity_filter"},
        ):
            result = await validator.validate_policies(
                policies=policies, validate_db=False
            )

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
            validator,
            "get_available_guardrails",
            return_value={"pii_blocker", "toxicity_filter"},
        ):
            result = await validator.validate_policies(
                policies=policies, validate_db=False
            )

        assert result.valid is True
        assert len(result.errors) == 0


class TestAttachmentScopeValidation:
    """Regression tests for LIT-4199: attachments must not accept non-existent teams/keys/models."""

    @pytest.mark.asyncio
    async def test_nonexistent_team_is_flagged(self):
        validator = PolicyValidator(prisma_client=_FakePrisma(teams={"real-team"}))
        errors = await validator.find_invalid_scope_entries(
            policy_name="p", teams=["real-team", "ghost-team"]
        )
        assert [(e.field, e.value) for e in errors] == [("teams", "ghost-team")]
        assert errors[0].error_type == PolicyValidationErrorType.INVALID_TEAM

    @pytest.mark.asyncio
    async def test_existing_team_passes(self):
        validator = PolicyValidator(prisma_client=_FakePrisma(teams={"payments"}))
        errors = await validator.find_invalid_scope_entries(policy_name="p", teams=["payments"])
        assert errors == []

    @pytest.mark.asyncio
    async def test_trailing_star_wildcard_is_allowed_even_when_it_matches_nothing(self):
        validator = PolicyValidator(prisma_client=_FakePrisma(teams=set()))
        errors = await validator.find_invalid_scope_entries(
            policy_name="p", teams=["healthcare-*", "brand-new-*"]
        )
        assert errors == []

    @pytest.mark.asyncio
    async def test_only_trailing_star_counts_as_a_wildcard(self):
        # Request-time matching treats only a trailing "*" as a wildcard; "?" and a
        # non-trailing "*" are compared literally, so they are validated as concrete
        # aliases (and here resolve to nothing -> flagged).
        validator = PolicyValidator(prisma_client=_FakePrisma(teams=set()))
        errors = await validator.find_invalid_scope_entries(
            policy_name="p", teams=["ops-?", "heal*care"]
        )
        assert {e.value for e in errors} == {"ops-?", "heal*care"}

    @pytest.mark.asyncio
    async def test_keys_and_models_are_validated_too(self):
        validator = PolicyValidator(
            prisma_client=_FakePrisma(keys={"prod-key"}),
            llm_router=_FakeRouter(model_names={"gpt-4o"}),
        )
        errors = await validator.find_invalid_scope_entries(
            policy_name="p",
            keys=["prod-key", "ghost-key"],
            models=["gpt-4o", "ghost-model", "bedrock/*"],
        )
        flagged = {(e.field, e.value) for e in errors}
        assert ("keys", "ghost-key") in flagged
        assert ("models", "ghost-model") in flagged
        assert ("keys", "prod-key") not in flagged
        assert ("models", "gpt-4o") not in flagged
        assert ("models", "bedrock/*") not in flagged  # wildcard model allowed through

    @pytest.mark.asyncio
    async def test_no_scope_entries_returns_no_errors(self):
        validator = PolicyValidator(prisma_client=_FakePrisma())
        errors = await validator.find_invalid_scope_entries(policy_name="p")
        assert errors == []

    @pytest.mark.asyncio
    async def test_without_db_connection_assumes_valid(self):
        # Fail-open: with no DB we cannot verify existence, so nothing is blocked.
        validator = PolicyValidator(prisma_client=None)
        errors = await validator.find_invalid_scope_entries(policy_name="p", teams=["anything"])
        assert errors == []
