"""
Unit tests for PolicyResolver - tests guardrail resolution for request contexts.
"""

import pytest

from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
from litellm.types.proxy.policy_engine import (
    Policy,
    PolicyGuardrails,
    PolicyMatchContext,
    PolicyScope,
)


class TestPolicyMatcherGetMatchingPolicies:
    """Test resolve_guardrails_for_context - the main entry point."""

    def test_resolve_guardrails_simple_match(self):
        """Test resolving guardrails for a simple matching policy."""
        policies = {
            "global": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker", "toxicity_filter"]),
                scope=PolicyScope(teams=["*"]),
            ),
        }

        context = PolicyMatchContext(team_alias="any-team", key_alias="k", model="gpt-4")
        guardrails = PolicyResolver.resolve_guardrails_for_context(
            context=context, policies=policies
        )

        assert set(guardrails) == {"pii_blocker", "toxicity_filter"}

    def test_resolve_guardrails_with_inheritance(self):
        """Test child policy inherits and adds guardrails from parent."""
        policies = {
            "base": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker"]),
                scope=PolicyScope(teams=["*"]),
            ),
            "healthcare": Policy(
                inherit="base",
                guardrails=PolicyGuardrails(add=["hipaa_audit"]),
                scope=PolicyScope(teams=["healthcare-team"]),
            ),
        }

        context = PolicyMatchContext(team_alias="healthcare-team", key_alias="k", model="gpt-4")
        guardrails = PolicyResolver.resolve_guardrails_for_context(
            context=context, policies=policies
        )

        # Both base and healthcare match, healthcare inherits from base
        assert set(guardrails) == {"pii_blocker", "hipaa_audit"}

    def test_resolve_guardrails_with_remove(self):
        """Test child policy can remove guardrails from parent in its inheritance chain."""
        policies = {
            "base": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker", "phi_blocker"]),
                scope=PolicyScope(teams=["internal-only"]),  # Does NOT match dev-team
            ),
            "dev": Policy(
                inherit="base",
                guardrails=PolicyGuardrails(add=["toxicity_filter"], remove=["phi_blocker"]),
                scope=PolicyScope(teams=["dev-team"]),  # Only this matches
            ),
        }

        # Only dev policy matches (base scope doesn't match)
        context = PolicyMatchContext(team_alias="dev-team", key_alias="k", model="gpt-4")
        guardrails = PolicyResolver.resolve_guardrails_for_context(
            context=context, policies=policies
        )

        # dev inherits pii_blocker from base, adds toxicity_filter, removes phi_blocker
        assert "pii_blocker" in guardrails
        assert "toxicity_filter" in guardrails
        assert "phi_blocker" not in guardrails

    def test_resolve_guardrails_no_match(self):
        """Test returns empty list when no policies match."""
        policies = {
            "healthcare": Policy(
                guardrails=PolicyGuardrails(add=["hipaa_audit"]),
                scope=PolicyScope(teams=["healthcare-team"]),
            ),
        }

        context = PolicyMatchContext(team_alias="finance-team", key_alias="k", model="gpt-4")
        guardrails = PolicyResolver.resolve_guardrails_for_context(
            context=context, policies=policies
        )

        assert guardrails == []
