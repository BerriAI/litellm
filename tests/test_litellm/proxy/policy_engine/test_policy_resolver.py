"""
Unit tests for PolicyResolver - tests guardrail resolution.

Tests:
- Inheritance chain resolution
- Inheritance with add/remove
- Model conditions
"""

import pytest

from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
from litellm.types.proxy.policy_engine import (
    Policy,
    PolicyCondition,
    PolicyGuardrails,
    PolicyMatchContext,
)


class TestPolicyResolverInheritance:
    """Test resolve_policy_guardrails - inheritance and add/remove."""

    def test_resolve_simple_policy(self):
        """Test resolving guardrails for a simple policy."""
        policies = {
            "global": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker", "toxicity_filter"]),
            ),
        }

        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="global", policies=policies
        )

        assert set(resolved.guardrails) == {"pii_blocker", "toxicity_filter"}
        assert resolved.inheritance_chain == ["global"]

    def test_resolve_with_inheritance(self):
        """Test child policy inherits and adds guardrails from parent."""
        policies = {
            "base": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker"]),
            ),
            "healthcare": Policy(
                inherit="base",
                guardrails=PolicyGuardrails(add=["hipaa_audit"]),
            ),
        }

        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="healthcare", policies=policies
        )

        # Healthcare inherits pii_blocker from base and adds hipaa_audit
        assert set(resolved.guardrails) == {"pii_blocker", "hipaa_audit"}
        assert resolved.inheritance_chain == ["base", "healthcare"]

    def test_resolve_with_remove(self):
        """Test child policy can remove guardrails from parent."""
        policies = {
            "base": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker", "phi_blocker"]),
            ),
            "dev": Policy(
                inherit="base",
                guardrails=PolicyGuardrails(add=["toxicity_filter"], remove=["phi_blocker"]),
            ),
        }

        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="dev", policies=policies
        )

        # dev inherits pii_blocker from base, adds toxicity_filter, removes phi_blocker
        assert "pii_blocker" in resolved.guardrails
        assert "toxicity_filter" in resolved.guardrails
        assert "phi_blocker" not in resolved.guardrails

    def test_resolve_deep_inheritance_chain(self):
        """Test multi-level inheritance chain."""
        policies = {
            "root": Policy(
                guardrails=PolicyGuardrails(add=["root_guardrail"]),
            ),
            "middle": Policy(
                inherit="root",
                guardrails=PolicyGuardrails(add=["middle_guardrail"]),
            ),
            "leaf": Policy(
                inherit="middle",
                guardrails=PolicyGuardrails(add=["leaf_guardrail"]),
            ),
        }

        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="leaf", policies=policies
        )

        assert set(resolved.guardrails) == {"root_guardrail", "middle_guardrail", "leaf_guardrail"}
        assert resolved.inheritance_chain == ["root", "middle", "leaf"]


class TestPolicyResolverWithConditions:
    """Test resolve_policy_guardrails with model conditions."""

    def test_condition_matches(self):
        """Test guardrails are added when condition matches."""
        policies = {
            "gpt4-policy": Policy(
                guardrails=PolicyGuardrails(add=["toxicity_filter"]),
                condition=PolicyCondition(model="gpt-4.*"),
            ),
        }

        # GPT-4 should get guardrails
        context = PolicyMatchContext(team_alias="team", key_alias="k", model="gpt-4")
        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="gpt4-policy",
            policies=policies,
            context=context,
        )

        assert "toxicity_filter" in resolved.guardrails

    def test_condition_does_not_match(self):
        """Test guardrails are NOT added when condition doesn't match."""
        policies = {
            "gpt4-policy": Policy(
                guardrails=PolicyGuardrails(add=["toxicity_filter"]),
                condition=PolicyCondition(model="gpt-4.*"),
            ),
        }

        # GPT-3.5 should NOT get guardrails
        context = PolicyMatchContext(team_alias="team", key_alias="k", model="gpt-3.5")
        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="gpt4-policy",
            policies=policies,
            context=context,
        )

        assert "toxicity_filter" not in resolved.guardrails

    def test_no_condition_always_applies(self):
        """Test policy without condition always applies."""
        policies = {
            "global": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker"]),
            ),
        }

        context = PolicyMatchContext(team_alias="any", key_alias="any", model="any")
        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="global",
            policies=policies,
            context=context,
        )

        assert "pii_blocker" in resolved.guardrails

    def test_inheritance_with_condition(self):
        """Test inheritance works with conditions."""
        policies = {
            "base": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker"]),
            ),
            "child": Policy(
                inherit="base",
                guardrails=PolicyGuardrails(add=["child_guardrail"]),
                condition=PolicyCondition(model="gpt-4"),
            ),
        }

        # GPT-4 should get both base and child guardrails
        context_gpt4 = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-4")
        resolved_gpt4 = PolicyResolver.resolve_policy_guardrails(
            policy_name="child",
            policies=policies,
            context=context_gpt4,
        )
        assert "pii_blocker" in resolved_gpt4.guardrails
        assert "child_guardrail" in resolved_gpt4.guardrails

        # GPT-3.5 should only get base guardrails (child condition doesn't match)
        context_gpt35 = PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-3.5")
        resolved_gpt35 = PolicyResolver.resolve_policy_guardrails(
            policy_name="child",
            policies=policies,
            context=context_gpt35,
        )
        assert "pii_blocker" in resolved_gpt35.guardrails
        assert "child_guardrail" not in resolved_gpt35.guardrails
