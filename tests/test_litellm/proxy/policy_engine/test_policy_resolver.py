"""
Unit tests for PolicyResolver - tests guardrail resolution.

Tests:
- Inheritance chain resolution
- Inheritance with add/remove
- Conditional statements
"""

import pytest

from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
from litellm.types.proxy.policy_engine import (
    ConditionOperator,
    Policy,
    PolicyCondition,
    PolicyGuardrails,
    PolicyMatchContext,
    PolicyStatement,
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


class TestPolicyResolverWithStatements:
    """Test resolve_policy_guardrails with conditional statements."""

    def test_statement_condition_matches(self):
        """Test statement guardrails are added when condition matches."""
        policies = {
            "conditional-policy": Policy(
                guardrails=PolicyGuardrails(add=["base_guardrail"]),
                statements=[
                    PolicyStatement(
                        sid="GPT4Safety",
                        guardrails=["toxicity_filter"],
                        condition=PolicyCondition(
                            model=ConditionOperator(in_=["gpt-4", "gpt-4-turbo"])
                        ),
                    ),
                ],
            ),
        }

        # GPT-4 should get both base and statement guardrails
        context = PolicyMatchContext(team_alias="team", key_alias="k", model="gpt-4")
        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="conditional-policy",
            policies=policies,
            context=context,
        )

        assert "base_guardrail" in resolved.guardrails
        assert "toxicity_filter" in resolved.guardrails

    def test_statement_condition_does_not_match(self):
        """Test statement guardrails are NOT added when condition doesn't match."""
        policies = {
            "conditional-policy": Policy(
                guardrails=PolicyGuardrails(add=["base_guardrail"]),
                statements=[
                    PolicyStatement(
                        sid="GPT4Safety",
                        guardrails=["toxicity_filter"],
                        condition=PolicyCondition(
                            model=ConditionOperator(in_=["gpt-4", "gpt-4-turbo"])
                        ),
                    ),
                ],
            ),
        }

        # GPT-3.5 should only get base guardrails, not statement guardrails
        context = PolicyMatchContext(team_alias="team", key_alias="k", model="gpt-3.5")
        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="conditional-policy",
            policies=policies,
            context=context,
        )

        assert "base_guardrail" in resolved.guardrails
        assert "toxicity_filter" not in resolved.guardrails

    def test_multiple_statements_some_match(self):
        """Test multiple statements where only some match."""
        policies = {
            "multi-statement": Policy(
                guardrails=PolicyGuardrails(add=["base"]),
                statements=[
                    PolicyStatement(
                        sid="GPT4Only",
                        guardrails=["gpt4_guardrail"],
                        condition=PolicyCondition(
                            model=ConditionOperator(equals="gpt-4")
                        ),
                    ),
                    PolicyStatement(
                        sid="HealthcareOnly",
                        guardrails=["hipaa_audit"],
                        condition=PolicyCondition(
                            team=ConditionOperator(prefix="healthcare-")
                        ),
                    ),
                ],
            ),
        }

        # Healthcare team with GPT-4 should get all guardrails
        context = PolicyMatchContext(
            team_alias="healthcare-team", key_alias="k", model="gpt-4"
        )
        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="multi-statement",
            policies=policies,
            context=context,
        )

        assert "base" in resolved.guardrails
        assert "gpt4_guardrail" in resolved.guardrails
        assert "hipaa_audit" in resolved.guardrails

        # Finance team with GPT-4 should only get base + gpt4_guardrail
        context_finance = PolicyMatchContext(
            team_alias="finance-team", key_alias="k", model="gpt-4"
        )
        resolved_finance = PolicyResolver.resolve_policy_guardrails(
            policy_name="multi-statement",
            policies=policies,
            context=context_finance,
        )

        assert "base" in resolved_finance.guardrails
        assert "gpt4_guardrail" in resolved_finance.guardrails
        assert "hipaa_audit" not in resolved_finance.guardrails

    def test_statement_with_no_condition_always_applies(self):
        """Test statement with no condition always applies."""
        policies = {
            "always-policy": Policy(
                guardrails=PolicyGuardrails(add=["base"]),
                statements=[
                    PolicyStatement(
                        sid="AlwaysApply",
                        guardrails=["always_guardrail"],
                        condition=None,  # No condition = always applies
                    ),
                ],
            ),
        }

        context = PolicyMatchContext(team_alias="any", key_alias="any", model="any")
        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="always-policy",
            policies=policies,
            context=context,
        )

        assert "base" in resolved.guardrails
        assert "always_guardrail" in resolved.guardrails

    def test_inheritance_with_statements(self):
        """Test inheritance works with statements."""
        policies = {
            "base": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker"]),
            ),
            "child": Policy(
                inherit="base",
                guardrails=PolicyGuardrails(add=["child_guardrail"]),
                statements=[
                    PolicyStatement(
                        sid="ConditionalStatement",
                        guardrails=["conditional_guardrail"],
                        condition=PolicyCondition(
                            model=ConditionOperator(equals="gpt-4")
                        ),
                    ),
                ],
            ),
        }

        context = PolicyMatchContext(team_alias="any-team", key_alias="k", model="gpt-4")
        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="child",
            policies=policies,
            context=context,
        )

        # Should have: inherited pii_blocker, child's child_guardrail, and conditional_guardrail
        assert "pii_blocker" in resolved.guardrails
        assert "child_guardrail" in resolved.guardrails
        assert "conditional_guardrail" in resolved.guardrails

    def test_inheritance_with_remove_and_statements(self):
        """Test inheritance with remove still works alongside statements."""
        policies = {
            "base": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker", "phi_blocker"]),
            ),
            "child": Policy(
                inherit="base",
                guardrails=PolicyGuardrails(
                    add=["child_guardrail"],
                    remove=["phi_blocker"],  # Remove phi_blocker from parent
                ),
                statements=[
                    PolicyStatement(
                        sid="Conditional",
                        guardrails=["conditional_guardrail"],
                        condition=PolicyCondition(
                            model=ConditionOperator(equals="gpt-4")
                        ),
                    ),
                ],
            ),
        }

        context = PolicyMatchContext(team_alias="any-team", key_alias="k", model="gpt-4")
        resolved = PolicyResolver.resolve_policy_guardrails(
            policy_name="child",
            policies=policies,
            context=context,
        )

        assert "pii_blocker" in resolved.guardrails  # Inherited
        assert "phi_blocker" not in resolved.guardrails  # Removed
        assert "child_guardrail" in resolved.guardrails  # Added by child
        assert "conditional_guardrail" in resolved.guardrails  # From statement
