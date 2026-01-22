"""
Unit tests for PolicyMatcher - tests wildcard pattern matching for policies.

Tests:
- Wildcard matching (*, prefix-*)
- Scope matching (teams, keys, models)
"""

import pytest

from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
from litellm.types.proxy.policy_engine import (
    Policy,
    PolicyGuardrails,
    PolicyMatchContext,
    PolicyScope,
)


class TestPolicyMatcherGetMatchingPolicies:
    """Test getting matching policies from a set of policies."""

    def test_get_matching_policies_by_team(self):
        """Test matching policies by team alias."""
        policies = {
            "healthcare": Policy(
                guardrails=PolicyGuardrails(add=["hipaa_audit"]),
                scope=PolicyScope(teams=["healthcare-team"]),
            ),
        }

        # Match
        context = PolicyMatchContext(team_alias="healthcare-team", key_alias="k", model="gpt-4")
        assert "healthcare" in PolicyMatcher.get_matching_policies(policies=policies, context=context)

        # No match
        context = PolicyMatchContext(team_alias="finance-team", key_alias="k", model="gpt-4")
        assert len(PolicyMatcher.get_matching_policies(policies=policies, context=context)) == 0

    def test_get_matching_policies_by_model_wildcard(self):
        """Test matching policies by model with wildcard pattern."""
        policies = {
            "bedrock-only": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker"]),
                scope=PolicyScope(models=["bedrock/*"]),
            ),
        }

        # Match - bedrock model
        context = PolicyMatchContext(team_alias="t", key_alias="k", model="bedrock/claude-3")
        assert "bedrock-only" in PolicyMatcher.get_matching_policies(policies=policies, context=context)

        # No match - different provider
        context = PolicyMatchContext(team_alias="t", key_alias="k", model="openai/gpt-4")
        assert len(PolicyMatcher.get_matching_policies(policies=policies, context=context)) == 0

    def test_get_matching_policies_by_key_pattern(self):
        """Test matching policies by key alias pattern."""
        policies = {
            "dev-keys": Policy(
                guardrails=PolicyGuardrails(add=["toxicity_filter"]),
                scope=PolicyScope(keys=["dev-key-*"]),
            ),
        }

        # Match
        context = PolicyMatchContext(team_alias="t", key_alias="dev-key-123", model="gpt-4")
        assert "dev-keys" in PolicyMatcher.get_matching_policies(policies=policies, context=context)

        # No match
        context = PolicyMatchContext(team_alias="t", key_alias="prod-key-123", model="gpt-4")
        assert len(PolicyMatcher.get_matching_policies(policies=policies, context=context)) == 0

    def test_get_matching_policies_global_wildcard(self):
        """Test global policy with '*' matches everything."""
        policies = {
            "global": Policy(
                guardrails=PolicyGuardrails(add=["pii_blocker"]),
                scope=PolicyScope(teams=["*"], keys=["*"], models=["*"]),
            ),
        }

        context = PolicyMatchContext(team_alias="any-team", key_alias="any-key", model="any-model")
        assert "global" in PolicyMatcher.get_matching_policies(policies=policies, context=context)
