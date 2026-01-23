"""
Unit tests for ConditionEvaluator - tests model condition evaluation.

Tests:
- Exact model match
- Regex pattern match
- List of models
"""

import pytest

from litellm.proxy.policy_engine.condition_evaluator import ConditionEvaluator
from litellm.types.proxy.policy_engine import (
    PolicyCondition,
    PolicyMatchContext,
)


class TestConditionEvaluator:
    """Test condition evaluation."""

    def test_no_condition_always_matches(self):
        """Test that None condition always matches."""
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-4")
        assert ConditionEvaluator.evaluate(None, context) is True

    def test_exact_model_match(self):
        """Test exact model string match."""
        condition = PolicyCondition(model="gpt-4")
        
        # Match
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-4")
        assert ConditionEvaluator.evaluate(condition, context) is True
        
        # No match
        context_other = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-3.5")
        assert ConditionEvaluator.evaluate(condition, context_other) is False

    def test_regex_pattern_match(self):
        """Test regex pattern matching."""
        condition = PolicyCondition(model="gpt-4.*")
        
        # Matches
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-4")
        ) is True
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-4-turbo")
        ) is True
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-4o")
        ) is True
        
        # No match
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-3.5")
        ) is False

    def test_list_of_models_match(self):
        """Test list of model values."""
        condition = PolicyCondition(model=["gpt-4", "gpt-4-turbo", "claude-3"])
        
        # Matches
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-4")
        ) is True
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="claude-3")
        ) is True
        
        # No match
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-3.5")
        ) is False

    def test_list_with_regex_patterns(self):
        """Test list can contain regex patterns."""
        condition = PolicyCondition(model=["gpt-4.*", "claude-.*"])
        
        # Matches
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="gpt-4-turbo")
        ) is True
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="claude-3")
        ) is True
        
        # No match
        assert ConditionEvaluator.evaluate(
            condition,
            PolicyMatchContext(team_alias="t", key_alias="k", model="llama-2")
        ) is False

    def test_none_model_does_not_match(self):
        """Test that None model value doesn't match conditions."""
        condition = PolicyCondition(model="gpt-4")
        context = PolicyMatchContext(team_alias="t", key_alias="k", model=None)
        assert ConditionEvaluator.evaluate(condition, context) is False

    def test_empty_condition_always_matches(self):
        """Test condition with no model field always matches."""
        condition = PolicyCondition()  # No model specified
        context = PolicyMatchContext(team_alias="t", key_alias="k", model="any-model")
        assert ConditionEvaluator.evaluate(condition, context) is True
