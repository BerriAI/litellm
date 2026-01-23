"""
Unit tests for ConditionEvaluator - tests AWS IAM-style condition evaluation.

Tests:
- Condition operators (equals, in, prefix, not_equals, not_in)
- PolicyCondition evaluation against request context
- Metadata condition evaluation
"""

import pytest

from litellm.proxy.policy_engine.condition_evaluator import ConditionEvaluator
from litellm.types.proxy.policy_engine import (
    ConditionOperator,
    PolicyCondition,
    PolicyMatchContext,
)


class TestConditionOperatorEvaluation:
    """Test individual condition operator evaluation."""

    def test_equals_operator_match(self):
        """Test equals operator matches exact value."""
        operator = ConditionOperator(equals="gpt-4")
        assert ConditionEvaluator.evaluate_operator(operator, "gpt-4") is True

    def test_equals_operator_no_match(self):
        """Test equals operator does not match different value."""
        operator = ConditionOperator(equals="gpt-4")
        assert ConditionEvaluator.evaluate_operator(operator, "gpt-3.5") is False

    def test_in_operator_match(self):
        """Test in operator matches value in list."""
        operator = ConditionOperator(in_=["gpt-4", "gpt-4-turbo", "gpt-4o"])
        assert ConditionEvaluator.evaluate_operator(operator, "gpt-4") is True
        assert ConditionEvaluator.evaluate_operator(operator, "gpt-4-turbo") is True

    def test_in_operator_no_match(self):
        """Test in operator does not match value not in list."""
        operator = ConditionOperator(in_=["gpt-4", "gpt-4-turbo"])
        assert ConditionEvaluator.evaluate_operator(operator, "gpt-3.5") is False

    def test_prefix_operator_match(self):
        """Test prefix operator matches value starting with prefix."""
        operator = ConditionOperator(prefix="bedrock/")
        assert ConditionEvaluator.evaluate_operator(operator, "bedrock/claude-3") is True
        assert ConditionEvaluator.evaluate_operator(operator, "bedrock/llama") is True

    def test_prefix_operator_no_match(self):
        """Test prefix operator does not match value not starting with prefix."""
        operator = ConditionOperator(prefix="bedrock/")
        assert ConditionEvaluator.evaluate_operator(operator, "openai/gpt-4") is False

    def test_not_equals_operator_match(self):
        """Test not_equals operator matches when value is different."""
        operator = ConditionOperator(not_equals="gpt-3.5")
        assert ConditionEvaluator.evaluate_operator(operator, "gpt-4") is True

    def test_not_equals_operator_no_match(self):
        """Test not_equals operator does not match when value is same."""
        operator = ConditionOperator(not_equals="gpt-4")
        assert ConditionEvaluator.evaluate_operator(operator, "gpt-4") is False

    def test_not_in_operator_match(self):
        """Test not_in operator matches when value not in list."""
        operator = ConditionOperator(not_in=["gpt-3.5", "gpt-3.5-turbo"])
        assert ConditionEvaluator.evaluate_operator(operator, "gpt-4") is True

    def test_not_in_operator_no_match(self):
        """Test not_in operator does not match when value in list."""
        operator = ConditionOperator(not_in=["gpt-4", "gpt-4-turbo"])
        assert ConditionEvaluator.evaluate_operator(operator, "gpt-4") is False

    def test_none_value_with_positive_operators(self):
        """Test None value does not match positive operators."""
        assert ConditionEvaluator.evaluate_operator(
            ConditionOperator(equals="gpt-4"), None
        ) is False
        assert ConditionEvaluator.evaluate_operator(
            ConditionOperator(in_=["gpt-4"]), None
        ) is False
        assert ConditionEvaluator.evaluate_operator(
            ConditionOperator(prefix="gpt"), None
        ) is False

    def test_none_value_with_negative_operators(self):
        """Test None value matches negative operators (None is not equal to anything)."""
        assert ConditionEvaluator.evaluate_operator(
            ConditionOperator(not_equals="gpt-4"), None
        ) is True
        assert ConditionEvaluator.evaluate_operator(
            ConditionOperator(not_in=["gpt-4"]), None
        ) is True

    def test_empty_operator_matches_any(self):
        """Test empty operator (no conditions) matches any value."""
        operator = ConditionOperator()
        assert ConditionEvaluator.evaluate_operator(operator, "anything") is True
        assert ConditionEvaluator.evaluate_operator(operator, None) is True


class TestPolicyConditionEvaluation:
    """Test PolicyCondition evaluation against request context."""

    def test_model_condition_match(self):
        """Test model condition matches."""
        condition = PolicyCondition(
            model=ConditionOperator(in_=["gpt-4", "gpt-4-turbo"])
        )
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-4")
        assert ConditionEvaluator.evaluate(condition, context) is True

    def test_model_condition_no_match(self):
        """Test model condition does not match."""
        condition = PolicyCondition(
            model=ConditionOperator(in_=["gpt-4", "gpt-4-turbo"])
        )
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-3.5")
        assert ConditionEvaluator.evaluate(condition, context) is False

    def test_team_condition_match(self):
        """Test team condition matches."""
        condition = PolicyCondition(
            team=ConditionOperator(prefix="healthcare-")
        )
        context = PolicyMatchContext(
            team_alias="healthcare-research", key_alias="key", model="gpt-4"
        )
        assert ConditionEvaluator.evaluate(condition, context) is True

    def test_team_condition_no_match(self):
        """Test team condition does not match."""
        condition = PolicyCondition(
            team=ConditionOperator(prefix="healthcare-")
        )
        context = PolicyMatchContext(
            team_alias="finance-team", key_alias="key", model="gpt-4"
        )
        assert ConditionEvaluator.evaluate(condition, context) is False

    def test_key_condition_match(self):
        """Test key condition matches."""
        condition = PolicyCondition(
            key=ConditionOperator(equals="production-key")
        )
        context = PolicyMatchContext(
            team_alias="team", key_alias="production-key", model="gpt-4"
        )
        assert ConditionEvaluator.evaluate(condition, context) is True

    def test_multiple_conditions_all_match(self):
        """Test multiple conditions all must match (AND logic)."""
        condition = PolicyCondition(
            model=ConditionOperator(in_=["gpt-4", "gpt-4-turbo"]),
            team=ConditionOperator(prefix="healthcare-"),
        )
        context = PolicyMatchContext(
            team_alias="healthcare-research", key_alias="key", model="gpt-4"
        )
        assert ConditionEvaluator.evaluate(condition, context) is True

    def test_multiple_conditions_one_fails(self):
        """Test multiple conditions - if one fails, all fails."""
        condition = PolicyCondition(
            model=ConditionOperator(in_=["gpt-4", "gpt-4-turbo"]),
            team=ConditionOperator(prefix="healthcare-"),
        )
        # Model matches but team doesn't
        context = PolicyMatchContext(
            team_alias="finance-team", key_alias="key", model="gpt-4"
        )
        assert ConditionEvaluator.evaluate(condition, context) is False

    def test_none_condition_always_matches(self):
        """Test None condition always matches."""
        context = PolicyMatchContext(team_alias="any", key_alias="any", model="any")
        assert ConditionEvaluator.evaluate(None, context) is True


class TestMetadataConditionEvaluation:
    """Test metadata condition evaluation."""

    def test_metadata_condition_match(self):
        """Test metadata condition matches."""
        condition = PolicyCondition(
            metadata={
                "environment": ConditionOperator(equals="production"),
            }
        )
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-4")
        metadata = {"environment": "production"}
        assert ConditionEvaluator.evaluate(condition, context, metadata) is True

    def test_metadata_condition_no_match(self):
        """Test metadata condition does not match."""
        condition = PolicyCondition(
            metadata={
                "environment": ConditionOperator(equals="production"),
            }
        )
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-4")
        metadata = {"environment": "staging"}
        assert ConditionEvaluator.evaluate(condition, context, metadata) is False

    def test_metadata_condition_missing_field(self):
        """Test metadata condition with missing field does not match."""
        condition = PolicyCondition(
            metadata={
                "environment": ConditionOperator(equals="production"),
            }
        )
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-4")
        metadata = {"other_field": "value"}
        assert ConditionEvaluator.evaluate(condition, context, metadata) is False

    def test_metadata_condition_with_model_condition(self):
        """Test combining metadata and model conditions."""
        condition = PolicyCondition(
            model=ConditionOperator(in_=["gpt-4"]),
            metadata={
                "environment": ConditionOperator(equals="production"),
            },
        )
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-4")
        metadata = {"environment": "production"}
        assert ConditionEvaluator.evaluate(condition, context, metadata) is True

        # Model matches but metadata doesn't
        metadata_staging = {"environment": "staging"}
        assert ConditionEvaluator.evaluate(condition, context, metadata_staging) is False


class TestEvaluateAllConditions:
    """Test evaluate_all_conditions helper."""

    def test_all_conditions_match(self):
        """Test all conditions match returns True."""
        conditions = [
            PolicyCondition(model=ConditionOperator(equals="gpt-4")),
            PolicyCondition(team=ConditionOperator(prefix="healthcare-")),
        ]
        context = PolicyMatchContext(
            team_alias="healthcare-team", key_alias="key", model="gpt-4"
        )
        assert ConditionEvaluator.evaluate_all_conditions(conditions, context) is True

    def test_one_condition_fails(self):
        """Test one condition fails returns False."""
        conditions = [
            PolicyCondition(model=ConditionOperator(equals="gpt-4")),
            PolicyCondition(team=ConditionOperator(prefix="healthcare-")),
        ]
        context = PolicyMatchContext(
            team_alias="finance-team", key_alias="key", model="gpt-4"
        )
        assert ConditionEvaluator.evaluate_all_conditions(conditions, context) is False

    def test_empty_conditions_returns_true(self):
        """Test empty conditions list returns True."""
        context = PolicyMatchContext(team_alias="any", key_alias="any", model="any")
        assert ConditionEvaluator.evaluate_all_conditions([], context) is True


class TestEvaluateAnyCondition:
    """Test evaluate_any_condition helper."""

    def test_any_condition_matches(self):
        """Test any condition matches returns True."""
        conditions = [
            PolicyCondition(model=ConditionOperator(equals="gpt-4")),
            PolicyCondition(model=ConditionOperator(equals="gpt-3.5")),
        ]
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="gpt-4")
        assert ConditionEvaluator.evaluate_any_condition(conditions, context) is True

    def test_no_condition_matches(self):
        """Test no condition matches returns False."""
        conditions = [
            PolicyCondition(model=ConditionOperator(equals="gpt-4")),
            PolicyCondition(model=ConditionOperator(equals="gpt-3.5")),
        ]
        context = PolicyMatchContext(team_alias="team", key_alias="key", model="claude-3")
        assert ConditionEvaluator.evaluate_any_condition(conditions, context) is False

    def test_empty_conditions_returns_true(self):
        """Test empty conditions list returns True."""
        context = PolicyMatchContext(team_alias="any", key_alias="any", model="any")
        assert ConditionEvaluator.evaluate_any_condition([], context) is True
