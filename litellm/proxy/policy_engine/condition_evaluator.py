"""
Condition Evaluator - Evaluates AWS IAM-style conditions.

Supports operators like equals, in, prefix, not_equals, not_in for
fine-grained policy statement matching.
"""

from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.policy_engine import (
    ConditionOperator,
    PolicyCondition,
    PolicyMatchContext,
)


class ConditionEvaluator:
    """
    Evaluates policy conditions against request context.

    Supports AWS IAM-style condition operators:
    - equals: Exact string match
    - in: Value must be in the list
    - prefix: Value must start with the prefix
    - not_equals: Value must NOT equal
    - not_in: Value must NOT be in the list

    All conditions in a PolicyCondition must match (AND logic).
    """

    @staticmethod
    def evaluate(
        condition: Optional[PolicyCondition],
        context: PolicyMatchContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate a policy condition against a request context.

        Args:
            condition: The condition to evaluate (None = always matches)
            context: The request context with team, key, model
            metadata: Optional request metadata for metadata conditions

        Returns:
            True if condition matches, False otherwise
        """
        # No condition means always matches
        if condition is None:
            return True

        # Check model condition
        if condition.model is not None:
            if not ConditionEvaluator.evaluate_operator(
                operator=condition.model,
                value=context.model,
            ):
                verbose_proxy_logger.debug(
                    f"Condition failed: model={context.model} did not match {condition.model}"
                )
                return False

        # Check team condition
        if condition.team is not None:
            if not ConditionEvaluator.evaluate_operator(
                operator=condition.team,
                value=context.team_alias,
            ):
                verbose_proxy_logger.debug(
                    f"Condition failed: team={context.team_alias} did not match {condition.team}"
                )
                return False

        # Check key condition
        if condition.key is not None:
            if not ConditionEvaluator.evaluate_operator(
                operator=condition.key,
                value=context.key_alias,
            ):
                verbose_proxy_logger.debug(
                    f"Condition failed: key={context.key_alias} did not match {condition.key}"
                )
                return False

        # Check metadata conditions
        if condition.metadata is not None and metadata is not None:
            for field_name, field_operator in condition.metadata.items():
                field_value = metadata.get(field_name)
                # Convert to string for comparison
                field_value_str = str(field_value) if field_value is not None else None
                if not ConditionEvaluator.evaluate_operator(
                    operator=field_operator,
                    value=field_value_str,
                ):
                    verbose_proxy_logger.debug(
                        f"Condition failed: metadata.{field_name}={field_value} "
                        f"did not match {field_operator}"
                    )
                    return False

        return True

    @staticmethod
    def evaluate_operator(
        operator: ConditionOperator,
        value: Optional[str],
    ) -> bool:
        """
        Evaluate a single condition operator against a value.

        Args:
            operator: The condition operator to evaluate
            value: The value to check (None if not provided)

        Returns:
            True if the value matches the operator, False otherwise
        """
        # Handle None value
        if value is None:
            # For positive operators (equals, in, prefix), None never matches
            if operator.equals is not None:
                return False
            if operator.in_ is not None:
                return False
            if operator.prefix is not None:
                return False
            # For negative operators (not_equals, not_in), None always matches
            # (None is not equal to anything, and not in any list)
            if operator.not_equals is not None:
                return True
            if operator.not_in is not None:
                return True
            # No operators specified = matches
            return True

        # Check equals
        if operator.equals is not None:
            if value != operator.equals:
                return False

        # Check in (value must be in list)
        if operator.in_ is not None:
            if value not in operator.in_:
                return False

        # Check prefix
        if operator.prefix is not None:
            if not value.startswith(operator.prefix):
                return False

        # Check not_equals
        if operator.not_equals is not None:
            if value == operator.not_equals:
                return False

        # Check not_in
        if operator.not_in is not None:
            if value in operator.not_in:
                return False

        return True

    @staticmethod
    def evaluate_all_conditions(
        conditions: list,
        context: PolicyMatchContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate multiple conditions (AND logic - all must match).

        Args:
            conditions: List of PolicyCondition objects
            context: The request context
            metadata: Optional request metadata

        Returns:
            True if ALL conditions match, False otherwise
        """
        for condition in conditions:
            if not ConditionEvaluator.evaluate(
                condition=condition,
                context=context,
                metadata=metadata,
            ):
                return False
        return True

    @staticmethod
    def evaluate_any_condition(
        conditions: list,
        context: PolicyMatchContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate multiple conditions (OR logic - any must match).

        Args:
            conditions: List of PolicyCondition objects
            context: The request context
            metadata: Optional request metadata

        Returns:
            True if ANY condition matches, False otherwise
        """
        if not conditions:
            return True

        for condition in conditions:
            if ConditionEvaluator.evaluate(
                condition=condition,
                context=context,
                metadata=metadata,
            ):
                return True
        return False
