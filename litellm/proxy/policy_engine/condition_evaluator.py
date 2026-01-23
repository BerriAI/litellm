"""
Condition Evaluator - Evaluates policy conditions.

Supports model-based conditions with exact match or regex patterns.
"""

import re
from typing import List, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.policy_engine import (
    PolicyCondition,
    PolicyMatchContext,
)


class ConditionEvaluator:
    """
    Evaluates policy conditions against request context.

    Supports model conditions with:
    - Exact string match: "gpt-4"
    - Regex pattern: "gpt-4.*"
    - List of values: ["gpt-4", "gpt-4-turbo"]
    """

    @staticmethod
    def evaluate(
        condition: Optional[PolicyCondition],
        context: PolicyMatchContext,
    ) -> bool:
        """
        Evaluate a policy condition against a request context.

        Args:
            condition: The condition to evaluate (None = always matches)
            context: The request context with team, key, model

        Returns:
            True if condition matches, False otherwise
        """
        # No condition means always matches
        if condition is None:
            return True

        # Check model condition
        if condition.model is not None:
            if not ConditionEvaluator._evaluate_model_condition(
                condition=condition.model,
                model=context.model,
            ):
                verbose_proxy_logger.debug(
                    f"Condition failed: model={context.model} did not match {condition.model}"
                )
                return False

        return True

    @staticmethod
    def _evaluate_model_condition(
        condition: Union[str, List[str]],
        model: Optional[str],
    ) -> bool:
        """
        Evaluate a model condition.

        Args:
            condition: String (exact or regex) or list of strings
            model: The model name to check

        Returns:
            True if model matches condition, False otherwise
        """
        if model is None:
            return False

        # Handle list of values
        if isinstance(condition, list):
            return any(
                ConditionEvaluator._matches_pattern(pattern, model)
                for pattern in condition
            )

        # Single value - check as pattern
        return ConditionEvaluator._matches_pattern(condition, model)

    @staticmethod
    def _matches_pattern(pattern: str, value: str) -> bool:
        """
        Check if value matches pattern (exact match or regex).

        Args:
            pattern: Pattern to match (exact string or regex)
            value: Value to check

        Returns:
            True if matches, False otherwise
        """
        # First try exact match
        if pattern == value:
            return True

        # Try as regex pattern
        try:
            if re.fullmatch(pattern, value):
                return True
        except re.error:
            # Invalid regex, treat as literal string (already checked above)
            pass

        return False
