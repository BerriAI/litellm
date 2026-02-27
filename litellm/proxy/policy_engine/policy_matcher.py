"""
Policy Matcher - Matches requests against policy attachments.

Uses existing wildcard pattern matching helpers to determine which policies
apply to a given request based on team alias, key alias, and model.

Policies are matched via policy_attachments which define WHERE each policy applies.
"""

from typing import Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.types.proxy.policy_engine import Policy, PolicyMatchContext, PolicyScope


class PolicyMatcher:
    """
    Matches incoming requests against policy attachments.

    Supports wildcard patterns:
    - "*" matches everything
    - "prefix-*" matches anything starting with "prefix-"

    Uses policy_attachments to determine which policies apply to a request.
    """

    @staticmethod
    def matches_pattern(value: Optional[str], patterns: List[str]) -> bool:
        """
        Check if a value matches any of the given patterns.

        Uses the existing RouteChecks._route_matches_wildcard_pattern helper.

        Args:
            value: The value to check (e.g., team alias, key alias, model)
            patterns: List of patterns to match against

        Returns:
            True if value matches any pattern, False otherwise
        """
        # If no value provided, only match if patterns include "*"
        if value is None:
            return "*" in patterns

        for pattern in patterns:
            # Use existing wildcard pattern matching helper
            if RouteChecks._route_matches_wildcard_pattern(
                route=value, pattern=pattern
            ):
                return True

        return False

    @staticmethod
    def scope_matches(scope: PolicyScope, context: PolicyMatchContext) -> bool:
        """
        Check if a policy scope matches the given context.

        A scope matches if ALL of its fields match:
        - teams matches context.team_alias
        - keys matches context.key_alias
        - models matches context.model

        Args:
            scope: The policy scope to check
            context: The request context

        Returns:
            True if scope matches context, False otherwise
        """
        # Check teams
        if not PolicyMatcher.matches_pattern(context.team_alias, scope.get_teams()):
            return False

        # Check keys
        if not PolicyMatcher.matches_pattern(context.key_alias, scope.get_keys()):
            return False

        # Check models
        if not PolicyMatcher.matches_pattern(context.model, scope.get_models()):
            return False

        # Check tags (only if scope specifies tags)
        # Unlike teams/keys/models, empty tags means "do not check" rather than "match all"
        scope_tags = scope.get_tags()
        if scope_tags:
            if not context.tags:
                return False
            # Match if ANY context tag matches ANY scope tag pattern
            if not any(
                PolicyMatcher.matches_pattern(tag, scope_tags)
                for tag in context.tags
            ):
                return False

        return True

    @staticmethod
    def get_matching_policies(
        context: PolicyMatchContext,
    ) -> List[str]:
        """
        Get list of policy names that match the given context via attachments.

        Args:
            context: The request context to match against

        Returns:
            List of policy names that match the context
        """
        from litellm.proxy.policy_engine.attachment_registry import (
            get_attachment_registry,
        )

        registry = get_attachment_registry()
        if not registry.is_initialized():
            verbose_proxy_logger.debug(
                "AttachmentRegistry not initialized, returning empty list"
            )
            return []

        return registry.get_attached_policies(context)

    @staticmethod
    def get_matching_policies_from_registry(
        context: PolicyMatchContext,
    ) -> List[str]:
        """
        Get list of policy names that match the given context from the global registry.

        Args:
            context: The request context to match against

        Returns:
            List of policy names that match the context
        """
        return PolicyMatcher.get_matching_policies(context=context)

    @staticmethod
    def get_policies_with_matching_conditions(
        policy_names: List[str],
        context: PolicyMatchContext,
        policies: Optional[Dict[str, Policy]] = None,
    ) -> List[str]:
        """
        Filter policies to only those whose conditions match the context.

        A policy's condition matches if:
        - The policy has no condition (condition is None), OR
        - The policy's condition evaluates to True for the given context

        Args:
            policy_names: List of policy names to filter
            context: The request context to evaluate conditions against
            policies: Dictionary of all policies (if None, uses global registry)

        Returns:
            List of policy names whose conditions match the context
        """
        from litellm.proxy.policy_engine.condition_evaluator import ConditionEvaluator
        from litellm.proxy.policy_engine.policy_registry import get_policy_registry

        if policies is None:
            registry = get_policy_registry()
            if not registry.is_initialized():
                return []
            policies = registry.get_all_policies()

        matching_policies = []
        for policy_name in policy_names:
            policy = policies.get(policy_name)
            if policy is None:
                continue
            # Policy matches if it has no condition OR condition evaluates to True
            if policy.condition is None or ConditionEvaluator.evaluate(
                policy.condition, context
            ):
                matching_policies.append(policy_name)

        return matching_policies
