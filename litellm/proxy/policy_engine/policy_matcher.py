"""
Policy Matcher - Matches requests against policy scopes.

Uses existing wildcard pattern matching helpers to determine which policies
apply to a given request based on team alias, key alias, and model.
"""

from typing import Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.types.proxy.policy_engine import Policy, PolicyMatchContext, PolicyScope


class PolicyMatcher:
    """
    Matches incoming requests against policy scopes.

    Supports wildcard patterns:
    - "*" matches everything
    - "prefix-*" matches anything starting with "prefix-"
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

        return True

    @staticmethod
    def get_matching_policies(
        policies: Dict[str, Policy],
        context: PolicyMatchContext,
    ) -> List[str]:
        """
        Get list of policy names that match the given context.

        Args:
            policies: Dictionary of all policies
            context: The request context to match against

        Returns:
            List of policy names that match the context
        """
        matching: List[str] = []

        for policy_name, policy in policies.items():
            if PolicyMatcher.scope_matches(scope=policy.scope, context=context):
                matching.append(policy_name)
                verbose_proxy_logger.debug(
                    f"Policy '{policy_name}' matches context: "
                    f"team_alias={context.team_alias}, "
                    f"key_alias={context.key_alias}, "
                    f"model={context.model}"
                )

        return matching

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
        from litellm.proxy.policy_engine.policy_registry import get_policy_registry

        registry = get_policy_registry()
        if not registry.is_initialized():
            return []

        return PolicyMatcher.get_matching_policies(
            policies=registry.get_all_policies(),
            context=context,
        )
