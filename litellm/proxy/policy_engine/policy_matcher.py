"""
Policy Matcher - Matches requests against policy attachments.

Uses existing wildcard pattern matching helpers to determine which policies
apply to a given request based on team alias, key alias, and model.

Policies are matched via policy_attachments which define WHERE each policy applies.
"""

from collections.abc import Callable, Sequence
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
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
    def matches_pattern(value: str | None, patterns: list[str]) -> bool:
        """
        Check if a value matches any of the given patterns.

        Uses the existing RouteChecks.route_matches_wildcard_pattern helper.

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
            if RouteChecks.route_matches_wildcard_pattern(route=value, pattern=pattern):
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
        scope_tags: Final = scope.get_tags()
        if scope_tags:
            if not context.tags:
                return False
            # Match if ANY context tag matches ANY scope tag pattern
            if not any(PolicyMatcher.matches_pattern(tag, scope_tags) for tag in context.tags):
                return False

        return True

    @staticmethod
    def get_matching_policies(
        context: PolicyMatchContext,
    ) -> list[str]:
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

        registry: Final = get_attachment_registry()
        if not registry.is_initialized():
            verbose_proxy_logger.debug("AttachmentRegistry not initialized, returning empty list")
            return []

        return registry.get_attached_policies(context, PolicyMatcher.policy_applies(context))

    @staticmethod
    def get_matching_policies_from_registry(
        context: PolicyMatchContext,
    ) -> list[str]:
        """
        Get list of policy names that match the given context from the global registry.

        Args:
            context: The request context to match against

        Returns:
            List of policy names that match the context
        """
        return PolicyMatcher.get_matching_policies(context=context)

    @staticmethod
    def policy_applies(
        context: PolicyMatchContext,
        policies: dict[str, Policy] | None = None,
    ) -> Callable[[str], bool]:
        """
        Predicate telling whether a policy exists and any policy in its
        inheritance chain applies to the context. Admissions where the
        policy's own condition missed but an ancestor applies are logged at
        INFO, once per attachment scan.
        """
        resolved: Final = policies if policies is not None else PolicyMatcher._registry_policies()

        def applies(policy_name: str) -> bool:
            applying: Final = PolicyMatcher._applying_chain_members(
                policy_name=policy_name, context=context, policies=resolved
            )
            if applying and policy_name not in applying:
                verbose_proxy_logger.info(
                    "Policy '%s' applied through ancestor '%s' although its own condition did not match "
                    "(team_alias=%s, key_alias=%s, model=%s)",
                    policy_name,
                    applying[0],
                    context.team_alias,
                    context.key_alias,
                    context.model,
                )
            return bool(applying)

        return applies

    @staticmethod
    def _applying_chain_members(
        policy_name: str,
        context: PolicyMatchContext,
        policies: dict[str, Policy],
    ) -> tuple[str, ...]:
        from litellm.proxy.policy_engine.condition_evaluator import ConditionEvaluator

        chain: Final = PolicyResolver.resolve_inheritance_chain(policy_name=policy_name, policies=policies)
        return tuple(
            name
            for name in chain
            if (policy := policies.get(name)) is not None
            and (policy.condition is None or ConditionEvaluator.evaluate(policy.condition, context))
        )

    @staticmethod
    def _registry_policies() -> dict[str, Policy]:
        from litellm.proxy.policy_engine.policy_registry import get_policy_registry

        registry: Final = get_policy_registry()
        return registry.get_all_policies() if registry.is_initialized() else {}

    @staticmethod
    def get_policies_with_matching_conditions(
        policy_names: Sequence[str],
        context: PolicyMatchContext,
        policies: dict[str, Policy] | None = None,
    ) -> list[str]:
        """
        Filter policies to only those that apply to the given context.

        A policy applies when any policy in its inheritance chain has no
        condition or a condition that evaluates to True for the context. The
        resolver then drops only the chain members whose own condition fails,
        so a child whose condition misses still contributes the guardrails of
        its unconditional ancestors. A missing policy resolves to an empty
        chain and does not apply.

        Args:
            policy_names: List of policy names to filter
            context: The request context to evaluate conditions against
            policies: Dictionary of all policies (if None, uses global registry)

        Returns:
            List of policy names that apply to the context
        """
        resolved: Final = policies if policies is not None else PolicyMatcher._registry_policies()
        return [
            policy_name
            for policy_name in policy_names
            if PolicyMatcher._applying_chain_members(policy_name, context, resolved)
        ]
