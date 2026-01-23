"""
Policy Resolver - Resolves final guardrail list from policies.

Handles:
- Inheritance chain resolution (inherit with add/remove)
- Applying add/remove guardrails
- Evaluating conditional statements
- Combining guardrails from multiple matching policies
"""

from typing import Any, Dict, List, Optional, Set

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.policy_engine import (
    Policy,
    PolicyMatchContext,
    ResolvedPolicy,
)


class PolicyResolver:
    """
    Resolves the final list of guardrails from policies.

    Handles:
    - Inheritance chains with add/remove operations
    - Conditional statements with AWS IAM-style conditions
    """

    @staticmethod
    def resolve_inheritance_chain(
        policy_name: str,
        policies: Dict[str, Policy],
        visited: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Get the inheritance chain for a policy (from root to policy).

        Args:
            policy_name: Name of the policy
            policies: Dictionary of all policies
            visited: Set of visited policies (for cycle detection)

        Returns:
            List of policy names from root ancestor to the given policy
        """
        if visited is None:
            visited = set()

        if policy_name in visited:
            verbose_proxy_logger.warning(
                f"Circular inheritance detected for policy '{policy_name}'"
            )
            return []

        policy = policies.get(policy_name)
        if policy is None:
            return []

        visited.add(policy_name)

        if policy.inherit:
            parent_chain = PolicyResolver.resolve_inheritance_chain(
                policy_name=policy.inherit, policies=policies, visited=visited
            )
            return parent_chain + [policy_name]

        return [policy_name]

    @staticmethod
    def resolve_policy_guardrails(
        policy_name: str,
        policies: Dict[str, Policy],
        context: Optional[PolicyMatchContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ResolvedPolicy:
        """
        Resolve the final guardrails for a single policy, including inheritance.

        This method:
        1. Resolves the inheritance chain
        2. Applies add/remove from each policy in the chain
        3. Evaluates conditional statements (if context provided)

        Args:
            policy_name: Name of the policy to resolve
            policies: Dictionary of all policies
            context: Optional request context for evaluating statement conditions
            metadata: Optional request metadata for condition evaluation

        Returns:
            ResolvedPolicy with final guardrails list
        """
        inheritance_chain = PolicyResolver.resolve_inheritance_chain(
            policy_name=policy_name, policies=policies
        )

        # Start with empty set of guardrails
        guardrails: Set[str] = set()

        # Apply each policy in the chain (from root to leaf)
        for chain_policy_name in inheritance_chain:
            policy = policies.get(chain_policy_name)
            if policy is None:
                continue

            # Add guardrails from guardrails.add
            for guardrail in policy.guardrails.get_add():
                guardrails.add(guardrail)

            # Remove guardrails from guardrails.remove
            for guardrail in policy.guardrails.get_remove():
                guardrails.discard(guardrail)

            # Evaluate statements (if context provided)
            if context is not None and policy.statements:
                statement_guardrails = PolicyResolver._evaluate_statements(
                    statements=policy.statements,
                    context=context,
                    metadata=metadata,
                )
                guardrails.update(statement_guardrails)
                if statement_guardrails:
                    verbose_proxy_logger.debug(
                        f"Policy '{chain_policy_name}' statements contributed: {statement_guardrails}"
                    )

        return ResolvedPolicy(
            policy_name=policy_name,
            guardrails=list(guardrails),
            inheritance_chain=inheritance_chain,
        )

    @staticmethod
    def _evaluate_statements(
        statements: list,
        context: PolicyMatchContext,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Set[str]:
        """
        Evaluate policy statements and return guardrails from matching statements.

        Args:
            statements: List of PolicyStatement objects
            context: The request context
            metadata: Optional request metadata

        Returns:
            Set of guardrail names from matching statements
        """
        from litellm.proxy.policy_engine.condition_evaluator import ConditionEvaluator

        guardrails: Set[str] = set()

        for statement in statements:
            # Evaluate the statement's condition
            if ConditionEvaluator.evaluate(
                condition=statement.condition,
                context=context,
                metadata=metadata,
            ):
                guardrails.update(statement.guardrails)
                verbose_proxy_logger.debug(
                    f"Statement '{statement.sid or 'unnamed'}' matched, "
                    f"adding guardrails: {statement.guardrails}"
                )

        return guardrails

    @staticmethod
    def resolve_guardrails_for_context(
        context: PolicyMatchContext,
        policies: Optional[Dict[str, Policy]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Resolve the final list of guardrails for a request context.

        This:
        1. Finds all policies that match the context via policy_attachments
        2. Resolves each policy's guardrails (including inheritance)
        3. Evaluates conditional statements
        4. Combines all guardrails (union)

        Args:
            context: The request context
            policies: Dictionary of all policies (if None, uses global registry)
            metadata: Optional request metadata for condition evaluation

        Returns:
            List of guardrail names to apply
        """
        from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
        from litellm.proxy.policy_engine.policy_registry import get_policy_registry

        if policies is None:
            registry = get_policy_registry()
            if not registry.is_initialized():
                return []
            policies = registry.get_all_policies()

        # Get matching policies via attachments
        matching_policy_names = PolicyMatcher.get_matching_policies(context=context)

        if not matching_policy_names:
            verbose_proxy_logger.debug(
                f"No policies match context: team_alias={context.team_alias}, "
                f"key_alias={context.key_alias}, model={context.model}"
            )
            return []

        # Resolve each matching policy and combine guardrails
        all_guardrails: Set[str] = set()

        for policy_name in matching_policy_names:
            resolved = PolicyResolver.resolve_policy_guardrails(
                policy_name=policy_name,
                policies=policies,
                context=context,
                metadata=metadata,
            )
            all_guardrails.update(resolved.guardrails)
            verbose_proxy_logger.debug(
                f"Policy '{policy_name}' contributes guardrails: {resolved.guardrails}"
            )

        result = list(all_guardrails)
        verbose_proxy_logger.debug(
            f"Final guardrails for context: {result}"
        )

        return result

    @staticmethod
    def get_all_resolved_policies(
        policies: Optional[Dict[str, Policy]] = None,
        context: Optional[PolicyMatchContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ResolvedPolicy]:
        """
        Resolve all policies and return their final guardrails.

        Useful for debugging and displaying policy configurations.

        Args:
            policies: Dictionary of all policies (if None, uses global registry)
            context: Optional context for evaluating statement conditions
            metadata: Optional metadata for condition evaluation

        Returns:
            Dictionary mapping policy names to ResolvedPolicy objects
        """
        from litellm.proxy.policy_engine.policy_registry import get_policy_registry

        if policies is None:
            registry = get_policy_registry()
            if not registry.is_initialized():
                return {}
            policies = registry.get_all_policies()

        resolved: Dict[str, ResolvedPolicy] = {}

        for policy_name in policies:
            resolved[policy_name] = PolicyResolver.resolve_policy_guardrails(
                policy_name=policy_name,
                policies=policies,
                context=context,
                metadata=metadata,
            )

        return resolved
