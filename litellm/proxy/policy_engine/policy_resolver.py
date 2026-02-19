"""
Policy Resolver - Resolves final guardrail list from policies.

Handles:
- Inheritance chain resolution (inherit with add/remove)
- Applying add/remove guardrails
- Evaluating model conditions
- Combining guardrails from multiple matching policies
"""

from typing import Dict, List, Optional, Set, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.policy_engine import (
    GuardrailPipeline,
    Policy,
    PolicyMatchContext,
    ResolvedPolicy,
)


class PolicyResolver:
    """
    Resolves the final list of guardrails from policies.

    Handles:
    - Inheritance chains with add/remove operations
    - Model-based conditions
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
    ) -> ResolvedPolicy:
        """
        Resolve the final guardrails for a single policy, including inheritance.

        This method:
        1. Resolves the inheritance chain
        2. Applies add/remove from each policy in the chain
        3. Evaluates model conditions (if context provided)

        Args:
            policy_name: Name of the policy to resolve
            policies: Dictionary of all policies
            context: Optional request context for evaluating conditions

        Returns:
            ResolvedPolicy with final guardrails list
        """
        from litellm.proxy.policy_engine.condition_evaluator import ConditionEvaluator

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

            # Check if policy condition matches (if context provided)
            if context is not None and policy.condition is not None:
                if not ConditionEvaluator.evaluate(
                    condition=policy.condition,
                    context=context,
                ):
                    verbose_proxy_logger.debug(
                        f"Policy '{chain_policy_name}' condition did not match, skipping guardrails"
                    )
                    continue

            # Add guardrails from guardrails.add
            for guardrail in policy.guardrails.get_add():
                guardrails.add(guardrail)

            # Remove guardrails from guardrails.remove
            for guardrail in policy.guardrails.get_remove():
                guardrails.discard(guardrail)

        return ResolvedPolicy(
            policy_name=policy_name,
            guardrails=list(guardrails),
            inheritance_chain=inheritance_chain,
        )

    @staticmethod
    def resolve_guardrails_for_context(
        context: PolicyMatchContext,
        policies: Optional[Dict[str, Policy]] = None,
    ) -> List[str]:
        """
        Resolve the final list of guardrails for a request context.

        This:
        1. Finds all policies that match the context via policy_attachments
        2. Resolves each policy's guardrails (including inheritance)
        3. Evaluates model conditions
        4. Combines all guardrails (union)

        Args:
            context: The request context
            policies: Dictionary of all policies (if None, uses global registry)

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
    def resolve_pipelines_for_context(
        context: PolicyMatchContext,
        policies: Optional[Dict[str, Policy]] = None,
    ) -> List[Tuple[str, GuardrailPipeline]]:
        """
        Resolve pipelines from matching policies for a request context.

        Returns (policy_name, pipeline) tuples for policies that have pipelines.
        Guardrails managed by pipelines should be excluded from the flat
        guardrails list to avoid double execution.

        Args:
            context: The request context
            policies: Dictionary of all policies (if None, uses global registry)

        Returns:
            List of (policy_name, GuardrailPipeline) tuples
        """
        from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
        from litellm.proxy.policy_engine.policy_registry import get_policy_registry

        if policies is None:
            registry = get_policy_registry()
            if not registry.is_initialized():
                return []
            policies = registry.get_all_policies()

        matching_policy_names = PolicyMatcher.get_matching_policies(context=context)
        if not matching_policy_names:
            return []

        pipelines: List[Tuple[str, GuardrailPipeline]] = []
        for policy_name in matching_policy_names:
            policy = policies.get(policy_name)
            if policy is None:
                continue
            if policy.pipeline is not None:
                pipelines.append((policy_name, policy.pipeline))
                verbose_proxy_logger.debug(
                    f"Policy '{policy_name}' has pipeline with "
                    f"{len(policy.pipeline.steps)} steps"
                )

        return pipelines

    @staticmethod
    def get_pipeline_managed_guardrails(
        pipelines: List[Tuple[str, GuardrailPipeline]],
    ) -> Set[str]:
        """
        Get the set of guardrail names managed by pipelines.

        These guardrails should be excluded from normal independent execution.
        """
        managed: Set[str] = set()
        for _policy_name, pipeline in pipelines:
            for step in pipeline.steps:
                managed.add(step.guardrail)
        return managed

    @staticmethod
    def get_all_resolved_policies(
        policies: Optional[Dict[str, Policy]] = None,
        context: Optional[PolicyMatchContext] = None,
    ) -> Dict[str, ResolvedPolicy]:
        """
        Resolve all policies and return their final guardrails.

        Useful for debugging and displaying policy configurations.

        Args:
            policies: Dictionary of all policies (if None, uses global registry)
            context: Optional context for evaluating conditions

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
            )

        return resolved
