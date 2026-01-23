"""
Policy Registry - In-memory storage for policies.

Handles storing, retrieving, and managing policies.

Policies define WHAT guardrails to apply. WHERE they apply is defined
by policy_attachments (see AttachmentRegistry).
"""

from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.types.proxy.policy_engine import (
    ConditionOperator,
    Policy,
    PolicyCondition,
    PolicyConfig,
    PolicyGuardrails,
    PolicyStatement,
)


class PolicyRegistry:
    """
    In-memory registry for storing and managing policies.

    This is a singleton that holds all loaded policies and provides
    methods to access them.

    Policies define WHAT guardrails to apply:
    - Base guardrails via guardrails.add/remove
    - Inheritance via inherit field
    - Conditional guardrails via statements
    """

    def __init__(self):
        self._policies: Dict[str, Policy] = {}
        self._initialized: bool = False

    def load_policies(self, policies_config: Dict[str, Any]) -> None:
        """
        Load policies from a configuration dictionary.

        Args:
            policies_config: Dictionary mapping policy names to policy definitions.
                            This is the raw config from the YAML file.
        """
        self._policies = {}

        for policy_name, policy_data in policies_config.items():
            try:
                policy = self._parse_policy(policy_name, policy_data)
                self._policies[policy_name] = policy
                verbose_proxy_logger.debug(f"Loaded policy: {policy_name}")
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error loading policy '{policy_name}': {str(e)}"
                )
                raise ValueError(f"Invalid policy '{policy_name}': {str(e)}") from e

        self._initialized = True
        verbose_proxy_logger.info(f"Loaded {len(self._policies)} policies")

    def _parse_policy(self, policy_name: str, policy_data: Dict[str, Any]) -> Policy:
        """
        Parse a policy from raw configuration data.

        Args:
            policy_name: Name of the policy
            policy_data: Raw policy configuration

        Returns:
            Parsed Policy object
        """
        # Parse guardrails
        guardrails_data = policy_data.get("guardrails", {})
        if isinstance(guardrails_data, dict):
            guardrails = PolicyGuardrails(
                add=guardrails_data.get("add"),
                remove=guardrails_data.get("remove"),
            )
        else:
            # Handle legacy format where guardrails might be a list
            guardrails = PolicyGuardrails(add=guardrails_data if guardrails_data else None)

        # Parse statements (conditional guardrails)
        statements = None
        statements_data = policy_data.get("statements")
        if statements_data:
            statements = [
                self._parse_statement(stmt_data) for stmt_data in statements_data
            ]

        return Policy(
            inherit=policy_data.get("inherit"),
            guardrails=guardrails,
            statements=statements,
            description=policy_data.get("description"),
        )

    def _parse_statement(self, stmt_data: Dict[str, Any]) -> PolicyStatement:
        """
        Parse a policy statement from raw configuration data.

        Args:
            stmt_data: Raw statement configuration

        Returns:
            Parsed PolicyStatement object
        """
        condition = None
        condition_data = stmt_data.get("condition")
        if condition_data:
            condition = self._parse_condition(condition_data)

        return PolicyStatement(
            sid=stmt_data.get("sid"),
            guardrails=stmt_data.get("guardrails", []),
            condition=condition,
        )

    def _parse_condition(self, condition_data: Dict[str, Any]) -> PolicyCondition:
        """
        Parse a policy condition from raw configuration data.

        Args:
            condition_data: Raw condition configuration

        Returns:
            Parsed PolicyCondition object
        """
        model_op = None
        team_op = None
        key_op = None
        metadata_ops = None

        if "model" in condition_data:
            model_op = self._parse_operator(condition_data["model"])
        if "team" in condition_data:
            team_op = self._parse_operator(condition_data["team"])
        if "key" in condition_data:
            key_op = self._parse_operator(condition_data["key"])
        if "metadata" in condition_data:
            metadata_ops = {
                field: self._parse_operator(op_data)
                for field, op_data in condition_data["metadata"].items()
            }

        return PolicyCondition(
            model=model_op,
            team=team_op,
            key=key_op,
            metadata=metadata_ops,
        )

    def _parse_operator(self, op_data: Dict[str, Any]) -> ConditionOperator:
        """
        Parse a condition operator from raw configuration data.

        Args:
            op_data: Raw operator configuration

        Returns:
            Parsed ConditionOperator object
        """
        # Build dict with alias names for Pydantic
        operator_dict: Dict[str, Any] = {}
        if "equals" in op_data:
            operator_dict["equals"] = op_data["equals"]
        if "in" in op_data:
            operator_dict["in"] = op_data["in"]
        if "prefix" in op_data:
            operator_dict["prefix"] = op_data["prefix"]
        if "not_equals" in op_data:
            operator_dict["not_equals"] = op_data["not_equals"]
        if "notIn" in op_data:
            operator_dict["notIn"] = op_data["notIn"]
        elif "not_in" in op_data:
            operator_dict["notIn"] = op_data["not_in"]

        return ConditionOperator.model_validate(operator_dict)

    def get_policy(self, policy_name: str) -> Optional[Policy]:
        """
        Get a policy by name.

        Args:
            policy_name: Name of the policy to retrieve

        Returns:
            Policy object if found, None otherwise
        """
        return self._policies.get(policy_name)

    def get_all_policies(self) -> Dict[str, Policy]:
        """
        Get all loaded policies.

        Returns:
            Dictionary mapping policy names to Policy objects
        """
        return self._policies.copy()

    def get_policy_names(self) -> List[str]:
        """
        Get list of all policy names.

        Returns:
            List of policy names
        """
        return list(self._policies.keys())

    def has_policy(self, policy_name: str) -> bool:
        """
        Check if a policy exists.

        Args:
            policy_name: Name of the policy to check

        Returns:
            True if policy exists, False otherwise
        """
        return policy_name in self._policies

    def is_initialized(self) -> bool:
        """
        Check if the registry has been initialized with policies.

        Returns:
            True if policies have been loaded, False otherwise
        """
        return self._initialized

    def clear(self) -> None:
        """
        Clear all policies from the registry.
        """
        self._policies = {}
        self._initialized = False

    def add_policy(self, policy_name: str, policy: Policy) -> None:
        """
        Add or update a single policy.

        Args:
            policy_name: Name of the policy
            policy: Policy object to add
        """
        self._policies[policy_name] = policy
        verbose_proxy_logger.debug(f"Added/updated policy: {policy_name}")

    def remove_policy(self, policy_name: str) -> bool:
        """
        Remove a policy by name.

        Args:
            policy_name: Name of the policy to remove

        Returns:
            True if policy was removed, False if it didn't exist
        """
        if policy_name in self._policies:
            del self._policies[policy_name]
            verbose_proxy_logger.debug(f"Removed policy: {policy_name}")
            return True
        return False


# Global singleton instance
_policy_registry: Optional[PolicyRegistry] = None


def get_policy_registry() -> PolicyRegistry:
    """
    Get the global PolicyRegistry singleton.

    Returns:
        The global PolicyRegistry instance
    """
    global _policy_registry
    if _policy_registry is None:
        _policy_registry = PolicyRegistry()
    return _policy_registry
