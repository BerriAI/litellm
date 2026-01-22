"""
Policy Validator - Validates policy configurations.

Validates:
- Guardrail names exist in the guardrail registry
- Non-wildcard team aliases exist in the database
- Non-wildcard key aliases exist in the database
- Non-wildcard model names exist in the router or match a wildcard route
- Inheritance chains are valid (no cycles, parents exist)
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.types.proxy.policy_engine import (
    Policy,
    PolicyValidationError,
    PolicyValidationErrorType,
    PolicyValidationResponse,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router


class PolicyValidator:
    """
    Validates policy configurations against actual data.
    """

    def __init__(
        self,
        prisma_client: Optional["PrismaClient"] = None,
        llm_router: Optional["Router"] = None,
    ):
        """
        Initialize the validator.

        Args:
            prisma_client: Optional Prisma client for database validation
            llm_router: Optional LLM router for model validation
        """
        self.prisma_client = prisma_client
        self.llm_router = llm_router

    @staticmethod
    def is_wildcard_pattern(pattern: str) -> bool:
        """
        Check if a pattern contains wildcards.

        Args:
            pattern: The pattern to check

        Returns:
            True if the pattern contains wildcard characters
        """
        return "*" in pattern or "?" in pattern

    def get_available_guardrails(self) -> Set[str]:
        """
        Get set of available guardrail names from the guardrail registry.

        Returns:
            Set of guardrail names
        """
        try:
            from litellm.proxy.guardrails.guardrail_registry import (
                IN_MEMORY_GUARDRAIL_HANDLER,
            )

            guardrails = IN_MEMORY_GUARDRAIL_HANDLER.list_in_memory_guardrails()
            return {g.get("guardrail_name", "") for g in guardrails if g.get("guardrail_name")}
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Could not get guardrails from registry: {str(e)}"
            )
            return set()

    async def check_team_alias_exists(self, team_alias: str) -> bool:
        """
        Check if a specific team alias exists in the database.

        Args:
            team_alias: The team alias to check

        Returns:
            True if the team alias exists
        """
        if self.prisma_client is None:
            return True  # Can't validate without DB, assume valid

        try:
            team = await self.prisma_client.db.litellm_teamtable.find_first(
                where={"team_alias": team_alias},
            )
            return team is not None
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Could not check team alias '{team_alias}': {str(e)}"
            )
            return True  # Assume valid on error

    async def check_key_alias_exists(self, key_alias: str) -> bool:
        """
        Check if a specific key alias exists in the database.

        Args:
            key_alias: The key alias to check

        Returns:
            True if the key alias exists
        """
        if self.prisma_client is None:
            return True  # Can't validate without DB, assume valid

        try:
            key = await self.prisma_client.db.litellm_verificationtoken.find_first(
                where={"key_alias": key_alias},
            )
            return key is not None
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Could not check key alias '{key_alias}': {str(e)}"
            )
            return True  # Assume valid on error

    def check_model_exists(self, model: str) -> bool:
        """
        Check if a model exists in the router or matches a wildcard pattern.

        Args:
            model: The model name to check

        Returns:
            True if the model exists or matches a pattern in the router
        """
        if self.llm_router is None:
            return True  # Can't validate without router, assume valid

        try:
            # Check if model is in router's model names
            if model in self.llm_router.model_names:
                return True

            # Check if model matches any pattern via pattern router
            if hasattr(self.llm_router, "pattern_router"):
                pattern_deployments = self.llm_router.pattern_router.get_deployments_by_pattern(
                    model=model
                )
                if pattern_deployments:
                    return True

            return False
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Could not check model '{model}': {str(e)}"
            )
            return True  # Assume valid on error

    def _validate_inheritance_chain(
        self,
        policy_name: str,
        policies: Dict[str, Policy],
        visited: Optional[Set[str]] = None,
    ) -> List[PolicyValidationError]:
        """
        Validate the inheritance chain for a policy.

        Checks for:
        - Parent policy exists
        - No circular inheritance

        Args:
            policy_name: Name of the policy to validate
            policies: All policies
            visited: Set of already visited policy names (for cycle detection)

        Returns:
            List of validation errors
        """
        errors: List[PolicyValidationError] = []

        if visited is None:
            visited = set()

        if policy_name in visited:
            errors.append(
                PolicyValidationError(
                    policy_name=policy_name,
                    error_type=PolicyValidationErrorType.CIRCULAR_INHERITANCE,
                    message=f"Circular inheritance detected: {' -> '.join(visited)} -> {policy_name}",
                    field="inherit",
                )
            )
            return errors

        policy = policies.get(policy_name)
        if policy is None:
            return errors

        if policy.inherit:
            if policy.inherit not in policies:
                errors.append(
                    PolicyValidationError(
                        policy_name=policy_name,
                        error_type=PolicyValidationErrorType.INVALID_INHERITANCE,
                        message=f"Parent policy '{policy.inherit}' not found",
                        field="inherit",
                        value=policy.inherit,
                    )
                )
            else:
                # Recursively check parent
                visited.add(policy_name)
                errors.extend(
                    self._validate_inheritance_chain(policy.inherit, policies, visited)
                )

        return errors

    async def validate_policies(
        self,
        policies: Dict[str, Policy],
        validate_db: bool = True,
    ) -> PolicyValidationResponse:
        """
        Validate a set of policies.

        Args:
            policies: Dictionary mapping policy names to Policy objects
            validate_db: Whether to validate against database (teams, keys)

        Returns:
            PolicyValidationResponse with errors and warnings
        """
        errors: List[PolicyValidationError] = []
        warnings: List[PolicyValidationError] = []

        # Get available guardrails
        available_guardrails = self.get_available_guardrails()

        for policy_name, policy in policies.items():
            # Validate guardrails
            for guardrail in policy.guardrails.get_add():
                if available_guardrails and guardrail not in available_guardrails:
                    errors.append(
                        PolicyValidationError(
                            policy_name=policy_name,
                            error_type=PolicyValidationErrorType.INVALID_GUARDRAIL,
                            message=f"Guardrail '{guardrail}' not found in guardrail registry",
                            field="guardrails.add",
                            value=guardrail,
                        )
                    )

            for guardrail in policy.guardrails.get_remove():
                if available_guardrails and guardrail not in available_guardrails:
                    warnings.append(
                        PolicyValidationError(
                            policy_name=policy_name,
                            error_type=PolicyValidationErrorType.INVALID_GUARDRAIL,
                            message=f"Guardrail '{guardrail}' in remove list not found in guardrail registry",
                            field="guardrails.remove",
                            value=guardrail,
                        )
                    )

            # Validate team aliases (non-wildcard only, query per alias)
            if validate_db and self.prisma_client:
                for team_pattern in policy.scope.get_teams():
                    if not self.is_wildcard_pattern(team_pattern):
                        exists = await self.check_team_alias_exists(team_alias=team_pattern)
                        if not exists:
                            warnings.append(
                                PolicyValidationError(
                                    policy_name=policy_name,
                                    error_type=PolicyValidationErrorType.INVALID_TEAM,
                                    message=f"Team alias '{team_pattern}' not found in database",
                                    field="scope.teams",
                                    value=team_pattern,
                                )
                            )

            # Validate key aliases (non-wildcard only, query per alias)
            if validate_db and self.prisma_client:
                for key_pattern in policy.scope.get_keys():
                    if not self.is_wildcard_pattern(key_pattern):
                        exists = await self.check_key_alias_exists(key_alias=key_pattern)
                        if not exists:
                            warnings.append(
                                PolicyValidationError(
                                    policy_name=policy_name,
                                    error_type=PolicyValidationErrorType.INVALID_KEY,
                                    message=f"Key alias '{key_pattern}' not found in database",
                                    field="scope.keys",
                                    value=key_pattern,
                                )
                            )

            # Validate models (non-wildcard only, check against router)
            if self.llm_router:
                for model_pattern in policy.scope.get_models():
                    if not self.is_wildcard_pattern(model_pattern):
                        exists = self.check_model_exists(model=model_pattern)
                        if not exists:
                            warnings.append(
                                PolicyValidationError(
                                    policy_name=policy_name,
                                    error_type=PolicyValidationErrorType.INVALID_MODEL,
                                    message=f"Model '{model_pattern}' not found in router",
                                    field="scope.models",
                                    value=model_pattern,
                                )
                            )

            # Validate inheritance
            inheritance_errors = self._validate_inheritance_chain(
                policy_name=policy_name, policies=policies
            )
            errors.extend(inheritance_errors)

        return PolicyValidationResponse(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    async def validate_policy_config(
        self,
        policy_config: Dict[str, Any],
        validate_db: bool = True,
    ) -> PolicyValidationResponse:
        """
        Validate a raw policy configuration dictionary.

        This parses the config and then validates it.

        Args:
            policy_config: Raw policy configuration from YAML
            validate_db: Whether to validate against database

        Returns:
            PolicyValidationResponse with errors and warnings
        """
        from litellm.proxy.policy_engine.policy_registry import PolicyRegistry

        # First, try to parse the policies
        errors: List[PolicyValidationError] = []
        policies: Dict[str, Policy] = {}

        temp_registry = PolicyRegistry()

        for policy_name, policy_data in policy_config.items():
            try:
                policy = temp_registry._parse_policy(policy_name, policy_data)
                policies[policy_name] = policy
            except Exception as e:
                errors.append(
                    PolicyValidationError(
                        policy_name=policy_name,
                        error_type=PolicyValidationErrorType.INVALID_SYNTAX,
                        message=f"Failed to parse policy: {str(e)}",
                    )
                )

        # If there were parsing errors, return early
        if errors:
            return PolicyValidationResponse(
                valid=False,
                errors=errors,
                warnings=[],
            )

        # Validate the parsed policies
        return await self.validate_policies(policies, validate_db=validate_db)
