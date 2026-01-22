"""
Policy Initialization - Loads policies from config and validates on startup.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.proxy.policy_engine.policy_validator import PolicyValidator
from litellm.types.proxy.policy_engine import PolicyValidationResponse

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


async def init_policies(
    policies_config: Dict[str, Any],
    prisma_client: Optional["PrismaClient"] = None,
    validate_db: bool = True,
    fail_on_error: bool = True,
) -> PolicyValidationResponse:
    """
    Initialize policies from configuration.

    This function:
    1. Parses the policy configuration
    2. Validates policies (guardrails exist, teams/keys exist in DB)
    3. Loads policies into the global registry

    Args:
        policies_config: Dictionary mapping policy names to policy definitions
        prisma_client: Optional Prisma client for database validation
        validate_db: Whether to validate team/key aliases against database
        fail_on_error: If True, raise exception on validation errors

    Returns:
        PolicyValidationResponse with validation results

    Raises:
        ValueError: If fail_on_error is True and validation errors are found
    """
    verbose_proxy_logger.info(f"Initializing {len(policies_config)} policies...")

    # Get the global registry
    registry = get_policy_registry()

    # Create validator
    validator = PolicyValidator(prisma_client=prisma_client)

    # Validate the configuration
    validation_result = await validator.validate_policy_config(
        policies_config,
        validate_db=validate_db,
    )

    # Log validation results
    if validation_result.errors:
        for error in validation_result.errors:
            verbose_proxy_logger.error(
                f"Policy validation error in '{error.policy_name}': "
                f"[{error.error_type}] {error.message}"
            )

    if validation_result.warnings:
        for warning in validation_result.warnings:
            verbose_proxy_logger.warning(
                f"Policy validation warning in '{warning.policy_name}': "
                f"[{warning.error_type}] {warning.message}"
            )

    # Fail if there are errors and fail_on_error is True
    if not validation_result.valid and fail_on_error:
        error_messages = [
            f"[{e.policy_name}] {e.message}" for e in validation_result.errors
        ]
        raise ValueError(
            f"Policy validation failed with {len(validation_result.errors)} error(s):\n"
            + "\n".join(error_messages)
        )

    # Load policies into registry (even with warnings)
    try:
        registry.load_policies(policies_config)
        verbose_proxy_logger.info(
            f"Successfully loaded {len(policies_config)} policies"
        )
    except Exception as e:
        verbose_proxy_logger.error(f"Failed to load policies: {str(e)}")
        raise

    return validation_result


def init_policies_sync(
    policies_config: Dict[str, Any],
    fail_on_error: bool = True,
) -> None:
    """
    Synchronous version of init_policies (without DB validation).

    Use this when async is not available or DB validation is not needed.

    Args:
        policies_config: Dictionary mapping policy names to policy definitions
        fail_on_error: If True, raise exception on validation errors
    """
    import asyncio

    # Run the async function without DB validation
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(
        init_policies(
            policies_config=policies_config,
            prisma_client=None,
            validate_db=False,
            fail_on_error=fail_on_error,
        )
    )


def get_policies_summary() -> Dict[str, Any]:
    """
    Get a summary of loaded policies for debugging/display.

    Returns:
        Dictionary with policy information
    """
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver

    registry = get_policy_registry()

    if not registry.is_initialized():
        return {"initialized": False, "policies": {}}

    resolved = PolicyResolver.get_all_resolved_policies()

    summary = {
        "initialized": True,
        "policy_count": len(resolved),
        "policies": {},
    }

    for policy_name, resolved_policy in resolved.items():
        policy = registry.get_policy(policy_name)
        summary["policies"][policy_name] = {
            "inherit": policy.inherit if policy else None,
            "scope": {
                "teams": policy.scope.get_teams() if policy else [],
                "keys": policy.scope.get_keys() if policy else [],
                "models": policy.scope.get_models() if policy else [],
            },
            "guardrails_add": policy.guardrails.get_add() if policy else [],
            "guardrails_remove": policy.guardrails.get_remove() if policy else [],
            "resolved_guardrails": resolved_policy.guardrails,
            "inheritance_chain": resolved_policy.inheritance_chain,
        }

    return summary
