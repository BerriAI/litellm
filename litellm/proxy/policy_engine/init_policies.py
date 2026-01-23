"""
Policy Initialization - Loads policies from config and validates on startup.

Configuration structure:
- policies: Define WHAT guardrails to apply (with inheritance and conditions)
- policy_attachments: Define WHERE policies apply (teams, keys, models)
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.proxy.policy_engine.policy_validator import PolicyValidator
from litellm.types.proxy.policy_engine import PolicyValidationResponse

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

# ANSI color codes for terminal output
_green_color_code = "\033[92m"
_blue_color_code = "\033[94m"
_yellow_color_code = "\033[93m"
_reset_color_code = "\033[0m"


def _print_policies_on_startup(
    policies_config: Dict[str, Any],
    policy_attachments_config: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Print loaded policies to console on startup (similar to model list).
    """
    import sys

    print(  # noqa: T201
        f"{_green_color_code}\nLiteLLM Policy Engine: Loaded {len(policies_config)} policies{_reset_color_code}\n"
    )
    sys.stdout.flush()

    for policy_name, policy_data in policies_config.items():
        guardrails = policy_data.get("guardrails", {})
        inherit = policy_data.get("inherit")
        condition = policy_data.get("condition")
        description = policy_data.get("description")

        guardrails_add = guardrails.get("add", []) if isinstance(guardrails, dict) else []
        guardrails_remove = guardrails.get("remove", []) if isinstance(guardrails, dict) else []
        inherit_str = f" (inherits: {inherit})" if inherit else ""

        print(  # noqa: T201
            f"{_blue_color_code}  - {policy_name}{inherit_str}{_reset_color_code}"
        )
        if description:
            print(f"      description: {description}")  # noqa: T201
        if guardrails_add:
            print(f"      guardrails.add: {guardrails_add}")  # noqa: T201
        if guardrails_remove:
            print(f"      guardrails.remove: {guardrails_remove}")  # noqa: T201
        if condition:
            model_condition = condition.get("model") if isinstance(condition, dict) else None
            if model_condition:
                print(f"      condition.model: {model_condition}")  # noqa: T201

    # Print attachments
    if policy_attachments_config:
        print(  # noqa: T201
            f"\n{_yellow_color_code}Policy Attachments: {len(policy_attachments_config)} attachment(s){_reset_color_code}"
        )
        for attachment in policy_attachments_config:
            policy = attachment.get("policy", "unknown")
            scope = attachment.get("scope")
            teams = attachment.get("teams")
            keys = attachment.get("keys")
            models = attachment.get("models")

            scope_parts = []
            if scope == "*":
                scope_parts.append("scope=* (global)")
            if teams:
                scope_parts.append(f"teams={teams}")
            if keys:
                scope_parts.append(f"keys={keys}")
            if models:
                scope_parts.append(f"models={models}")
            scope_str = ", ".join(scope_parts) if scope_parts else "all"

            print(f"  - {policy} -> {scope_str}")  # noqa: T201
    else:
        print(  # noqa: T201
            f"\n{_yellow_color_code}Warning: No policy_attachments configured. Policies will not be applied to any requests.{_reset_color_code}"
        )

    print()  # noqa: T201
    sys.stdout.flush()


async def init_policies(
    policies_config: Dict[str, Any],
    policy_attachments_config: Optional[List[Dict[str, Any]]] = None,
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
    4. Loads attachments into the attachment registry (if provided)

    Args:
        policies_config: Dictionary mapping policy names to policy definitions
        policy_attachments_config: Optional list of policy attachment configurations
        prisma_client: Optional Prisma client for database validation
        validate_db: Whether to validate team/key aliases against database
        fail_on_error: If True, raise exception on validation errors

    Returns:
        PolicyValidationResponse with validation results

    Raises:
        ValueError: If fail_on_error is True and validation errors are found
    """
    verbose_proxy_logger.info(f"Initializing {len(policies_config)} policies...")

    # Print policies to console on startup
    _print_policies_on_startup(policies_config, policy_attachments_config)

    # Get the global registries
    policy_registry = get_policy_registry()
    attachment_registry = get_attachment_registry()

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
        policy_registry.load_policies(policies_config)
        verbose_proxy_logger.info(
            f"Successfully loaded {len(policies_config)} policies"
        )
    except Exception as e:
        verbose_proxy_logger.error(f"Failed to load policies: {str(e)}")
        raise

    # Load attachments if provided
    if policy_attachments_config:
        try:
            attachment_registry.load_attachments(policy_attachments_config)
            verbose_proxy_logger.info(
                f"Successfully loaded {len(policy_attachments_config)} policy attachments"
            )
        except Exception as e:
            verbose_proxy_logger.error(f"Failed to load policy attachments: {str(e)}")
            raise

    return validation_result


def init_policies_sync(
    policies_config: Dict[str, Any],
    policy_attachments_config: Optional[List[Dict[str, Any]]] = None,
    fail_on_error: bool = True,
) -> None:
    """
    Synchronous version of init_policies (without DB validation).

    Use this when async is not available or DB validation is not needed.

    Args:
        policies_config: Dictionary mapping policy names to policy definitions
        policy_attachments_config: Optional list of policy attachment configurations
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
            policy_attachments_config=policy_attachments_config,
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

    policy_registry = get_policy_registry()
    attachment_registry = get_attachment_registry()

    if not policy_registry.is_initialized():
        return {"initialized": False, "policies": {}, "attachments": []}

    resolved = PolicyResolver.get_all_resolved_policies()

    summary: Dict[str, Any] = {
        "initialized": True,
        "policy_count": len(resolved),
        "attachment_count": len(attachment_registry.get_all_attachments()),
        "policies": {},
        "attachments": [],
    }

    for policy_name, resolved_policy in resolved.items():
        policy = policy_registry.get_policy(policy_name)
        summary["policies"][policy_name] = {
            "inherit": policy.inherit if policy else None,
            "description": policy.description if policy else None,
            "guardrails_add": policy.guardrails.get_add() if policy else [],
            "guardrails_remove": policy.guardrails.get_remove() if policy else [],
            "condition": policy.condition.model_dump() if policy and policy.condition else None,
            "resolved_guardrails": resolved_policy.guardrails,
            "inheritance_chain": resolved_policy.inheritance_chain,
        }

    # Add attachment info
    for attachment in attachment_registry.get_all_attachments():
        summary["attachments"].append({
            "policy": attachment.policy,
            "scope": attachment.scope,
            "teams": attachment.teams,
            "keys": attachment.keys,
            "models": attachment.models,
        })

    return summary
