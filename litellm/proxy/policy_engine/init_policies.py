"""
Policy Initialization - Loads policies from config and validates on startup.

Configuration structure:
- policies: Define WHAT guardrails to apply (with inheritance and conditions)
- policy_attachments: Define WHERE policies apply (teams, keys, models)
"""

import sys
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, Final, Optional

from litellm._logging import print_startup_message, verbose_proxy_logger
from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.proxy.policy_engine.policy_validator import PolicyValidator
from litellm.types.proxy.policy_engine import Policy, PolicyValidationResponse, ResolvedPolicy

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

# ANSI color codes for terminal output
_green_color_code: Final = "\033[92m"
_blue_color_code: Final = "\033[94m"
_yellow_color_code: Final = "\033[93m"
_reset_color_code: Final = "\033[0m"


def _policy_lines(policy_name: str, policy_data: Mapping[str, object]) -> tuple[str, ...]:
    guardrails: Final = policy_data.get("guardrails", {})
    inherit: Final = policy_data.get("inherit")
    condition: Final = policy_data.get("condition")
    description: Final = policy_data.get("description")

    guardrails_add: Final = guardrails.get("add", []) if isinstance(guardrails, dict) else []
    guardrails_remove: Final = guardrails.get("remove", []) if isinstance(guardrails, dict) else []
    model_condition: Final = condition.get("model") if isinstance(condition, dict) else None
    inherit_str: Final = f" (inherits: {inherit})" if inherit else ""

    return (
        f"{_blue_color_code}  - {policy_name}{inherit_str}{_reset_color_code}",
        *((f"      description: {description}",) if description else ()),
        *((f"      guardrails.add: {guardrails_add}",) if guardrails_add else ()),
        *((f"      guardrails.remove: {guardrails_remove}",) if guardrails_remove else ()),
        *((f"      condition.model: {model_condition}",) if model_condition else ()),
    )


def _attachment_line(attachment: Mapping[str, object]) -> str:
    policy: Final = attachment.get("policy", "unknown")
    scope: Final = attachment.get("scope")
    teams: Final = attachment.get("teams")
    keys: Final = attachment.get("keys")
    models: Final = attachment.get("models")

    scope_parts: Final = (
        *(("scope=* (global)",) if scope == "*" else ()),
        *((f"teams={teams}",) if teams else ()),
        *((f"keys={keys}",) if keys else ()),
        *((f"models={models}",) if models else ()),
    )
    return f"  - {policy} -> {', '.join(scope_parts) if scope_parts else 'all'}"


def _print_policies_on_startup(
    policies_config: Mapping[str, Mapping[str, object]],
    policy_attachments_config: Sequence[Mapping[str, object]] | None = None,
) -> None:
    """
    Print loaded policies to console on startup (similar to model list).
    """
    header: Final = (
        f"{_green_color_code}\nLiteLLM Policy Engine: Loaded {len(policies_config)} policies{_reset_color_code}\n"
    )
    policy_lines: Final = tuple(
        chain.from_iterable(
            _policy_lines(policy_name, policy_data) for policy_name, policy_data in policies_config.items()
        )
    )

    if policy_attachments_config:
        print_startup_message(
            "\n".join(
                (
                    header,
                    *policy_lines,
                    f"\n{_yellow_color_code}Policy Attachments: {len(policy_attachments_config)} attachment(s){_reset_color_code}",
                    *(_attachment_line(attachment) for attachment in policy_attachments_config),
                    "",
                )
            )
        )
    else:
        print_startup_message("\n".join((header, *policy_lines)))
        print_startup_message(
            f"\n{_yellow_color_code}Warning: No policy_attachments configured. Policies will not be applied to any requests.{_reset_color_code}\n",
        )
    sys.stdout.flush()


async def init_policies(
    policies_config: dict[str, Any],
    policy_attachments_config: list[dict[str, object]] | None = None,
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
    verbose_proxy_logger.info("Initializing %s policies...", len(policies_config))

    # Print policies to console on startup
    _print_policies_on_startup(policies_config, policy_attachments_config)

    # Get the global registries
    policy_registry: Final = get_policy_registry()
    attachment_registry: Final = get_attachment_registry()

    # Create validator
    validator: Final = PolicyValidator(prisma_client=prisma_client)

    # Validate the configuration
    validation_result: Final = await validator.validate_policy_config(
        policies_config,
        validate_db=validate_db,
    )

    # Log validation results
    if validation_result.errors:
        for error in validation_result.errors:
            verbose_proxy_logger.error(
                "Policy validation error in '%s': [%s] %s", error.policy_name, error.error_type, error.message
            )

    if validation_result.warnings:
        for warning in validation_result.warnings:
            verbose_proxy_logger.warning(
                "Policy validation warning in '%s': [%s] %s", warning.policy_name, warning.error_type, warning.message
            )

    # Fail if there are errors and fail_on_error is True
    if not validation_result.valid and fail_on_error:
        error_messages: Final = [f"[{e.policy_name}] {e.message}" for e in validation_result.errors]
        raise ValueError(
            f"Policy validation failed with {len(validation_result.errors)} error(s):\n" + "\n".join(error_messages)
        )

    # Load policies into registry (even with warnings)
    try:
        policy_registry.load_policies(policies_config)
        verbose_proxy_logger.info("Successfully loaded %s policies", len(policies_config))
    except Exception as e:
        verbose_proxy_logger.error("Failed to load policies: %s", e)
        raise

    # Load attachments if provided
    if policy_attachments_config:
        try:
            attachment_registry.load_attachments(policy_attachments_config)
            verbose_proxy_logger.info("Successfully loaded %s policy attachments", len(policy_attachments_config))
        except Exception as e:
            verbose_proxy_logger.error("Failed to load policy attachments: %s", e)
            raise

    return validation_result


def init_policies_sync(
    policies_config: dict[str, object],
    policy_attachments_config: list[dict[str, object]] | None = None,
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


def get_policies_summary() -> dict[str, object]:
    """
    Get a summary of loaded policies for debugging/display.

    Returns:
        Dictionary with policy information
    """
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver

    policy_registry: Final = get_policy_registry()
    attachment_registry: Final = get_attachment_registry()

    if not policy_registry.is_initialized():
        return {"initialized": False, "policies": {}, "attachments": []}

    resolved: Final = PolicyResolver.get_all_resolved_policies()

    return {
        "initialized": True,
        "policy_count": len(resolved),
        "attachment_count": len(attachment_registry.get_all_attachments()),
        "policies": {
            policy_name: _policy_summary_entry(policy_registry.get_policy(policy_name), resolved_policy)
            for policy_name, resolved_policy in resolved.items()
        },
        "attachments": [
            {
                "policy": attachment.policy,
                "scope": attachment.scope,
                "teams": attachment.teams,
                "keys": attachment.keys,
                "models": attachment.models,
            }
            for attachment in attachment_registry.get_all_attachments()
        ],
    }


def _policy_summary_entry(policy: Policy | None, resolved_policy: ResolvedPolicy) -> Mapping[str, object]:
    return {
        "inherit": policy.inherit if policy else None,
        "description": policy.description if policy else None,
        "guardrails_add": policy.guardrails.get_add() if policy else [],
        "guardrails_remove": policy.guardrails.get_remove() if policy else [],
        "condition": (policy.condition.model_dump() if policy and policy.condition else None),
        "resolved_guardrails": resolved_policy.guardrails,
        "inheritance_chain": resolved_policy.inheritance_chain,
    }
