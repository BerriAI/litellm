"""
POLICY MANAGEMENT

All /policy management endpoints

/policy/validate - Validate a policy configuration
/policy/list - List all loaded policies
/policy/info - Get information about a specific policy
/policy/templates - Get policy templates (GitHub with local fallback)
"""

import json
import os

from fastapi import APIRouter, Depends, HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.types.proxy.policy_engine import (
    PolicyGuardrailsResponse,
    PolicyInfoResponse,
    PolicyListResponse,
    PolicyMatchContext,
    PolicyScopeResponse,
    PolicySummaryItem,
    PolicyTestResponse,
    PolicyValidateRequest,
    PolicyValidationResponse,
)

router = APIRouter()


@router.post(
    "/policy/validate",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyValidationResponse,
)
@management_endpoint_wrapper
async def validate_policy(
    request: Request,
    data: PolicyValidateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PolicyValidationResponse:
    """
    Validate a policy configuration before applying it.

    Checks:
    - All referenced guardrails exist in the guardrail registry
    - All non-wildcard team aliases exist in the database
    - All non-wildcard key aliases exist in the database
    - Inheritance chains are valid (no cycles, parents exist)
    - Scope patterns are syntactically valid

    Returns:
    - valid: True if the policy configuration is valid (no blocking errors)
    - errors: List of blocking validation errors
    - warnings: List of non-blocking validation warnings

    Example request:
    ```json
    {
        "policies": {
            "global-baseline": {
                "guardrails": {
                    "add": ["pii_blocker", "phi_blocker"]
                },
                "scope": {
                    "teams": ["*"],
                    "keys": ["*"],
                    "models": ["*"]
                }
            },
            "healthcare-compliance": {
                "inherit": "global-baseline",
                "guardrails": {
                    "add": ["hipaa_audit"]
                },
                "scope": {
                    "teams": ["healthcare-team"]
                }
            }
        }
    }
    ```
    """
    from litellm.proxy.policy_engine.policy_validator import PolicyValidator
    from litellm.proxy.proxy_server import prisma_client

    verbose_proxy_logger.debug(
        f"Validating policy configuration with {len(data.policies)} policies"
    )

    validator = PolicyValidator(prisma_client=prisma_client)

    result = await validator.validate_policy_config(
        data.policies,
        validate_db=prisma_client is not None,
    )

    return result


@router.get(
    "/policy/list",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyListResponse,
)
@management_endpoint_wrapper
async def list_policies(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PolicyListResponse:
    """
    List all loaded policies with their resolved guardrails.

    Returns information about each policy including:
    - Inheritance configuration
    - Scope (teams, keys, models)
    - Guardrails to add/remove
    - Resolved guardrails (after inheritance)
    - Inheritance chain
    """
    from litellm.proxy.policy_engine.init_policies import get_policies_summary

    summary = get_policies_summary()
    return PolicyListResponse(
        policies={
            name: PolicySummaryItem(
                inherit=data.get("inherit"),
                scope=PolicyScopeResponse(**data.get("scope", {})),
                guardrails=PolicyGuardrailsResponse(**data.get("guardrails", {})),
                resolved_guardrails=data.get("resolved_guardrails", []),
                inheritance_chain=data.get("inheritance_chain", []),
            )
            for name, data in summary.get("policies", {}).items()
        },
        total_count=summary.get("total_count", 0),
    )


@router.get(
    "/policy/info/{policy_name}",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyInfoResponse,
)
@management_endpoint_wrapper
async def get_policy_info(
    request: Request,
    policy_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PolicyInfoResponse:
    """
    Get detailed information about a specific policy.

    Returns:
    - Policy configuration
    - Resolved guardrails (after inheritance)
    - Inheritance chain
    """
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver

    registry = get_policy_registry()

    if not registry.is_initialized():
        raise HTTPException(
            status_code=404,
            detail="Policy engine not initialized. No policies loaded.",
        )

    policy = registry.get_policy(policy_name)
    if policy is None:
        raise HTTPException(
            status_code=404,
            detail=f"Policy '{policy_name}' not found",
        )

    resolved = PolicyResolver.resolve_policy_guardrails(
        policy_name=policy_name, policies=registry.get_all_policies()
    )

    return PolicyInfoResponse(
        policy_name=policy_name,
        inherit=policy.inherit,
        scope=PolicyScopeResponse(
            teams=[],
            keys=[],
            models=[],
        ),
        guardrails=PolicyGuardrailsResponse(
            add=policy.guardrails.get_add(),
            remove=policy.guardrails.get_remove(),
        ),
        resolved_guardrails=resolved.guardrails,
        inheritance_chain=resolved.inheritance_chain,
    )


@router.post(
    "/policy/test",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyTestResponse,
)
@management_endpoint_wrapper
async def test_policy_matching(
    request: Request,
    context: PolicyMatchContext,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PolicyTestResponse:
    """
    Test which policies would match a given request context.

    This is useful for debugging and understanding policy behavior.

    Request body:
    ```json
    {
        "team_alias": "healthcare-team",
        "key_alias": "my-api-key",
        "model": "gpt-4"
    }
    ```

    Returns:
    - matching_policies: List of policy names that match
    - resolved_guardrails: Final list of guardrails that would be applied
    """
    from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver

    registry = get_policy_registry()

    if not registry.is_initialized():
        return PolicyTestResponse(
            context=context,
            matching_policies=[],
            resolved_guardrails=[],
            message="Policy engine not initialized. No policies loaded.",
        )

    policies = registry.get_all_policies()

    # Get matching policies
    matching_policy_names = PolicyMatcher.get_matching_policies(context=context)

    # Resolve guardrails
    resolved_guardrails = PolicyResolver.resolve_guardrails_for_context(
        context=context, policies=policies
    )

    return PolicyTestResponse(
        context=context,
        matching_policies=matching_policy_names,
        resolved_guardrails=resolved_guardrails,
    )


POLICY_TEMPLATES_GITHUB_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/policy_templates.json"


def _load_policy_templates_from_local_backup() -> list:
    """Load policy templates from local backup file (litellm/policy_templates_backup.json)."""
    backup_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "policy_templates_backup.json",
    )
    path = os.path.abspath(backup_path)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


@router.get(
    "/policy/templates",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def get_policy_templates(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> list:
    """
    Get policy templates for the UI (pre-configured guardrail combinations).

    Fetches from GitHub with automatic fallback to local backup on failure.
    Set LITELLM_LOCAL_POLICY_TEMPLATES=true to skip GitHub and use local backup only.
    """
    use_local = os.getenv("LITELLM_LOCAL_POLICY_TEMPLATES", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    if use_local:
        return _load_policy_templates_from_local_backup()

    try:
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
        from litellm.types.llms.custom_http import httpxSpecialProvider

        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.UI,
            params={"timeout": 10.0},
        )
        response = await async_client.get(POLICY_TEMPLATES_GITHUB_URL)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        verbose_proxy_logger.debug(
            "Failed to fetch policy templates from GitHub, using local backup: %s", e
        )

    return _load_policy_templates_from_local_backup()
