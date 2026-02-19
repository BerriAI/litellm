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
from pydantic import BaseModel

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


class EnrichTemplateRequest(BaseModel):
    template_id: str
    parameters: dict


@router.post(
    "/policy/templates/enrich",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def enrich_policy_template(
    data: EnrichTemplateRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> dict:
    """
    Enrich a policy template with LLM-discovered data (e.g. competitor names).

    Calls an onboarded LLM to discover competitors for the given brand name,
    then returns enriched guardrailDefinitions with the discovered data populated.
    """
    templates = _load_policy_templates_from_local_backup()
    template = next((t for t in templates if t.get("id") == data.template_id), None)
    if template is None:
        raise HTTPException(status_code=404, detail=f"Template '{data.template_id}' not found")

    llm_enrichment = template.get("llm_enrichment")
    if llm_enrichment is None:
        raise HTTPException(
            status_code=400,
            detail="Template does not support LLM enrichment",
        )

    brand_name = data.parameters.get(llm_enrichment["parameter"], "")
    if not brand_name:
        raise HTTPException(
            status_code=400,
            detail=f"Parameter '{llm_enrichment['parameter']}' is required",
        )

    prompt = llm_enrichment["prompt"].replace(
        "{{" + llm_enrichment["parameter"] + "}}", brand_name
    )

    competitors = await _discover_competitors_via_llm(prompt)

    enriched_definitions = _build_competitor_guardrail_definitions(
        template.get("guardrailDefinitions", []),
        competitors,
        brand_name,
    )

    return {"guardrailDefinitions": enriched_definitions, "competitors": competitors}


async def _discover_competitors_via_llm(prompt: str) -> list:
    """Call an onboarded LLM to discover competitor names."""
    import litellm

    try:
        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = response.choices[0].message.content or ""  # type: ignore
        competitors = [
            line.strip().strip(".-) ").strip()
            for line in raw.strip().split("\n")
            if line.strip() and len(line.strip()) > 1
        ]
        return competitors[:15]
    except Exception as e:
        verbose_proxy_logger.error("LLM competitor discovery failed: %s", e)
        return []


def _build_competitor_guardrail_definitions(
    definitions: list,
    competitors: list,
    brand_name: str,
) -> list:
    """Build enriched guardrailDefinitions with competitor names populated."""
    import copy

    enriched = copy.deepcopy(definitions)

    output_blocked = [
        {"keyword": comp, "action": "BLOCK", "description": f"Competitor: {comp}"}
        for comp in competitors
    ]

    recommendation_blocked = []
    for comp in competitors:
        recommendation_blocked.append(
            {"keyword": f"try {comp}", "action": "BLOCK", "description": "Recommendation to competitor"}
        )
        recommendation_blocked.append(
            {"keyword": f"use {comp}", "action": "BLOCK", "description": "Recommendation to competitor"}
        )
        recommendation_blocked.append(
            {"keyword": f"switch to {comp}", "action": "BLOCK", "description": "Recommendation to competitor"}
        )
        recommendation_blocked.append(
            {"keyword": f"consider {comp}", "action": "BLOCK", "description": "Recommendation to competitor"}
        )

    comparison_blocked = []
    for comp in competitors:
        comparison_blocked.append(
            {"keyword": f"{comp} is better", "action": "BLOCK", "description": "Unfavorable comparison"}
        )
        comparison_blocked.append(
            {"keyword": f"better than {brand_name}", "action": "BLOCK", "description": "Unfavorable comparison"}
        )
        comparison_blocked.append(
            {"keyword": f"{brand_name} is worse", "action": "BLOCK", "description": "Unfavorable comparison"}
        )

    blocked_words_map = {
        "competitor-output-blocker": output_blocked,
        "competitor-recommendation-filter": recommendation_blocked,
        "competitor-comparison-filter": comparison_blocked,
    }

    for defn in enriched:
        guardrail_name = defn.get("guardrail_name", "")
        if guardrail_name in blocked_words_map:
            defn["litellm_params"]["blocked_words"] = blocked_words_map[guardrail_name]

    return enriched
