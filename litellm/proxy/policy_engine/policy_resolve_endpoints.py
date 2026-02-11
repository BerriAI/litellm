"""
Policy resolve and attachment impact estimation endpoints.

- /policies/resolve — debug which guardrails apply for a given context
- /policies/attachments/estimate-impact — preview blast radius before creating an attachment
"""

import json

from fastapi import APIRouter, Depends, HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.types.proxy.policy_engine import (
    AttachmentImpactResponse,
    PolicyAttachmentCreateRequest,
    PolicyMatchContext,
    PolicyMatchDetail,
    PolicyResolveRequest,
    PolicyResolveResponse,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Policy Resolve Endpoint
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/policies/resolve",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyResolveResponse,
)
async def resolve_policies_for_context(
    request: PolicyResolveRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Resolve which policies and guardrails apply for a given context.

    Use this endpoint to debug "what guardrails would apply to a request
    with this team/key/model/tags combination?"

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/policies/resolve" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "tags": ["healthcare"],
            "model": "gpt-4"
        }'
    ```
    """
    from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # Sync from DB to ensure in-memory state is current
        await get_policy_registry().sync_policies_from_db(prisma_client)
        await get_attachment_registry().sync_attachments_from_db(prisma_client)

        # Build context from request
        context = PolicyMatchContext(
            team_alias=request.team_alias,
            key_alias=request.key_alias,
            model=request.model,
            tags=request.tags,
        )

        # Get matching policies with reasons
        match_results = get_attachment_registry().get_attached_policies_with_reasons(
            context=context
        )

        if not match_results:
            return PolicyResolveResponse(
                effective_guardrails=[],
                matched_policies=[],
            )

        # Filter by conditions
        policy_names = [r["policy_name"] for r in match_results]
        applied_policy_names = PolicyMatcher.get_policies_with_matching_conditions(
            policy_names=policy_names,
            context=context,
        )

        # Resolve guardrails for each applied policy
        matched_policies = []
        all_guardrails: set = set()
        for result in match_results:
            pname = result["policy_name"]
            if pname not in applied_policy_names:
                continue
            resolved = PolicyResolver.resolve_policy_guardrails(
                policy_name=pname,
                policies=get_policy_registry().get_all_policies(),
                context=context,
            )
            guardrails = resolved.guardrails if resolved else []
            all_guardrails.update(guardrails)
            matched_policies.append(
                PolicyMatchDetail(
                    policy_name=pname,
                    matched_via=result["matched_via"],
                    guardrails_added=guardrails,
                )
            )

        return PolicyResolveResponse(
            effective_guardrails=sorted(all_guardrails),
            matched_policies=matched_policies,
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error resolving policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Attachment Impact Estimation Endpoint
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/policies/attachments/estimate-impact",
    tags=["Policies"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=AttachmentImpactResponse,
)
async def estimate_attachment_impact(
    request: PolicyAttachmentCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Estimate how many keys and teams would be affected by a policy attachment.

    Use this before creating an attachment to preview the blast radius.

    Example Request:
    ```bash
    curl -X POST "http://localhost:4000/policies/attachments/estimate-impact" \\
        -H "Authorization: Bearer <your_api_key>" \\
        -H "Content-Type: application/json" \\
        -d '{
            "policy_name": "hipaa-compliance",
            "tags": ["healthcare", "health-*"]
        }'
    ```
    """
    from litellm.proxy.auth.route_checks import RouteChecks
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        tag_patterns = request.tags or []
        team_patterns = request.teams or []

        affected_keys: list = []
        affected_teams: list = []

        # If global scope, everything is affected — not useful to enumerate
        if request.scope == "*":
            return AttachmentImpactResponse(
                affected_keys_count=-1,
                affected_teams_count=-1,
                sample_keys=["(global scope — affects all keys)"],
                sample_teams=["(global scope — affects all teams)"],
            )

        # Check tag-based impact: find keys/teams whose metadata.tags match the patterns
        if tag_patterns:
            # Query keys with metadata
            keys = await prisma_client.db.litellm_verificationtoken.find_many(
                where={},
                order={"created_at": "desc"},
            )
            for key in keys:
                key_alias = key.key_alias or ""
                key_metadata = key.metadata_json if hasattr(key, "metadata_json") else (key.metadata or {})
                if isinstance(key_metadata, str):
                    try:
                        key_metadata = json.loads(key_metadata)
                    except (json.JSONDecodeError, TypeError):
                        key_metadata = {}
                key_tags = key_metadata.get("tags", []) if isinstance(key_metadata, dict) else []
                if key_tags and any(
                    RouteChecks._route_matches_wildcard_pattern(
                        route=tag, pattern=pattern
                    )
                    for tag in key_tags
                    for pattern in tag_patterns
                ):
                    affected_keys.append(key_alias or str(key.token)[:8] + "...")

            # Query teams with metadata
            teams = await prisma_client.db.litellm_teamtable.find_many(
                where={},
                order={"created_at": "desc"},
            )
            for team in teams:
                team_alias = team.team_alias or ""
                team_metadata = team.metadata or {}
                if isinstance(team_metadata, str):
                    try:
                        team_metadata = json.loads(team_metadata)
                    except (json.JSONDecodeError, TypeError):
                        team_metadata = {}
                team_tags = team_metadata.get("tags", []) if isinstance(team_metadata, dict) else []
                if team_tags and any(
                    RouteChecks._route_matches_wildcard_pattern(
                        route=tag, pattern=pattern
                    )
                    for tag in team_tags
                    for pattern in tag_patterns
                ):
                    affected_teams.append(team_alias or str(team.team_id)[:8] + "...")

        # Check team-based impact
        matched_team_ids: list = []
        if team_patterns:
            teams = await prisma_client.db.litellm_teamtable.find_many(
                where={},
                order={"created_at": "desc"},
            )
            for team in teams:
                team_alias = team.team_alias or ""
                if team_alias and any(
                    RouteChecks._route_matches_wildcard_pattern(
                        route=team_alias, pattern=pattern
                    )
                    for pattern in team_patterns
                ):
                    if team_alias not in affected_teams:
                        affected_teams.append(team_alias)
                        matched_team_ids.append(str(team.team_id))

            # Also find keys belonging to matched teams
            if matched_team_ids:
                keys = await prisma_client.db.litellm_verificationtoken.find_many(
                    where={"team_id": {"in": matched_team_ids}},
                    order={"created_at": "desc"},
                )
                for key in keys:
                    key_alias = key.key_alias or str(key.token)[:8] + "..."
                    if key_alias not in affected_keys:
                        affected_keys.append(key_alias)

        # Check key-based impact (direct key alias matching)
        key_patterns = request.keys or []
        if key_patterns:
            keys = await prisma_client.db.litellm_verificationtoken.find_many(
                where={},
                order={"created_at": "desc"},
            )
            for key in keys:
                key_alias = key.key_alias or ""
                if key_alias and any(
                    RouteChecks._route_matches_wildcard_pattern(
                        route=key_alias, pattern=pattern
                    )
                    for pattern in key_patterns
                ):
                    if key_alias not in affected_keys:
                        affected_keys.append(key_alias)

        return AttachmentImpactResponse(
            affected_keys_count=len(affected_keys),
            affected_teams_count=len(affected_teams),
            sample_keys=affected_keys[:10],
            sample_teams=affected_teams[:10],
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error estimating attachment impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))
