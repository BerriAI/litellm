"""
Policy resolve and attachment impact estimation endpoints.

- /policies/resolve — debug which guardrails apply for a given context
- /policies/attachments/estimate-impact — preview blast radius before creating an attachment
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm._logging import verbose_proxy_logger
from litellm.constants import MAX_POLICY_ESTIMATE_IMPACT_ROWS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
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


def _build_alias_where(field: str, patterns: list) -> dict:
    """Build a Prisma ``where`` clause for alias patterns.

    Supports exact matches and suffix wildcards (``prefix*``).
    Returns something like:
        {"OR": [{"field": {"in": ["a","b"]}}, {"field": {"startsWith": "dev-"}}]}
    """
    exact: list = []
    prefix_conditions: list = []
    for pat in patterns:
        if pat.endswith("*"):
            prefix_conditions.append({field: {"startsWith": pat[:-1]}})
        else:
            exact.append(pat)

    conditions: list = []
    if exact:
        conditions.append({field: {"in": exact}})
    conditions.extend(prefix_conditions)

    if not conditions:
        return {field: {"not": None}}
    if len(conditions) == 1:
        return conditions[0]
    return {"OR": conditions}


def _parse_metadata(raw_metadata: object) -> dict:
    """Parse metadata that may be a dict, JSON string, or None."""
    if raw_metadata is None:
        return {}
    if isinstance(raw_metadata, str):
        try:
            return json.loads(raw_metadata)
        except (json.JSONDecodeError, TypeError):
            return {}
    return raw_metadata if isinstance(raw_metadata, dict) else {}


def _get_tags_from_metadata(metadata: object, json_metadata: object = None) -> list:
    """Extract tags list from a metadata field (or metadata_json fallback)."""
    raw = json_metadata if json_metadata is not None else metadata
    parsed = _parse_metadata(raw)
    return parsed.get("tags", []) or []


async def _fetch_all_teams(prisma_client: object) -> list:
    """Fetch teams from DB once. Reuse the result across tag and alias lookups."""
    return await prisma_client.db.litellm_teamtable.find_many(  # type: ignore
        where={}, order={"created_at": "desc"}, take=MAX_POLICY_ESTIMATE_IMPACT_ROWS,
    )


def _filter_keys_by_tags(keys: list, tag_patterns: list) -> tuple:
    """Filter key rows whose metadata.tags match any of the given patterns.

    Returns (named_aliases, unnamed_count).
    """

    affected: list = []
    unnamed_count = 0
    for key in keys:
        key_alias = key.key_alias or ""
        key_tags = _get_tags_from_metadata(
            key.metadata, getattr(key, "metadata_json", None)
        )
        if key_tags and any(
            RouteChecks._route_matches_wildcard_pattern(route=tag, pattern=pat)
            for tag in key_tags
            for pat in tag_patterns
        ):
            if key_alias:
                affected.append(key_alias)
            else:
                unnamed_count += 1
    return affected, unnamed_count


def _filter_teams_by_tags(teams: list, tag_patterns: list) -> tuple:
    """Filter pre-fetched team rows whose metadata.tags match any patterns.

    Returns (named_aliases, unnamed_count).
    """

    affected: list = []
    unnamed_count = 0
    for team in teams:
        team_alias = team.team_alias or ""
        team_tags = _get_tags_from_metadata(team.metadata)
        if team_tags and any(
            RouteChecks._route_matches_wildcard_pattern(route=tag, pattern=pat)
            for tag in team_tags
            for pat in tag_patterns
        ):
            if team_alias:
                affected.append(team_alias)
            else:
                unnamed_count += 1
    return affected, unnamed_count


async def _find_affected_by_team_patterns(
    prisma_client: object,
    all_teams: list,
    team_patterns: list,
    existing_teams: list,
    existing_keys: list,
) -> tuple:
    """Filter pre-fetched teams by alias patterns, then fetch their keys.

    Returns (new_teams, new_keys, unnamed_keys_count).
    """

    new_teams: list = []
    matched_team_ids: list = []

    for team in all_teams:
        team_alias = team.team_alias or ""
        if team_alias and any(
            RouteChecks._route_matches_wildcard_pattern(route=team_alias, pattern=pat)
            for pat in team_patterns
        ):
            if team_alias not in existing_teams:
                new_teams.append(team_alias)
                matched_team_ids.append(str(team.team_id))

    new_keys: list = []
    unnamed_keys_count = 0
    if matched_team_ids:
        keys = await prisma_client.db.litellm_verificationtoken.find_many(  # type: ignore
            where={"team_id": {"in": matched_team_ids}},
            order={"created_at": "desc"}, take=MAX_POLICY_ESTIMATE_IMPACT_ROWS,
        )
        for key in keys:
            key_alias = key.key_alias or ""
            if key_alias:
                if key_alias not in existing_keys:
                    new_keys.append(key_alias)
            else:
                unnamed_keys_count += 1

    return new_teams, new_keys, unnamed_keys_count


async def _find_affected_keys_by_alias(
    prisma_client: object, key_patterns: list, existing_keys: list
) -> list:
    """Find keys whose alias matches the given patterns."""

    affected: list = []

    keys = await prisma_client.db.litellm_verificationtoken.find_many(  # type: ignore
        where=_build_alias_where("key_alias", key_patterns),
        order={"created_at": "desc"}, take=MAX_POLICY_ESTIMATE_IMPACT_ROWS,
    )
    for key in keys:
        key_alias = key.key_alias or ""
        if key_alias and any(
            RouteChecks._route_matches_wildcard_pattern(route=key_alias, pattern=pat)
            for pat in key_patterns
        ):
            if key_alias not in existing_keys:
                affected.append(key_alias)
    return affected


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
    force_sync: bool = Query(
        default=False,
        description="Force a DB sync before resolving. Default uses in-memory cache.",
    ),
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
        # Only sync from DB when explicitly requested; otherwise use in-memory cache
        if force_sync:
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
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        # If global scope, everything is affected — not useful to enumerate
        if request.scope == "*":
            return AttachmentImpactResponse(
                affected_keys_count=-1,
                affected_teams_count=-1,
                sample_keys=["(global scope — affects all keys)"],
                sample_teams=["(global scope — affects all teams)"],
            )

        affected_keys: list = []
        affected_teams: list = []
        unnamed_keys = 0
        unnamed_teams = 0

        tag_patterns = request.tags or []
        team_patterns = request.teams or []

        # Fetch teams once — reused by both tag-based and alias-based lookups
        all_teams: list = []
        if tag_patterns or team_patterns:
            all_teams = await _fetch_all_teams(prisma_client)

        # Tag-based impact
        if tag_patterns:
            keys = await prisma_client.db.litellm_verificationtoken.find_many(  # type: ignore
                where={}, order={"created_at": "desc"},
                take=MAX_POLICY_ESTIMATE_IMPACT_ROWS,
            )
            affected_keys, unnamed_keys = _filter_keys_by_tags(keys, tag_patterns)
            affected_teams, unnamed_teams = _filter_teams_by_tags(
                all_teams, tag_patterns,
            )

        # Team-based impact (alias matching + keys belonging to those teams)
        if team_patterns:
            new_teams, new_keys, new_unnamed = await _find_affected_by_team_patterns(
                prisma_client, all_teams, team_patterns,
                affected_teams, affected_keys,
            )
            affected_teams.extend(new_teams)
            affected_keys.extend(new_keys)
            unnamed_keys += new_unnamed

        # Key-based impact (direct alias matching)
        key_patterns = request.keys or []
        if key_patterns:
            new_keys = await _find_affected_keys_by_alias(
                prisma_client, key_patterns, affected_keys,
            )
            affected_keys.extend(new_keys)

        return AttachmentImpactResponse(
            affected_keys_count=len(affected_keys) + unnamed_keys,
            affected_teams_count=len(affected_teams) + unnamed_teams,
            unnamed_keys_count=unnamed_keys,
            unnamed_teams_count=unnamed_teams,
            sample_keys=affected_keys[:10],
            sample_teams=affected_teams[:10],
        )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error estimating attachment impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))
