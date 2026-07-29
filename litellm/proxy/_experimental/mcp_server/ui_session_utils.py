"""Helpers to resolve real team contexts for UI session tokens."""

from __future__ import annotations

from typing import List

from fastapi import HTTPException

from litellm._logging import verbose_logger
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import UserAPIKeyAuth


def clone_user_api_key_auth_with_team(
    user_api_key_auth: UserAPIKeyAuth,
    team_id: str,
) -> UserAPIKeyAuth:
    """Return a deep copy of the auth context with a different team id."""

    try:
        cloned_auth = user_api_key_auth.model_copy()
    except AttributeError:
        cloned_auth = user_api_key_auth.copy()  # type: ignore[attr-defined]
    cloned_auth.team_id = team_id
    return cloned_auth


def is_ui_session_credential(user_api_key_auth: UserAPIKeyAuth) -> bool:
    """Whether the caller is the dashboard's SSO-minted session token acting as its user,
    the only credential shape allowed to widen a request to the owning user's identity."""

    return user_api_key_auth.team_id == UI_SESSION_TOKEN_TEAM_ID and bool(user_api_key_auth.user_id)


async def resolve_ui_session_team_ids(
    user_api_key_auth: UserAPIKeyAuth,
) -> List[str]:
    """Resolve the real team ids backing a UI session token."""

    if not is_ui_session_credential(user_api_key_auth):
        return []

    from litellm.proxy.auth.auth_checks import get_user_object
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    if prisma_client is None:
        verbose_logger.debug("Cannot resolve UI session team ids without DB access")
        return []

    try:
        user_obj = await get_user_object(
            user_id=user_api_key_auth.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            parent_otel_span=user_api_key_auth.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        verbose_logger.warning(
            "Failed to load teams for UI session token user.",
            exc,
        )
        return []

    if user_obj is None or not user_obj.teams:
        return []

    resolved_team_ids: List[str] = []
    for team_id in user_obj.teams:
        if team_id and team_id not in resolved_team_ids:
            resolved_team_ids.append(team_id)
    return resolved_team_ids


async def _admitted_user_context(user_api_key_auth: UserAPIKeyAuth) -> UserAPIKeyAuth | None:
    """The user-identity context for a dashboard session: the same admitted-subject auth a
    gateway OAuth session for this user resolves with, carrying the user row's own object
    permission. None for any other credential (a caller-passed key is never widened) and on
    reload failure (falls back to team contexts only)."""

    if not is_ui_session_credential(user_api_key_auth) or user_api_key_auth.user_id is None:
        return None
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    try:
        return await MCPRequestHandler._reload_admitted_user(user_api_key_auth.user_id)
    except HTTPException:
        return None


async def build_effective_auth_contexts(
    user_api_key_auth: UserAPIKeyAuth,
) -> List[UserAPIKeyAuth]:
    """Return auth contexts that reflect the actual teams for UI session tokens."""

    resolved_team_ids = await resolve_ui_session_team_ids(user_api_key_auth)
    team_contexts = (
        [clone_user_api_key_auth_with_team(user_api_key_auth, team_id) for team_id in resolved_team_ids]
        if resolved_team_ids
        else [user_api_key_auth]
    )
    admitted_context = await _admitted_user_context(user_api_key_auth)
    if admitted_context is None:
        return team_contexts
    return [*team_contexts, admitted_context]
