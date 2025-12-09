"""Helpers to resolve real team contexts for UI session tokens."""

from __future__ import annotations

from typing import List

from litellm._logging import verbose_logger
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import UserAPIKeyAuth


def clone_user_api_key_auth_with_team(
    user_api_key_auth: UserAPIKeyAuth,
    team_id: str,
) -> UserAPIKeyAuth:
    """Return a deep copy of the auth context with a different team id."""

    try:
        cloned_auth = user_api_key_auth.model_copy(deep=True)
    except AttributeError:
        cloned_auth = user_api_key_auth.copy(deep=True)  # type: ignore[attr-defined]
    cloned_auth.team_id = team_id
    return cloned_auth


async def resolve_ui_session_team_ids(
    user_api_key_auth: UserAPIKeyAuth,
) -> List[str]:
    """Resolve the real team ids backing a UI session token."""

    if (
        user_api_key_auth.team_id != UI_SESSION_TOKEN_TEAM_ID
        or not user_api_key_auth.user_id
    ):
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


async def build_effective_auth_contexts(
    user_api_key_auth: UserAPIKeyAuth,
) -> List[UserAPIKeyAuth]:
    """Return auth contexts that reflect the actual teams for UI session tokens."""

    resolved_team_ids = await resolve_ui_session_team_ids(user_api_key_auth)
    if resolved_team_ids:
        return [
            clone_user_api_key_auth_with_team(user_api_key_auth, team_id)
            for team_id in resolved_team_ids
        ]
    return [user_api_key_auth]
