"""Helpers to resolve the identity a dashboard UI session token acts as."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from itertools import chain
from typing import Final, TypeAlias

from fastapi import HTTPException

from litellm._logging import verbose_logger
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth

EffectiveAuthContexts: TypeAlias = Callable[
    [UserAPIKeyAuth], Awaitable[Sequence[UserAPIKeyAuth]]  # mutable-ok: Callable parameter syntax
]
TeamObjectPermission: TypeAlias = Callable[
    [UserAPIKeyAuth], Awaitable[LiteLLM_ObjectPermissionTable | None]  # mutable-ok: Callable parameter syntax
]


def clone_user_api_key_auth_with_team(
    user_api_key_auth: UserAPIKeyAuth,
    team_id: str,
) -> UserAPIKeyAuth:
    """Return a deep copy of the auth context with a different team id."""

    try:
        cloned_auth = user_api_key_auth.model_copy()
    except AttributeError:
        cloned_auth = user_api_key_auth.copy()
    cloned_auth.team_id = team_id
    return cloned_auth


def is_ui_session_credential(user_api_key_auth: UserAPIKeyAuth) -> bool:
    """Whether the caller is the dashboard's SSO-minted session token acting as its user,
    the only credential shape allowed to widen a request to the owning user's identity."""

    return user_api_key_auth.team_id == UI_SESSION_TOKEN_TEAM_ID and bool(user_api_key_auth.user_id)


async def resolve_ui_session_team_ids(
    user_api_key_auth: UserAPIKeyAuth,
) -> list[str]:
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
        user_obj: Final = await get_user_object(
            user_id=user_api_key_auth.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            check_db_only=user_api_key_auth.requires_fresh_policy,
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

    resolved_team_ids: Final[list[str]] = []
    for team_id in user_obj.teams:
        if team_id and team_id not in resolved_team_ids:
            resolved_team_ids.append(team_id)
    return resolved_team_ids


async def admitted_user_context(user_api_key_auth: UserAPIKeyAuth) -> UserAPIKeyAuth | None:
    """THE owner of "resolve this dashboard session's user identity": the same admitted-subject auth a
    gateway OAuth session for this user resolves with, carrying the user row's own object permission,
    on this request's tracing span. None for any other credential (a caller-passed key is never
    widened) and on reload failure, which every caller reads as "no user-level identity available"."""

    user_id: Final = user_api_key_auth.user_id
    if not is_ui_session_credential(user_api_key_auth) or user_id is None:
        return None
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    try:
        admitted: Final = await MCPRequestHandler.reload_admitted_user(
            user_id, requires_fresh_policy=user_api_key_auth.requires_fresh_policy
        )
    except HTTPException as e:
        verbose_logger.warning("MCP dashboard session: admitted-subject reload failed for %s: %s", user_id, e.detail)
        return None
    return admitted.model_copy(update={"parent_otel_span": user_api_key_auth.parent_otel_span})


async def acting_user_auth(user_api_key_auth: UserAPIKeyAuth) -> UserAPIKeyAuth:
    """The principal acting-as-user MCP routes resolve permissions with. A non-admin dashboard
    session acts as the admitted subject, the same identity a gateway session resolves with, so
    server reachability, per-source tool ceilings, rate limits, and billing bind identically on
    both surfaces. An admin session keeps its operator view and any caller-passed credential is
    returned unchanged, never widened.

    Do not combine this with a narrowing that rewrites a single credential's ``object_permission``
    (toolset scope): the admitted subject resolves per grant source and a team source deliberately
    carries none of the caller's own grants, so the narrowing would silently evaporate on every
    team-granted server. A request carrying such a scope keeps the caller's own credential."""

    if not is_ui_session_credential(user_api_key_auth):
        return user_api_key_auth
    from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view

    if _user_has_admin_view(user_api_key_auth):
        return user_api_key_auth
    admitted: Final = await admitted_user_context(user_api_key_auth)
    return admitted if admitted is not None else user_api_key_auth


async def build_effective_auth_contexts(
    user_api_key_auth: UserAPIKeyAuth,
) -> list[UserAPIKeyAuth]:
    """Every auth context a management or listing surface must resolve a UI session token through:
    one per real team backing the session, plus the session user's own admitted identity, so a grant
    made directly to the user row is as visible to the dashboard as it is to a gateway session."""

    resolved_team_ids: Final = await resolve_ui_session_team_ids(user_api_key_auth)
    team_contexts: Final = (
        [clone_user_api_key_auth_with_team(user_api_key_auth, team_id) for team_id in resolved_team_ids]
        if resolved_team_ids
        else [user_api_key_auth]
    )
    admitted_context: Final = await admitted_user_context(user_api_key_auth)
    if admitted_context is None:
        return team_contexts
    return [*team_contexts, admitted_context]


async def can_access_mcp_server(
    user_api_key_auth: UserAPIKeyAuth,
    server_id: str,
    allowed_servers: Callable[[UserAPIKeyAuth], Awaitable[list[str]]],
) -> bool:
    """Resolve server access through the same credential contexts as MCP management."""
    for context in await build_effective_auth_contexts(user_api_key_auth):
        if server_id in await allowed_servers(context):
            return True
    return False


def _restricts_mcp(permission: LiteLLM_ObjectPermissionTable | None) -> bool:
    return permission is not None and bool(
        permission.mcp_servers
        or permission.mcp_toolsets
        or permission.mcp_tool_permissions
        or permission.mcp_access_groups
    )


def _context_toolset_ids(
    own: LiteLLM_ObjectPermissionTable | None,
    team: LiteLLM_ObjectPermissionTable | None,
    inherits_team: bool,
) -> Sequence[str]:
    if own is not None and _restricts_mcp(own):
        return own.mcp_toolsets or ()
    if not inherits_team or team is None:
        return ()
    return team.mcp_toolsets or ()


async def granted_toolset_ids(
    user_api_key_auth: UserAPIKeyAuth,
    effective_contexts: EffectiveAuthContexts = build_effective_auth_contexts,
    team_object_permission: TeamObjectPermission | None = None,
    require_key_access: bool | None = None,
) -> frozenset[str]:
    """Toolset ids the credential holds, resolved per effective context with the key/team rule the
    aggregate /mcp listing applies: a context that declares any MCP grant of its own is scoped to its own
    toolsets, one that declares none inherits its team's, except a virtual key under
    ``require_key_mcp_access_defined``, which inherits nothing. A dashboard session or gateway-admitted
    user has no key to declare access on, so its team contexts always inherit. No grant anywhere yields
    the empty set."""
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )
    from litellm.proxy.proxy_server import general_settings

    keyless: Final = is_ui_session_credential(user_api_key_auth) or user_api_key_auth.mcp_admitted_user_subject
    require: Final = (
        bool(general_settings.get("require_key_mcp_access_defined", False))
        if require_key_access is None
        else require_key_access
    )
    inherits_team: Final = keyless or not require
    load_team_permission: Final = team_object_permission or MCPRequestHandler.team_object_permission
    contexts: Final = await effective_contexts(user_api_key_auth)
    team_permissions: Final = await asyncio.gather(*(load_team_permission(context) for context in contexts))
    return frozenset(
        chain.from_iterable(
            _context_toolset_ids(context.object_permission, team, inherits_team)
            for context, team in zip(contexts, team_permissions, strict=True)
        )
    )
