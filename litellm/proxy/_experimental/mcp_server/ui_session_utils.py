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

EffectiveAuthContexts: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[Sequence[UserAPIKeyAuth]]]
TeamObjectPermission: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[LiteLLM_ObjectPermissionTable | None]]
OwnObjectPermission: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[LiteLLM_ObjectPermissionTable | None]]
AdmittedContext: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[UserAPIKeyAuth | None]]
ActingUser: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[UserAPIKeyAuth]]
GrantedToolsetIds: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[frozenset[str]]]


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

    A toolset narrowing is never applied to the admitted subject by rewriting its ``object_permission``:
    it resolves per grant source and a team source deliberately carries none of the caller's own grants,
    so the rewrite would evaporate on every team-granted server. The route pins ``mcp_toolset_id``
    instead, which every source's grant is intersected with."""

    if not is_ui_session_credential(user_api_key_auth):
        return user_api_key_auth
    from litellm.proxy.management_endpoints.common_utils import user_api_key_has_admin_view

    if user_api_key_has_admin_view(user_api_key_auth):
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


def is_keyless_mcp_subject(user_api_key_auth: UserAPIKeyAuth) -> bool:
    """A principal with no virtual key to declare MCP access on: the dashboard's own session token or a
    gateway-admitted user. Its grants are resolved per source, never through a key row."""

    return is_ui_session_credential(user_api_key_auth) or user_api_key_auth.mcp_admitted_user_subject is True


async def toolset_grant_contexts(
    user_api_key_auth: UserAPIKeyAuth,
    admitted_context: AdmittedContext = admitted_user_context,
    admitted_sources: EffectiveAuthContexts | None = None,
) -> Sequence[UserAPIKeyAuth]:
    """The grant sources a toolset is looked up through. A virtual key is its own single source. A keyless
    subject fans out exactly as the aggregate /mcp resolution does: its own user row plus every team whose
    live roster still lists it, so a membership that only survives in the user's cached team list grants
    nothing."""

    if not is_keyless_mcp_subject(user_api_key_auth):
        return (user_api_key_auth,)
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    load_sources: Final = admitted_sources or MCPRequestHandler.admitted_subject_sources
    acting: Final = await admitted_context(user_api_key_auth)
    return tuple(await load_sources(acting if acting is not None else user_api_key_auth))


async def _own_toolset_ids(
    context: UserAPIKeyAuth,
    load_own_permission: OwnObjectPermission,
) -> Sequence[str] | None:
    """The source's own toolsets, or None when it declares no MCP grant of its own. A source that names an
    ``object_permission_id`` is a known restriction even when the row is unhydrated, unreadable or gone,
    so it is loaded rather than read as unrestricted, and grants nothing when it cannot be read."""
    if context.object_permission is None and not context.object_permission_id:
        return None
    try:
        own: Final = await load_own_permission(context)
    except Exception as exc:  # noqa: BLE001  # a named but unreadable own grant must deny, not widen to the team
        verbose_logger.warning(
            "MCP toolset grants: object permission %s unreadable, granting nothing through it: %s",
            context.object_permission_id,
            exc,
        )
        return ()
    if own is None:
        return ()
    if not _restricts_mcp(own):
        return None
    return own.mcp_toolsets or ()


async def _inherited_toolset_ids(
    context: UserAPIKeyAuth,
    load_team_permission: TeamObjectPermission,
) -> Sequence[str]:
    try:
        team: Final = await load_team_permission(context)
    except Exception as exc:  # noqa: BLE001  # an unreadable team grants nothing through this source and must not fail the caller's other sources
        verbose_logger.warning(
            "MCP toolset grants: team %s unreadable, inheriting nothing from it: %s",
            context.team_id,
            exc,
        )
        return ()
    return () if team is None else (team.mcp_toolsets or ())


async def _context_toolset_ids(
    context: UserAPIKeyAuth,
    inherits_team: bool,
    load_team_permission: TeamObjectPermission,
    load_own_permission: OwnObjectPermission,
) -> Sequence[str]:
    own: Final = await _own_toolset_ids(context, load_own_permission)
    if own is not None:
        return own
    if not inherits_team or not context.team_id:
        return ()
    return await _inherited_toolset_ids(context, load_team_permission)


async def granted_toolset_ids(
    user_api_key_auth: UserAPIKeyAuth,
    effective_contexts: EffectiveAuthContexts = toolset_grant_contexts,
    team_object_permission: TeamObjectPermission | None = None,
    require_key_access: bool | None = None,
    own_object_permission: OwnObjectPermission | None = None,
) -> frozenset[str]:
    """Toolset ids the principal holds, resolved per grant source with the key/team rule the aggregate
    /mcp listing applies: a source that declares any MCP grant of its own is scoped to its own toolsets and
    never reads its team, one that declares none inherits its team's, except a virtual key under
    ``require_key_mcp_access_defined``, which inherits nothing. A keyless subject's team sources always
    inherit. A team that cannot be read contributes nothing while every other source still counts, and an
    own grant that is named but cannot be read grants nothing. No grant anywhere yields the empty set."""
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )
    from litellm.proxy.proxy_server import general_settings

    require: Final = (
        require_key_access
        if require_key_access is not None
        else bool(
            general_settings.get(  # pyright: ignore[reportUnknownArgumentType]  # general_settings is an untyped dict; truthiness must match the /mcp path's read of this flag
                "require_key_mcp_access_defined", False
            )
        )
    )
    inherits_team: Final = is_keyless_mcp_subject(user_api_key_auth) or not require
    load_team_permission: Final = team_object_permission or MCPRequestHandler.team_object_permission
    load_own_permission: Final = own_object_permission or MCPRequestHandler.key_object_permission_hydrated
    contexts: Final = await effective_contexts(user_api_key_auth)
    per_context: Final = await asyncio.gather(
        *(
            _context_toolset_ids(context, inherits_team, load_team_permission, load_own_permission)
            for context in contexts
        )
    )
    return frozenset(chain.from_iterable(per_context))
