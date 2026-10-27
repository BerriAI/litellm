"""Resolve dashboard model requests through the signed-in user's existing grants."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import (
    LiteLLM_TeamMembership,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    can_key_call_model,
    can_team_access_model,
    can_user_call_model,
    check_team_member_model_access,
)
from litellm.proxy.auth.model_access_denied import model_access_denied_client_message
from litellm.proxy.auth.resolvers.grants import (
    GrantResolver,
    ResolvedGrants,
    UserLookup,
    user_models,
)
from litellm.proxy.auth.team_grants import team_grants
from litellm.proxy.mcp_server.ui_session_utils import is_ui_session_credential
from litellm.types.proxy.management_endpoints.model_management_endpoints import (
    ModelGroupTeam,
)

if TYPE_CHECKING:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.router import Router


def is_model_ui_session(auth: UserAPIKeyAuth) -> bool:
    return (
        is_ui_session_credential(auth)
        and auth.agent_id is None
        and auth.user_role in (None, LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
    )


@dataclass(frozen=True, slots=True)
class TeamModelSource:
    team: LiteLLM_TeamTableCachedObj
    membership: LiteLLM_TeamMembership | None
    auth: UserAPIKeyAuth


@dataclass(frozen=True, slots=True)
class UISessionModelSources:
    auth: UserAPIKeyAuth
    user: LiteLLM_UserTable
    teams: tuple[TeamModelSource, ...]
    router: Router | None
    prisma: PrismaClient
    cache: UserApiKeyCache
    logging: ProxyLogging

    async def directly_allows(self, models: Sequence[str]) -> bool:
        try:
            for model in models:
                await can_key_call_model(model, None, self.auth, self.router, self.prisma)
                await can_user_call_model(model=model, user_object=self.user, llm_router=self.router)
        except ProxyException:
            return False
        return True

    async def team_allows(self, source: TeamModelSource, models: Sequence[str]) -> bool:
        if source.team.blocked:
            return False
        try:
            for model in models:
                await can_team_access_model(
                    model=model,
                    team_object=source.team,
                    llm_router=self.router,
                    team_model_aliases=source.auth.team_model_aliases,
                    prisma_client=self.prisma,
                )
                await check_team_member_model_access(
                    model=model,
                    team_object=source.team,
                    valid_token=source.auth,
                    llm_router=self.router,
                    prisma_client=self.prisma,
                    user_api_key_cache=self.cache,
                    proxy_logging_obj=self.logging,
                    team_membership=source.membership,
                    team_membership_loaded=True,
                )
        except ProxyException:
            return False
        return True

    async def available_teams(self, model: str) -> tuple[ModelGroupTeam, ...] | None:
        if await self.directly_allows((model,)):
            return ()
        allowed: Final = tuple(
            [
                ModelGroupTeam(team_id=source.team.team_id, team_alias=source.team.team_alias)
                for source in self.teams
                if await self.team_allows(source, (model,))
            ]
        )
        return allowed or None

    async def for_request(self, models: Sequence[str], selected_team_id: str | None) -> UserAPIKeyAuth:
        if selected_team_id is None and await self.directly_allows(models):
            return self.auth
        allowed: Final = tuple([source for source in self.teams if await self.team_allows(source, models)])
        if selected_team_id is not None:
            selected: Final = next(
                (source for source in allowed if source.team.team_id == selected_team_id),
                None,
            )
            if selected is None:
                raise ProxyException(
                    message="The selected team does not grant access to this model.",
                    type=ProxyErrorTypes.team_model_access_denied,
                    param="x-litellm-team-id",
                    code=403,
                )
            return selected.auth
        if len(allowed) == 1:
            return allowed[0].auth
        if len(allowed) > 1:
            raise ProxyException(
                message="Choose a team for this model before sending the request.",
                type=ProxyErrorTypes.team_selection_required,
                param="x-litellm-team-id",
                code=409,
            )
        raise ProxyException(
            message=model_access_denied_client_message(model=list(models)),
            type=ProxyErrorTypes.key_model_access_denied,
            param="model",
            code=403,
        )


async def load_ui_session_model_sources(
    auth: UserAPIKeyAuth,
    router: Router | None,
    prisma: PrismaClient | None,
    cache: UserApiKeyCache,
    logging: ProxyLogging,
) -> UISessionModelSources | None:
    if not is_model_ui_session(auth):
        return None
    if prisma is None:
        raise ProxyException(
            message="Unable to resolve the signed-in user's model access.",
            type=ProxyErrorTypes.no_db_connection,
            param=None,
            code=503,
        )
    resolver: Final = GrantResolver(prisma, cache, parent_otel_span=auth.parent_otel_span, proxy_logging_obj=logging)
    identity: Final = await resolver.resolve(UserLookup(auth.user_id), team_id=None)
    if not isinstance(identity, ResolvedGrants) or identity.user_object is None:
        raise ProxyException(
            message="Unable to resolve the signed-in user's model access.",
            type=ProxyErrorTypes.auth_error,
            param=None,
            code=403,
        )
    user: Final = identity.user_object
    if user.user_role not in (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY):
        return None
    # Login copies the user model list into the UI key. Refresh that snapshot
    # without changing the session's identity, restrictions, or budget.
    live_auth: Final = auth.model_copy(update={"models": list(user_models(user))})
    sources: Final = tuple(
        [
            await _team_model_source(resolver, auth, user, team_id)
            for team_id in dict.fromkeys(user.teams)
            if team_id != UI_SESSION_TOKEN_TEAM_ID
        ]
    )
    return UISessionModelSources(
        live_auth, user, tuple(source for source in sources if source is not None), router, prisma, cache, logging
    )


async def _team_model_source(
    resolver: GrantResolver, auth: UserAPIKeyAuth, user: LiteLLM_UserTable, team_id: str
) -> TeamModelSource | None:
    grant: Final = await resolver.resolve(UserLookup(user.user_id), team_id=team_id)
    if not isinstance(grant, ResolvedGrants) or grant.team_object is None:
        return None
    team: Final = grant.team_object
    scoped: Final = auth.model_copy(
        update={
            **team_grants(team, grant.team_membership, user.user_id),
            "team_id": team.team_id,
            "org_id": auth.org_id or team.organization_id,
            "models": [],
        }
    )
    return TeamModelSource(team, grant.team_membership, scoped)
