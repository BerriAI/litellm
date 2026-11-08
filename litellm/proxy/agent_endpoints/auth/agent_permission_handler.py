"""
Agent Permission Handler for LiteLLM Proxy.

Handles agent permission checking for keys and teams using object_permission_id.
Follows the same pattern as MCP permission handling.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, assert_never

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError

from litellm._auth import agent_access, agent_grants, managed
from litellm._auth.failure import NativeAuthFailure
from litellm._logging import verbose_logger
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.agent_endpoints.auth.agent_access_groups import (
    CeilingResolver,
    resolve_agent_access_group_ceiling,
)
from litellm.proxy.agent_endpoints.auth.agent_caller import agent_caller_auth
from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_agent_policy
from litellm.proxy.mcp_server.ui_session_utils import build_effective_auth_contexts
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import AgentsRepository
from litellm.types.agents import AgentResponse


@dataclass(frozen=True, slots=True)
class UnrestrictedAgentAccess:
    """No agent grant exists on the key or its team, so every agent is reachable."""


@dataclass(frozen=True, slots=True)
class RestrictedAgentAccess:
    """Only ``agent_ids`` are reachable. An empty set denies every agent."""

    agent_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class AgentPolicyUnavailable:
    team_id: str


AgentAccess: TypeAlias = UnrestrictedAgentAccess | RestrictedAgentAccess


class AgentAccessLookup(Protocol):
    def __call__(self, user_api_key_auth: UserAPIKeyAuth | None, /, *, strict: bool) -> Awaitable[AgentAccess]: ...


_AGENT_GROUPS: Final = TypeAdapter(tuple[str, ...])

_DEFAULT_DENY_FLAG: Final = TypeAdapter(bool)


def agent_access_default_deny() -> bool:
    from litellm.proxy.proxy_server import general_settings_view

    try:
        return _DEFAULT_DENY_FLAG.validate_python(general_settings_view().get("agent_access_default_deny", False))
    except ValidationError:
        return True


def _to_stable_ids(agent_ids: frozenset[str]) -> frozenset[str]:
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    return frozenset(global_agent_registry.stable_agent_id(agent_id) for agent_id in agent_ids)


def _restricted_ids(access: AgentAccess) -> frozenset[str] | None:
    if isinstance(access, UnrestrictedAgentAccess):
        return None
    return _to_stable_ids(access.agent_ids)


def _project_agent_grants(decision: managed.ManagedDecision | agent_grants.Decision) -> AgentAccess:
    return UnrestrictedAgentAccess() if decision.grants is None else RestrictedAgentAccess(frozenset(decision.grants))


def _intersect_agent_access(key_access: AgentAccess, team_access: AgentAccess) -> AgentAccess:
    key_ids: Final = _restricted_ids(key_access)
    team_ids: Final = _restricted_ids(team_access)
    return _project_agent_grants(
        managed.decide(
            managed.IntersectGrants(
                tuple(key_ids) if key_ids is not None else None, tuple(team_ids) if team_ids is not None else None
            )
        )
    )


class AgentRequestHandler:
    """
    Class to handle agent permission checking, including:
    1. Key-level agent permissions
    2. Team-level agent permissions
    3. Agent access group resolution

    Follows the same inheritance logic as MCP:
    - If team has restrictions and key has restrictions: use intersection
    - If team has restrictions and key has none: inherit from team
    - If team has no restrictions: use key restrictions
    - If no restrictions: allow all agents unless agent_access_default_deny is enabled for a non-admin key
    """

    @staticmethod
    async def resolve_agent_access(
        user_api_key_auth: UserAPIKeyAuth | None = None,
        resolve_ceiling: CeilingResolver = resolve_agent_access_group_ceiling,
        *,
        strict: bool = False,
        default_deny: Callable[[], bool] = agent_access_default_deny,
        lookup_key_access: AgentAccessLookup | None = None,
        lookup_team_access: AgentAccessLookup | None = None,
    ) -> AgentAccess:
        """Agents the key may reach: key and team grants, intersected with the agent's access group ceiling
        and, for an agent key acting on behalf of an invoking user, with that user's team grants."""
        return await _read_resolved_agent_grants(
            user_api_key_auth,
            resolve_ceiling,
            default_deny,
            lookup_key_access,
            lookup_team_access,
            agent_grants.Resolve(
                managed=managed_agent_policy(user_api_key_auth) is not None,
                strict=strict,
                role=user_api_key_auth.user_role if user_api_key_auth is not None else None,
            ),
        )

    @staticmethod
    async def agent_caller_access(user_api_key_auth: UserAPIKeyAuth | None, *, strict: bool = False) -> AgentAccess:
        caller_auth: Final = agent_caller_auth(user_api_key_auth) if user_api_key_auth else None
        if caller_auth is None:
            return UnrestrictedAgentAccess()
        return await AgentRequestHandler._get_allowed_agents_for_team(caller_auth, strict=strict)

    @staticmethod
    async def resolve_key_team_agent_access(
        user_api_key_auth: UserAPIKeyAuth | None,
        *,
        strict: bool = False,
        lookup_key_access: AgentAccessLookup | None = None,
        lookup_team_access: AgentAccessLookup | None = None,
    ) -> AgentAccess:
        try:
            key_lookup: Final = (
                AgentRequestHandler.get_allowed_agents_for_key if lookup_key_access is None else lookup_key_access
            )
            team_lookup: Final = (
                AgentRequestHandler._get_allowed_agents_for_team if lookup_team_access is None else lookup_team_access
            )
            key_access: Final = await key_lookup(user_api_key_auth, strict=strict)
            team_access: Final = await team_lookup(user_api_key_auth, strict=strict)
        except NativeAuthFailure:
            raise
        except Exception as e:
            failure: Final = agent_grants.decide(agent_grants.Failure(strict, closed=False))
            if failure.action == "Unavailable":
                raise HTTPException(503, "Agent invocation policy is unavailable") from e
            verbose_logger.warning("Failed to get allowed agents: %s", e)
            return _project_agent_grants(failure)
        return _intersect_agent_access(key_access, team_access)

    @staticmethod
    async def _agent_access_group_ceiling(
        user_api_key_auth: UserAPIKeyAuth | None,
        resolve_ceiling: CeilingResolver,
        *,
        strict: bool = False,
    ) -> frozenset[str] | None:
        if user_api_key_auth is None or not user_api_key_auth.agent_id:
            return None
        ceiling: Final = (
            await resolve_agent_access_group_ceiling(user_api_key_auth.agent_id, check_db_only=True)
            if strict
            else await resolve_ceiling(user_api_key_auth.agent_id)
        )
        if ceiling is None:
            return None
        return _to_stable_ids(ceiling.agent_ids)

    @staticmethod
    async def is_agent_allowed(
        agent_id: str,
        user_api_key_auth: UserAPIKeyAuth | None = None,
        resolve_ceiling: CeilingResolver = resolve_agent_access_group_ceiling,
    ) -> bool:
        facts: Final = await read_agent_access_facts(agent_id, user_api_key_auth, resolve_ceiling)
        return agent_access.decide(facts) == "Allow"

    @staticmethod
    def _get_key_object_permission(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> LiteLLM_ObjectPermissionTable | None:
        """
        Get key object_permission - already loaded by get_key_object() in main auth flow.

        Note: object_permission is automatically populated when the key is fetched via
        get_key_object() in litellm/proxy/auth/auth_checks.py
        """
        if not user_api_key_auth:
            return None

        return user_api_key_auth.object_permission

    @staticmethod
    async def _get_team_object_permission(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> LiteLLM_ObjectPermissionTable | None:
        """
        Get team object_permission - automatically loaded by get_team_object() in main auth flow.

        Note: object_permission is automatically populated when the team is fetched via
        get_team_object() in litellm/proxy/auth/auth_checks.py
        """
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if not user_api_key_auth or not user_api_key_auth.team_id or not prisma_client:
            return None

        # Get the team object (which has object_permission already loaded)
        team_obj: Final[LiteLLM_TeamTable | None] = await get_team_object(
            team_id=user_api_key_auth.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_auth.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )

        if not team_obj:
            return None

        return team_obj.object_permission

    @staticmethod
    async def get_allowed_agents_for_key(
        user_api_key_auth: UserAPIKeyAuth | None = None,
        *,
        strict: bool = False,
    ) -> AgentAccess:
        """
        Get allowed agents for a key.

        1. First checks native key-level agent permissions (object_permission)
        2. Also includes agents from key's access_group_ids (unified access groups)

        A key that declares agents or access groups is restricted even when those
        declarations resolve to nothing, so an emptied or deleted access group denies
        rather than opening the key up. Lookup failures still propagate to the caller,
        which keeps them fail-open.

        Note: object_permission is already loaded by get_key_object() in main auth flow.
        """
        if user_api_key_auth is None:
            return UnrestrictedAgentAccess()

        try:
            # 1. Get agents from object_permission (native permissions)
            key_object_permission: Final = AgentRequestHandler._get_key_object_permission(user_api_key_auth)
            direct_agents: Final = tuple(
                key_object_permission.agents or () if key_object_permission is not None else ()
            )
            declared_access_groups: Final = tuple(
                key_object_permission.agent_access_groups or () if key_object_permission is not None else ()
            )
            # 2. Fallback: get agent IDs from key's access_group_ids (unified access groups)
            key_access_group_ids: Final = tuple(user_api_key_auth.access_group_ids or ())

            return await _read_declared_grants(
                managed.DeclaredGrants(direct_agents, declared_access_groups, key_access_group_ids), strict
            )
        except NativeAuthFailure:
            raise
        except Exception as e:
            if strict:
                raise HTTPException(503, "Agent invocation policy is unavailable") from e
            verbose_logger.warning("Failed to get allowed agents for key: %s", e)
            return UnrestrictedAgentAccess()

    @staticmethod
    async def _personal_key_owner_team_agent_access(user_api_key_auth: UserAPIKeyAuth, *, strict: bool) -> AgentAccess:
        """Union owner-team access; teams without a rule add nothing once any team declares one."""
        from litellm.proxy.auth.auth_checks import personal_key_owner_teams_with_status
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        try:
            owner_teams: Final = await personal_key_owner_teams_with_status(
                valid_token=user_api_key_auth,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
                check_db_only=strict,
            )
        except NativeAuthFailure:
            raise
        except Exception as e:  # noqa: BLE001  # fail closed: an owner whose teams cannot be read grants nothing
            if strict:
                raise HTTPException(503, "Agent invocation policy is unavailable") from e
            verbose_logger.warning("Failed to load owner teams for personal key agent access: %s", e)
            return RestrictedAgentAccess(frozenset())
        team_results: Final = await asyncio.gather(
            *(AgentRequestHandler._owner_team_agent_grants(team, strict=strict) for team in owner_teams.teams)
        )
        withheld_results: Final = await asyncio.gather(
            *(AgentRequestHandler._owner_team_agent_grants(team, strict=strict) for team in owner_teams.withheld)
        )
        decision: Final = agent_grants.decide(
            agent_grants.OwnerUnion(
                tuple(_agent_grant_source(None, access) for access in team_results),
                tuple(_agent_grant_source(None, access) for access in withheld_results),
                owner_teams.unreadable,
            )
        )
        if decision.action == "Unavailable":
            raise HTTPException(503, "Agent invocation policy is unavailable")
        return _project_agent_grants(decision)

    @staticmethod
    async def personal_key_agent_billing_auth(
        auth: UserAPIKeyAuth,
        agent_id: str,
        *,
        strict: bool = False,
    ) -> UserAPIKeyAuth:
        try:
            from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
            from litellm.proxy.auth.auth_checks import (
                personal_key_inherits_team_grants,
                personal_key_owner_teams,
            )
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            if not personal_key_inherits_team_grants(auth):
                return auth
            teams: Final = await personal_key_owner_teams(
                valid_token=auth,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
                check_db_only=strict,
            )
            accesses: Final = await asyncio.gather(
                *(AgentRequestHandler._owner_team_agent_grants(team, strict=strict) for team in teams)
            )
            stable_agent_id: Final = global_agent_registry.stable_agent_id(agent_id)

            selection: Final = agent_grants.decide(
                agent_grants.BillingTeam(
                    tuple(
                        agent_grants.Source(
                            team.team_id,
                            tuple(_to_stable_ids(access.agent_ids))
                            if isinstance(access, RestrictedAgentAccess)
                            else None,
                            isinstance(access, AgentPolicyUnavailable),
                        )
                        for team, access in zip(teams, accesses, strict=True)
                    ),
                    stable_agent_id,
                )
            )
            if selection.team_id is None:
                verbose_logger.warning("No owner team grants agent %s; billing against the personal key", agent_id)
                return auth
            team: Final = next(team for team in teams if team.team_id == selection.team_id)
        except NativeAuthFailure:
            raise
        except Exception as e:  # noqa: BLE001  # attribution must never fail an authorized call
            verbose_logger.warning("Failed to attribute agent %s spend to an owner team: %s", agent_id, e)
            return auth
        return await AgentRequestHandler._personal_key_agent_billing_auth_for_team(
            auth,
            agent_id,
            team,
            strict=strict,
        )

    @staticmethod
    async def _personal_key_agent_billing_auth_for_team(
        auth: UserAPIKeyAuth,
        agent_id: str,
        team: LiteLLM_TeamTable,
        *,
        strict: bool,
    ) -> UserAPIKeyAuth:
        from litellm.proxy.auth.auth_checks import get_team_membership
        from litellm.proxy.auth.team_grants import team_grant_auth_update, team_grant_billing_auth_update
        from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

        try:
            membership: Final = (
                await get_team_membership(
                    user_id=auth.user_id,
                    team_id=team.team_id,
                    prisma_client=prisma_client,
                    user_api_key_cache=user_api_key_cache,
                    proxy_logging_obj=proxy_logging_obj,
                    check_db_only=strict,
                )
                if auth.user_id and prisma_client is not None
                else None
            )
            team_auth_update: Final = await team_grant_billing_auth_update(auth, team, membership)
        except NativeAuthFailure:
            raise
        except Exception as e:  # noqa: BLE001  # a chosen team's controls must survive billing lookup failures
            verbose_logger.warning(
                "Failed to load billing controls for agent %s from owner team %s: %s",
                agent_id,
                team.team_id,
                e,
            )
            return auth.model_copy(update=team_grant_auth_update(team, None, auth.user_id, key_org_id=auth.org_id))
        return auth.model_copy(update=team_auth_update)

    @staticmethod
    async def _owner_team_agent_grants(
        team: LiteLLM_TeamTable,
        *,
        strict: bool,
    ) -> AgentAccess | AgentPolicyUnavailable:
        decision: Final = agent_grants.decide(
            agent_grants.OwnerTeam(
                strict,
                team.object_permission_id is not None,
                team.object_permission is not None,
            )
        )
        if decision.action == "Unavailable":
            return AgentPolicyUnavailable(team_id=team.team_id)
        if decision.action == "Complete":
            return _project_agent_grants(decision)
        try:
            return await AgentRequestHandler._team_agent_grants(team, strict=strict)
        except NativeAuthFailure:
            raise
        except Exception as e:  # noqa: BLE001  # per-team isolation: a team whose grants cannot be read grants nothing
            failure: Final = agent_grants.decide(agent_grants.Failure(strict, closed=True))
            if failure.action == "Unavailable":
                return AgentPolicyUnavailable(team_id=team.team_id)
            verbose_logger.warning("Failed to get allowed agents for owner team %s: %s", team.team_id, e)
            return _project_agent_grants(failure)

    @staticmethod
    async def _team_agent_grants(team: LiteLLM_TeamTable, *, strict: bool) -> AgentAccess:
        """The agents one loaded team grants: its object_permission agents plus its access groups.
        Declared-but-empty grants stay restricted; a team with no grant at all does not restrict."""
        # 1. Get agents from object_permission (native permissions)
        object_permissions: Final = team.object_permission
        direct_agents: Final = tuple(object_permissions.agents or () if object_permissions is not None else ())
        declared_access_groups: Final = tuple(
            object_permissions.agent_access_groups or () if object_permissions is not None else ()
        )
        # 2. Also include agents from team's access_group_ids (unified access groups)
        team_access_group_ids: Final = tuple(team.access_group_ids or ())

        return await _read_declared_grants(
            managed.DeclaredGrants(direct_agents, declared_access_groups, team_access_group_ids), strict
        )

    @staticmethod
    async def _get_allowed_agents_for_team(
        user_api_key_auth: UserAPIKeyAuth | None = None,
        *,
        strict: bool = False,
    ) -> AgentAccess:
        """
        Get allowed agents for a team.

        1. First checks native team-level agent permissions (object_permission)
        2. Also includes agents from team's access_group_ids (unified access groups)

        Fetches the team object once and reuses it for both permission sources.
        Declared-but-empty grants stay restricted; see `get_allowed_agents_for_key`.
        """
        facts: Final = agent_grants.TeamLookup(
            subject=user_api_key_auth is not None,
            team_id=user_api_key_auth.team_id if user_api_key_auth is not None else None,
            database=True,
            strict=strict,
        )
        initial: Final = agent_grants.decide(facts)
        if initial.action == "Complete":
            return _project_agent_grants(initial)
        assert user_api_key_auth is not None
        if initial.action == "ReadPersonalEligibility":
            from litellm.proxy.auth.auth_checks import personal_key_inherits_team_grants

            personal: Final = agent_grants.decide(
                replace(
                    facts,
                    personal=personal_key_inherits_team_grants(user_api_key_auth),
                )
            )
            if personal.action == "Complete":
                return _project_agent_grants(personal)
            return await AgentRequestHandler._personal_key_owner_team_agent_access(user_api_key_auth, strict=strict)

        try:
            from litellm.proxy.auth.auth_checks import get_team_object
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            database: Final = agent_grants.decide(replace(facts, database=bool(prisma_client)))
            if database.action == "Complete":
                return _project_agent_grants(database)

            # Fetch the team object once for both permission sources
            assert user_api_key_auth.team_id is not None
            team_obj: Final[LiteLLM_TeamTable | None] = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
                check_db_only=strict,
            )

            loaded: Final = agent_grants.decide(replace(facts, loaded=True, row_present=team_obj is not None))  # pyright: ignore[reportUnnecessaryComparison]  # legacy adapters may return no row
            if loaded.action == "Complete":
                return _project_agent_grants(loaded)
            assert team_obj is not None

            return await AgentRequestHandler._team_agent_grants(team_obj, strict=strict)
        except NativeAuthFailure:
            raise
        except Exception as e:
            failure: Final = agent_grants.decide(agent_grants.Failure(strict, closed=False))
            if failure.action == "Unavailable":
                raise HTTPException(503, "Agent invocation policy is unavailable") from e
            # litellm-dashboard is the default UI team and will never have agents;
            # skip noisy warnings for it.
            if user_api_key_auth.team_id != UI_TEAM_ID:
                verbose_logger.warning("Failed to get allowed agents for team: %s", e)
            return _project_agent_grants(failure)

    @staticmethod
    def _get_config_agent_ids_for_access_groups(
        config_agents: Sequence[AgentResponse], access_groups: Sequence[str]
    ) -> set[str]:
        """
        Helper to get agent_ids from config-loaded agents that match any of the given access groups.
        """
        decision: Final = agent_grants.decide(
            agent_grants.GroupMatches(
                tuple(
                    agent_grants.ConfiguredAgent(
                        agent.agent_id, _AGENT_GROUPS.validate_python(getattr(agent, "agent_access_groups", None) or ())
                    )
                    for agent in config_agents
                ),
                tuple(access_groups),
            )
        )
        return set(decision.grants or ())

    @staticmethod
    async def _get_db_agent_ids_for_access_groups(
        prisma_client: PrismaClient | None, access_groups: Sequence[str], *, check_db_only: bool = False
    ) -> set[str]:
        """
        Helper to get agent_ids from DB agents that match any of the given access groups.

        Query failures propagate so the caller can tell "this group is empty" (deny)
        apart from "the lookup failed" (fail-open).
        """
        if not access_groups or prisma_client is None:
            return set()

        agents: Final = await AgentsRepository(prisma_client, use_writer=check_db_only).table.find_many(
            where={"agent_access_groups": {"hasSome": access_groups}}
        )
        return {agent.agent_id for agent in agents}

    @staticmethod
    async def _get_unified_access_group_agents(
        access_group_ids: Sequence[str], *, check_db_only: bool = False
    ) -> list[str]:
        """
        Resolve unified access group ids to agent IDs.
        """
        from litellm.proxy.auth.auth_checks import get_agent_ids_from_access_groups

        return await get_agent_ids_from_access_groups(
            access_group_ids=list(access_group_ids), check_db_only=check_db_only
        )

    @staticmethod
    async def _get_agents_from_access_groups(
        access_groups: Sequence[str],
        *,
        check_db_only: bool = False,
    ) -> list[str]:
        """
        Resolve agent access groups to agent IDs by querying BOTH the agent table (DB) AND config-loaded agents.
        """
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.proxy.proxy_server import prisma_client

        config_agent_ids: Final = AgentRequestHandler._get_config_agent_ids_for_access_groups(
            global_agent_registry.agent_list, access_groups
        )

        # Use the helper for DB agents
        db_agent_ids: Final = await AgentRequestHandler._get_db_agent_ids_for_access_groups(
            prisma_client, access_groups, check_db_only=check_db_only
        )

        return list(config_agent_ids | db_agent_ids)

    @staticmethod
    async def get_agent_access_groups(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> list[str]:
        """
        Get list of agent access groups for the given user/key based on permissions.
        """
        access_groups: list[str] = []
        access_groups_for_key: Final = await AgentRequestHandler._get_agent_access_groups_for_key(user_api_key_auth)
        access_groups_for_team: Final = await AgentRequestHandler._get_agent_access_groups_for_team(user_api_key_auth)

        # If team has access groups, then key must have a subset of the team's access groups
        if len(access_groups_for_team) > 0:
            for access_group in access_groups_for_key:
                if access_group in access_groups_for_team:
                    access_groups.append(access_group)
        else:
            access_groups = access_groups_for_key

        return list(set(access_groups))

    @staticmethod
    async def _get_agent_access_groups_for_key(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> list[str]:
        """Get agent access groups for the key."""
        from litellm.proxy.auth.auth_checks import get_object_permission
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.object_permission_id is None:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        try:
            key_object_permission: Final = await get_object_permission(
                object_permission_id=user_api_key_auth.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if key_object_permission is None:
                return []

            return key_object_permission.agent_access_groups or []
        except NativeAuthFailure:
            raise
        except Exception as e:
            verbose_logger.warning("Failed to get agent access groups for key: %s", e)
            return []

    @staticmethod
    async def _get_agent_access_groups_for_team(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> list[str]:
        """Get agent access groups for the team."""
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.team_id is None:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        try:
            team_obj: Final[LiteLLM_TeamTable | None] = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if team_obj is None:  # pyright: ignore[reportUnnecessaryComparison]  # legacy adapters may return no row
                verbose_logger.debug("team_obj is None")
                return []

            object_permissions: Final = team_obj.object_permission
            if object_permissions is None:
                return []

            return object_permissions.agent_access_groups or []
        except NativeAuthFailure:
            raise
        except Exception as e:
            verbose_logger.warning("Failed to get agent access groups for team: %s", e)
            return []


def _granted_ids(access: AgentAccess) -> frozenset[str]:
    match access:
        case UnrestrictedAgentAccess():
            return frozenset()
        case RestrictedAgentAccess(agent_ids):
            return agent_ids


ResolveAgentAccess: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[AgentAccess]]
EffectiveAuthContexts: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[Sequence[UserAPIKeyAuth]]]


async def _granted_agent_ids(
    user_api_key_auth: UserAPIKeyAuth,
    resolve_access: ResolveAgentAccess,
    effective_contexts: EffectiveAuthContexts,
) -> frozenset[str]:
    """Union of the explicit grants reachable from the key, its team, or (for a dashboard session)
    the user's real teams and user row. No grant anywhere yields the empty set, unlike the
    open-by-default ``resolve_agent_access`` that guards direct access."""
    accesses: Final = await asyncio.gather(
        *(resolve_access(auth_context) for auth_context in await effective_contexts(user_api_key_auth))
    )
    return frozenset[str]().union(*(_granted_ids(access) for access in accesses))


async def accessible_agents(
    user_api_key_auth: UserAPIKeyAuth,
    all_agents: tuple[AgentResponse, ...] | None = None,
    resolve_access: ResolveAgentAccess | None = None,
    effective_contexts: EffectiveAuthContexts = build_effective_auth_contexts,
) -> tuple[AgentResponse, ...]:
    """Every registry agent for proxy admins, else only the agents the caller was granted."""
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    agents: Final = global_agent_registry.get_agent_list() if all_agents is None else all_agents
    if user_api_key_auth.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN.value):
        return agents
    allowed_agent_ids: Final = await _granted_agent_ids(
        user_api_key_auth,
        AgentRequestHandler.resolve_agent_access if resolve_access is None else resolve_access,
        effective_contexts,
    )
    allowed: Final = await asyncio.gather(
        *(
            AgentRequestHandler.is_agent_allowed(agent.agent_id, user_api_key_auth)
            for agent in agents
            if agent.identity_managed
        )
    )
    managed_ids: Final = frozenset(
        agent.agent_id
        for agent, permitted in zip((agent for agent in agents if agent.identity_managed), allowed)
        if permitted
    )
    return tuple(
        agent
        for agent in agents
        if (agent.agent_id in managed_ids if agent.identity_managed else agent.agent_id in allowed_agent_ids)
    )


async def _strict_agent_access(auth: UserAPIKeyAuth) -> AgentAccess:
    return await AgentRequestHandler.resolve_agent_access(auth, strict=True)


async def _managed_actor_agent_access(auth: UserAPIKeyAuth) -> AgentAccess:
    agent: Final = managed_agent_policy(auth)
    context: Final = auth.managed_agent_context
    return await _read_managed_agent_grants(
        auth,
        managed.ManagedAgentGrants(
            permission_present=bool(agent is not None and agent.object_permission),
            mode=context.mode if context is not None else None,
            user_present=context is not None and context.user_id is not None,
        ),
    )


async def _read_managed_agent_grants(auth: UserAPIKeyAuth, facts: managed.ManagedAgentGrants) -> AgentAccess:
    decision: Final = managed.decide(facts)
    agent: Final = managed_agent_policy(auth)
    match decision.action:
        case "Grants":
            return _project_agent_grants(decision)
        case "ReadOwn":
            permission: Final = LiteLLM_ObjectPermissionTable.model_validate(
                agent.object_permission if agent is not None else MappingProxyType({})
            )
            own_auth: Final = UserAPIKeyAuth(object_permission=permission)
            own: Final = await AgentRequestHandler.get_allowed_agents_for_key(own_auth, strict=True)
            return await _read_managed_agent_grants(auth, replace(facts, own=tuple(_granted_ids(own))))
        case "ReadCeilings":
            from litellm.proxy.agent_endpoints.auth.agent_access_groups import resolve_managed_agent_ceilings

            if agent is None:
                raise RuntimeError("native requested absent managed agent")
            ceilings: Final = await resolve_managed_agent_ceilings(agent)
            return await _read_managed_agent_grants(
                auth, replace(facts, ceilings=tuple(tuple(ceiling.agent_ids) for ceiling in ceilings))
            )
        case "ReadCaller":
            caller: Final = await AgentRequestHandler.agent_caller_access(auth, strict=True)
            return await _read_managed_agent_grants(
                auth,
                replace(
                    facts,
                    caller_loaded=True,
                    caller=tuple(caller.agent_ids) if isinstance(caller, RestrictedAgentAccess) else None,
                ),
            )
        case "ReadHumanGrants":
            context: Final = auth.managed_agent_context
            human: Final = await verified_human_agent_grants(
                context.user_id if context is not None else None, auth.team_id
            )
            return await _read_managed_agent_grants(auth, replace(facts, human=tuple(human)))
        case _:
            raise RuntimeError("invalid native managed grants transition")


async def _read_declared_grants(facts: managed.DeclaredGrants, strict: bool) -> AgentAccess:
    decision: Final = managed.decide(facts)
    match decision.action:
        case "Grants":
            return _project_agent_grants(decision)
        case "ReadDeclaredGroups":
            groups: Final = await AgentRequestHandler._get_agents_from_access_groups(  # pyright: ignore[reportPrivateUsage]  # module-owned staged reader
                facts.declared_groups, check_db_only=strict
            )
            return await _read_declared_grants(replace(facts, group_agents=tuple(groups)), strict)
        case "ReadUnifiedGroups":
            unified: Final = await AgentRequestHandler._get_unified_access_group_agents(  # pyright: ignore[reportPrivateUsage]  # module-owned staged reader
                facts.unified_groups, check_db_only=strict
            )
            return await _read_declared_grants(replace(facts, unified_agents=tuple(unified)), strict)
        case _:
            raise RuntimeError("invalid native declared grants transition")


async def _verified_human_agent_sources(
    user_id: str | None, *, allowed_team_ids: frozenset[str] | None = None
) -> tuple[tuple[str | None, frozenset[str]], ...]:
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler

    if user_id is None:
        return ()
    human: Final = await MCPRequestHandler.reload_admitted_user(user_id, requires_fresh_policy=True)
    sources: Final = await MCPRequestHandler.admitted_subject_sources(human, allowed_team_ids=allowed_team_ids)
    access: Final = await asyncio.gather(*(_strict_agent_access(source) for source in sources))
    return tuple((source.team_id, _granted_ids(grant)) for source, grant in zip(sources, access, strict=True))


async def verified_human_agent_grants(user_id: str | None, team_id: str | None = None) -> frozenset[str]:
    sources: Final = await _verified_human_agent_sources(
        user_id, allowed_team_ids=frozenset((team_id,)) if team_id else frozenset()
    )
    decision: Final = agent_grants.decide(
        agent_grants.Human(
            tuple(agent_grants.Source(source, tuple(grants)) for source, grants in sources),
            team_id,
        )
    )
    return frozenset(decision.grants or ())


async def resolve_delegated_agent_team(
    user_id: str | None,
    agent_id: str,
    team_id: str | None,
    *,
    explicit_team: bool,
    allowed_team_ids: frozenset[str] | None = None,
) -> str | None:
    sources: Final = await _verified_human_agent_sources(user_id)
    decision: Final = agent_grants.decide(
        agent_grants.Delegated(
            tuple(agent_grants.Source(source, tuple(grants)) for source, grants in sources),
            agent_id,
            team_id,
            explicit_team,
            tuple(allowed_team_ids) if allowed_team_ids is not None else None,
        )
    )
    if decision.action == "SelectTeam":
        raise HTTPException(403, "Select a team that grants access to this agent using x-litellm-team-id")
    return decision.team_id


def _agent_grant_source(team_id: str | None, access: AgentAccess | AgentPolicyUnavailable) -> agent_grants.Source:
    return agent_grants.Source(
        team_id,
        tuple(access.agent_ids) if isinstance(access, RestrictedAgentAccess) else None,
        isinstance(access, AgentPolicyUnavailable),
    )


async def _read_resolved_agent_grants(
    auth: UserAPIKeyAuth | None,
    ceiling: CeilingResolver,
    default_deny: Callable[[], bool],
    key_lookup: AgentAccessLookup | None,
    team_lookup: AgentAccessLookup | None,
    facts: agent_grants.Resolve,
) -> AgentAccess:
    decision: Final = agent_grants.decide(facts)
    match decision.action:
        case "Complete":
            return _project_agent_grants(decision)
        case "ReadManaged":
            assert auth is not None
            return await _managed_actor_agent_access(auth)
        case "ReadKeyTeam":
            access: Final = await AgentRequestHandler.resolve_key_team_agent_access(
                auth,
                strict=facts.strict,
                lookup_key_access=key_lookup,
                lookup_team_access=team_lookup,
            )
            return await _read_resolved_agent_grants(
                auth,
                ceiling,
                default_deny,
                key_lookup,
                team_lookup,
                replace(
                    facts,
                    key_team_loaded=True,
                    key_team=_agent_grant_source(None, access).grants,
                ),
            )
        case "ReadDefault":
            return await _read_resolved_agent_grants(
                auth,
                ceiling,
                default_deny,
                key_lookup,
                team_lookup,
                replace(
                    facts,
                    default_deny=default_deny(),
                ),
            )
        case "ReadCaller":
            caller: Final = await AgentRequestHandler.agent_caller_access(auth, strict=facts.strict)
            own: Final = managed.decide(
                managed.IntersectGrants(
                    facts.key_team,
                    _agent_grant_source(None, caller).grants,
                )
            )
            return await _read_resolved_agent_grants(
                auth,
                ceiling,
                default_deny,
                key_lookup,
                team_lookup,
                replace(
                    facts,
                    caller_loaded=True,
                    own=own.grants,
                ),
            )
        case "ReadCeiling":
            ids: Final = await AgentRequestHandler._agent_access_group_ceiling(auth, ceiling, strict=facts.strict)  # pyright: ignore[reportPrivateUsage]  # module-owned staged reader
            return await _read_resolved_agent_grants(
                auth,
                ceiling,
                default_deny,
                key_lookup,
                team_lookup,
                replace(
                    facts,
                    ceiling_loaded=True,
                    ceiling=tuple(ids) if ids is not None else None,
                ),
            )
        case _:
            raise NativeAuthFailure(f"unexpected agent grant directive {decision.action}")


async def read_agent_access_facts(
    agent_id: str,
    auth: UserAPIKeyAuth | None,
    resolve_ceiling: CeilingResolver = resolve_agent_access_group_ceiling,
) -> agent_access.AgentAccessFacts:
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.proxy.proxy_server import prisma_client

    registered: Final = global_agent_registry.get_agent_by_id(agent_id)
    facts: Final = agent_access.AgentAccessFacts(
        registry_managed=isinstance(registered, AgentResponse) and registered.identity_managed,
        database_present=prisma_client is not None,
        subject_present=auth is not None,
        key_hash_present=bool(auth.api_key or auth.token) if auth is not None else False,
        subject_managed=managed_agent_policy(auth) is not None,
        session=auth.is_session_token if auth is not None else False,
        custom=auth.authenticated_by_custom_auth if auth is not None else False,
    )
    return await _read_agent_access(agent_id, auth, resolve_ceiling, facts)


async def _read_agent_access(
    agent_id: str,
    auth: UserAPIKeyAuth | None,
    resolve_ceiling: CeilingResolver,
    facts: agent_access.AgentAccessFacts,
    authority: UserAPIKeyAuth | None = None,
) -> agent_access.AgentAccessFacts:
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
    from litellm.proxy.agent_endpoints.auth.managed_authorization import project_agent
    from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore
    from litellm.proxy.agent_endpoints.managed_identity import raise_identity_failure
    from litellm.proxy.proxy_server import prisma_client
    from litellm.types.proxy.agent_identity import AgentIdentityFailure

    action: Final = agent_access.decide(facts)
    match action:
        case "Allow" | "Deny":
            return facts
        case "ReadTarget":
            target: Final = await AgentIdentityStore.from_client(prisma_client).agent(agent_id)
            if isinstance(target, AgentIdentityFailure):
                raise_identity_failure(target)
            return await _read_agent_access(
                agent_id,
                auth,
                resolve_ceiling,
                replace(facts, target_loaded=True, target=project_agent(target) if target is not None else None),
                authority,
            )
        case "ReadAuthority":
            from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler

            assert auth is not None
            key_hash: Final = auth.api_key or auth.token
            assert key_hash is not None
            reloaded: Final = await MCPRequestHandler._reload_admitted_key(key_hash, check_db_only=True)  # pyright: ignore[reportPrivateUsage]  # authoritative key reload has no public seam
            return await _read_agent_access(
                agent_id, auth, resolve_ceiling, replace(facts, authority_loaded=True), reloaded
            )
        case "ReadExplicitGrants":
            assert auth is not None
            source: Final = authority if authority is not None else auth
            fresh: Final = source.model_copy(
                update=MappingProxyType({"requires_fresh_policy": True, "agent_caller": auth.agent_caller})
            )
            explicit: Final = await _granted_agent_ids(fresh, _strict_agent_access, build_effective_auth_contexts)
            return await _read_agent_access(
                agent_id, auth, resolve_ceiling, replace(facts, explicit_grants=tuple(explicit)), authority
            )
        case "ReadLegacyGrants":
            grants: Final = await AgentRequestHandler.resolve_agent_access(
                auth, resolve_ceiling, strict=bool(auth is not None and auth.requires_fresh_policy)
            )
            return await _read_agent_access(
                agent_id,
                auth,
                resolve_ceiling,
                replace(
                    facts,
                    legacy_loaded=True,
                    legacy_grants=tuple(grants.agent_ids) if isinstance(grants, RestrictedAgentAccess) else None,
                ),
                authority,
            )
        case "ReadTargetIds":
            stable: Final = global_agent_registry.stable_agent_id(agent_id)
            ids: Final = global_agent_registry.ids_for_agent(stable)
            return await _read_agent_access(
                agent_id, auth, resolve_ceiling, replace(facts, target_ids=tuple(ids)), authority
            )
        case _:
            assert_never(action)
