from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import NativeAuthRuntimeError, native_boundary
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.managed_models import (
    AgentCeilingSourceFacts,
    ManagedModelFacts,
    agent_ceiling_source,
    managed_model_grants,
    managed_models,
)
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.agent_access_groups import (
    AgentAccessGroupCeiling,
    resolve_agent_access_group_ceiling,
    resolve_managed_agent_ceilings,
)
from litellm.proxy.agent_endpoints.auth.agent_caller import (
    agent_caller_auth,
    load_agent_caller_team,
    load_agent_caller_user,
)
from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_agent_policy
from litellm.proxy.auth.member_model_adapter import read_member_model_facts
from litellm.proxy.auth.model_access_adapter import object_model_grants, project_model_facts, raise_model_denial
from litellm.proxy.auth.team_alias_adapter import resolve_team_alias
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router_api import RegistryReader
from litellm.types.agents import AgentResponse

CeilingResolver: TypeAlias = Callable[[str], Awaitable[AgentAccessGroupCeiling | None]]
CallerTeamLoader: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[LiteLLM_TeamTable | None]]
CallerUserLoader: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[LiteLLM_UserTable | None]]
_ITEMS: Final = TypeAdapter(tuple[object, ...])
_NAMES: Final = TypeAdapter(tuple[str, ...])


@dataclass(frozen=True, slots=True)
class _Source:
    model: str | Sequence[str] | None
    token: UserAPIKeyAuth | None
    router: RegistryReader | None
    prisma: PrismaClient | None
    cache: UserApiKeyCache | None
    logging: ProxyLogging | None
    ceiling: CeilingResolver
    team_loader: CallerTeamLoader
    user_loader: CallerUserLoader
    managed_grants: tuple[object, ...] = ()
    dispatched: str | Sequence[str] = ()
    caller: UserAPIKeyAuth | None = None
    team: LiteLLM_TeamTable | None = None
    user: LiteLLM_UserTable | None = None
    key_aliases: Mapping[str, str] | None = None


async def agent_access_group_ceilings(
    valid_token: UserAPIKeyAuth, resolve_ceiling: CeilingResolver = resolve_agent_access_group_ceiling
) -> tuple[AgentAccessGroupCeiling, ...]:
    return await _ceilings(AgentCeilingSourceFacts(bool(valid_token.agent_id)), valid_token, resolve_ceiling)


async def _ceilings(
    facts: AgentCeilingSourceFacts,
    token: UserAPIKeyAuth,
    resolver: CeilingResolver,
    managed: AgentResponse | None = None,
    rows: tuple[AgentAccessGroupCeiling, ...] = (),
) -> tuple[AgentAccessGroupCeiling, ...]:
    decision: Final = agent_ceiling_source(facts)
    match decision.kind:
        case "Complete":
            return tuple(rows[index] for index in decision.indices)
        case "ReadManaged":
            policy: Final = managed_agent_policy(token)
            return await _ceilings(
                replace(facts, managed_loaded=True, managed_present=policy is not None), token, resolver, policy
            )
        case "ReadManagedCeilings":
            if managed is None:
                raise NativeAuthRuntimeError("missing managed agent ceiling source")
            loaded: Final = await resolve_managed_agent_ceilings(managed)
            return await _ceilings(replace(facts, loaded=True, count=len(loaded)), token, resolver, managed, loaded)
        case "ReadUnmanagedCeiling":
            if token.agent_id is None:
                raise NativeAuthRuntimeError("missing agent ceiling identity")
            row: Final = await resolver(token.agent_id)
            projected: Final = (row,) if row is not None else ()
            return await _ceilings(
                replace(facts, loaded=True, count=len(projected)), token, resolver, managed, projected
            )


async def read_managed_model_facts(
    model: str | Sequence[str] | None,
    valid_token: UserAPIKeyAuth | None,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache | None,
    proxy_logging_obj: ProxyLogging | None,
    *,
    purpose: Literal["All", "Ceilings", "Caller"] = "All",
    resolve_ceiling: CeilingResolver = resolve_agent_access_group_ceiling,
    load_team: CallerTeamLoader = load_agent_caller_team,
    load_user: CallerUserLoader = load_agent_caller_user,
) -> ManagedModelFacts:
    facts, _source = await _read(
        ManagedModelFacts(purpose, bool(model), valid_token is not None),
        _Source(
            model,
            valid_token,
            llm_router,
            prisma_client,
            user_api_key_cache,
            proxy_logging_obj,
            resolve_ceiling,
            load_team,
            load_user,
        ),
    )
    return facts


async def can_agent_call_model(
    model: str | Sequence[str] | None,
    valid_token: UserAPIKeyAuth | None,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache | None,
    proxy_logging_obj: ProxyLogging | None,
    *,
    purpose: Literal["All", "Ceilings", "Caller"] = "All",
    resolve_ceiling: CeilingResolver = resolve_agent_access_group_ceiling,
    load_team: CallerTeamLoader = load_agent_caller_team,
    load_user: CallerUserLoader = load_agent_caller_user,
) -> None:
    facts, source = await _read(
        ManagedModelFacts(purpose, bool(model), valid_token is not None),
        _Source(
            model,
            valid_token,
            llm_router,
            prisma_client,
            user_api_key_cache,
            proxy_logging_obj,
            resolve_ceiling,
            load_team,
            load_user,
        ),
    )
    decision: Final = managed_models(facts)
    if decision.kind == "Allow":
        return
    if decision.kind != "Deny" or decision.denial is None or model is None:
        raise NativeAuthRuntimeError("incomplete managed model authorization")
    raise_model_denial(
        decision.denial,
        model,
        user_id=source.caller.user_id
        if decision.member_models is not None and source.caller is not None
        else valid_token.agent_id
        if valid_token is not None
        else None,
        team_id=source.caller.team_id if source.caller is not None else None,
        member_models=decision.member_models,
    )


@native_boundary
def _managed_grants(raw: tuple[object, ...]) -> tuple[str, ...]:
    return _NAMES.validate_python(raw)


def _declared_grants(token: UserAPIKeyAuth) -> tuple[str | None, ...] | None:
    managed: Final = managed_agent_policy(token)
    raw: Final = (managed.object_permission or {}).get("models") if managed is not None else None
    if not isinstance(raw, (list, tuple)):
        return None
    return tuple(value if isinstance(value, str) else None for value in _ITEMS.validate_python(raw))


def managed_agent_model_grants(valid_token: UserAPIKeyAuth) -> frozenset[str]:
    return managed_model_grants(_declared_grants(valid_token), ())


async def agent_model_grants(
    valid_token: UserAPIKeyAuth, resolve_ceiling: CeilingResolver = resolve_agent_access_group_ceiling
) -> frozenset[str]:
    ceilings: Final = await agent_access_group_ceilings(valid_token, resolve_ceiling)
    return managed_model_grants(_declared_grants(valid_token), tuple(tuple(ceiling.models) for ceiling in ceilings))


async def check_agent_access_group_model_access(
    model: str | Sequence[str] | None,
    valid_token: UserAPIKeyAuth | None,
    llm_router: RegistryReader | None,
    resolve_ceiling: CeilingResolver = resolve_agent_access_group_ceiling,
) -> Literal[True]:
    await can_agent_call_model(
        model, valid_token, llm_router, None, None, None, purpose="Ceilings", resolve_ceiling=resolve_ceiling
    )
    return True


async def check_agent_caller_model_access(
    model: str | Sequence[str] | None,
    valid_token: UserAPIKeyAuth | None,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    load_team: CallerTeamLoader = load_agent_caller_team,
    load_user: CallerUserLoader = load_agent_caller_user,
) -> None:
    await can_agent_call_model(
        model,
        valid_token,
        llm_router,
        prisma_client,
        user_api_key_cache,
        proxy_logging_obj,
        purpose="Caller",
        load_team=load_team,
        load_user=load_user,
    )


def _model(source: _Source) -> str | Sequence[str]:
    if source.model is None:
        raise NativeAuthRuntimeError("managed model read without a request model")
    return source.model


def _token(source: _Source) -> UserAPIKeyAuth:
    if source.token is None:
        raise NativeAuthRuntimeError("managed model read without a token")
    return source.token


def _dispatch(source: _Source) -> str | Sequence[str]:
    from litellm.proxy.auth.auth_checks import team_model_aliases_for_auth_check

    token: Final = _token(source)
    return resolve_team_alias(_model(source), team_model_aliases_for_auth_check(token), token.team_id, source.router)


async def _read(facts: ManagedModelFacts, source: _Source) -> tuple[ManagedModelFacts, _Source]:
    from litellm.proxy.auth.auth_checks import get_models_from_access_groups, key_model_aliases_for_auth_check

    decision: Final = managed_models(facts)
    match decision.kind:
        case "Allow" | "Deny":
            return facts, source
        case "ReadManaged":
            policy: Final = managed_agent_policy(source.token)
            raw: Final = (policy.object_permission or {}).get("models", ()) if policy is not None else ()
            items: Final = _ITEMS.validate_python(raw) if isinstance(raw, (list, tuple)) else ()
            return await _read(
                replace(
                    facts,
                    managed_loaded=True,
                    managed_present=policy is not None,
                    managed_models_valid=isinstance(raw, (list, tuple)),
                    managed_models_count=len(items),
                ),
                replace(source, managed_grants=items),
            )
        case "ReadManagedScope":
            dispatched: Final = _dispatch(source)
            managed: Final = project_model_facts(
                dispatched,
                source.router,
                _managed_grants(source.managed_grants),
                team_id=_token(source).team_id,
                key_model_aliases=key_model_aliases_for_auth_check(source.token),
                object_type="agent",
            )
            return await _read(replace(facts, managed=managed), source)
        case "ReadCeilings":
            ceilings: Final = await agent_access_group_ceilings(_token(source), source.ceiling)
            return await _read(
                replace(facts, ceilings=tuple(tuple(sorted(ceiling.models)) for ceiling in ceilings)), source
            )
        case "ReadAliases":
            return await _read(replace(facts, aliases_loaded=True), replace(source, dispatched=_dispatch(source)))
        case "ReadCeilingScope":
            if facts.ceilings is None:
                raise NativeAuthRuntimeError("managed model scope without ceilings")
            ceiling_scope: Final = project_model_facts(
                source.dispatched,
                source.router,
                facts.ceilings[decision.index],
                team_id=_token(source).team_id,
                key_model_aliases=key_model_aliases_for_auth_check(source.token),
                object_type="agent",
            )
            return await _read(replace(facts, scopes=(*facts.scopes, ceiling_scope)), source)
        case "ReadCaller":
            caller: Final = agent_caller_auth(_token(source))
            return await _read(
                replace(facts, caller_loaded=True, caller_present=caller is not None), replace(source, caller=caller)
            )
        case "ReadTeam":
            team: Final = await source.team_loader(_token(source))
            aliases: Final = key_model_aliases_for_auth_check(source.token)
            return await _read(
                replace(facts, team_loaded=True, team_present=team is not None),
                replace(source, team=team, key_aliases=aliases),
            )
        case "ReadTeamScope":
            if source.team is None:
                raise NativeAuthRuntimeError("missing caller team for model scope")
            scope: Final = project_model_facts(
                _model(source),
                source.router,
                object_model_grants(source.team),
                team_id=source.team.team_id,
                key_model_aliases=source.key_aliases,
                object_type="team",
            )
            return await _read(
                replace(facts, team=GroupedModelScope(scope, tuple(source.team.access_group_ids or ()))), source
            )
        case "ReadTeamGroups":
            if facts.team is None:
                raise NativeAuthRuntimeError("missing caller team group scope")
            grants: Final = await get_models_from_access_groups(facts.team.group_ids, prisma_client=source.prisma)
            return await _read(replace(facts, team=replace(facts.team, group_models=tuple(grants))), source)
        case "ReadMember":
            if source.team is None or source.caller is None or source.cache is None or source.logging is None:
                raise NativeAuthRuntimeError("missing caller member model context")
            member: Final = await read_member_model_facts(
                _model(source),
                source.team,
                source.caller,
                source.router,
                source.prisma,
                source.cache,
                source.logging,
                key_model_aliases=source.key_aliases,
            )
            return await _read(replace(facts, member=member), source)
        case "ReadUser":
            user: Final = await source.user_loader(_token(source))
            return await _read(
                replace(
                    facts,
                    user_loaded=True,
                    user_present=user is not None,
                    user_grants=object_model_grants(user) if user is not None else (),
                ),
                replace(source, user=user),
            )
        case "ReadUserScope":
            user_scope: Final = project_model_facts(
                _model(source), source.router, facts.user_grants, key_model_aliases=source.key_aliases
            )
            return await _read(replace(facts, user=user_scope), source)
