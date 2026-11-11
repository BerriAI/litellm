"""Answers one `RustAuthStep` with the `PythonAuthHost` method it names. The sections follow the `AuthHost` trait in
`gateway-auth-core/src/host.rs`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Final

import litellm
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.host.facts import (
    AliasMissing,
    FoundTeam,
    TeamFacts,
    TeamMissing,
    deactivated,
    key_facts,
    user_facts,
)
from litellm.rust_bridge._native import RustAuth, RustAuthStep
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

if TYPE_CHECKING:
    from litellm.proxy.auth.host.python import PythonAuthHost


def _key_row(call: RustAuth, row: UserAPIKeyAuth | None) -> None:
    if row is None:
        call.key()
    else:
        call.key(key_facts(row), row)


def _team_read(
    read: FoundTeam | TeamMissing | AliasMissing,
) -> tuple[TeamFacts | TeamMissing | AliasMissing, LiteLLM_TeamTable | None]:
    match read:
        case FoundTeam(facts=facts, row=row):
            return facts, row
        case missing:
            return missing, None


def _org(call: RustAuth, org: LiteLLM_OrganizationTable | None) -> None:
    if org is None or org.organization_id is None:
        call.org()
    else:
        call.org(org.organization_id, org)


def _user(call: RustAuth, user: LiteLLM_UserTable | None) -> None:
    if user is None:
        call.user()
    else:
        call.user(user_facts(user), user)


async def _budget(call: RustAuth, check: Awaitable[None]) -> None:
    try:
        await check
    except litellm.BudgetExceededError as exceeded:
        call.budget_exceeded(exceeded)
        return
    call.done()


async def _finish_jwt(call: RustAuth, finish: Awaitable[UserAPIKeyAuth]) -> None:
    try:
        token: Final = await finish
    except litellm.BudgetExceededError as exceeded:
        call.budget_exceeded(exceeded)
        return
    call.row(token)


async def _key_reads(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.CachedKey(hash=key_hash):
            _key_row(call, await host.cached_key(key_hash))
        case RustAuthStep.UiKey(token=token):
            _key_row(call, await host.ui_key(token))
        case RustAuthStep.DbKey(hash=key_hash, end_user=end_user):
            _key_row(call, await host.db_key(key_hash, end_user))
        case RustAuthStep.KeyOwner(user_id=user_id):
            owner: Final = await host.key_owner(user_id)
            call.key_owner(owner, deactivated=owner is not None and deactivated(owner))
        case RustAuthStep.CachedTeam(team_id=team_id):
            team: Final = await host.cached_team(team_id)
            call.cached_team(team, refreshed_at=team.last_refreshed_at)
        case RustAuthStep.KeyTeam(team_id=team_id):
            call.found_row(await host.key_team(team_id))
        case RustAuthStep.ObjectPermission(id=object_permission_id):
            call.found_row(await host.object_permission(object_permission_id))
        case RustAuthStep.Project(project_id=project_id):
            call.found_row(await host.project(project_id))
        case _:
            return False
    return True


async def _key_rebuilds(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.WithSessionGrants(key=session_key, user=user, team=team, membership=membership):
            _key_row(call, await host.with_session_grants(session_key, user, team, membership))
        case RustAuthStep.WithTeam(key=cached_key, team=cached_team):
            call.row(await host.with_team(cached_key, cached_team))
        case RustAuthStep.WithTempBudget(key=request_key):
            call.row(await host.with_temp_budget(request_key))
        case RustAuthStep.NormalizeModel(key=request_key):
            await host.normalize_model(request_key)
            call.done()
        case RustAuthStep.PrefetchAuthObjects(key=request_key, end_user=end_user):
            await host.prefetch_auth_objects(request_key, end_user)
            call.done()
        case RustAuthStep.EvictKey(hash=key_hash):
            await host.evict_key(key_hash)
            call.done()
        case _:
            return False
    return True


async def _jwt_io(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.FetchJson(url=url):
            call.json(await host.fetch_json(url))
        case RustAuthStep.FetchUserinfo(endpoint=endpoint, token=token):
            call.json(await host.fetch_userinfo(endpoint, token))
        case RustAuthStep.CachedJson(key=key):
            call.json(await host.cached_json(key))
        case RustAuthStep.CacheJson(key=key, value=value, ttl_seconds=ttl_seconds):
            await host.cache_json(key, value, ttl_seconds)
            call.done()
        case RustAuthStep.ForgetJson(key=key):
            await host.forget_json(key)
            call.done()
        case _:
            return False
    return True


async def _jwt_reads(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.Team(team_id=team_id, upsert=upsert):
            call.team(*_team_read(await host.team(team_id, upsert)))
        case RustAuthStep.TeamByAlias(alias=alias):
            call.team_alias(*_team_read(await host.team_by_alias(alias)))
        case RustAuthStep.GrantingTeam(team_id=team_id):
            call.team(*_team_read(await host.granting_team(team_id)))
        case RustAuthStep.LoadTeamMembership(user_id=user_id, team_id=team_id):
            call.found_row(await host.load_team_membership(user_id, team_id))
        case RustAuthStep.User(user_id=user_id, user_email=user_email, sso_user_id=sso_user_id, upsert=upsert):
            _user(call, await host.user(user_id, user_email, sso_user_id, upsert))
        case RustAuthStep.ResolveUserIdentity(
            user_id=user_id, user_email=user_email, sso_user_id=sso_user_id, upsert=upsert, team_id=team_id
        ):
            match await host.resolve_user_identity(user_id, user_email, sso_user_id, upsert, team_id):
                case (user, membership):
                    call.user(user_facts(user), user, membership=membership)
                case None:
                    call.user()
        case RustAuthStep.Org(org_id=org_id):
            _org(call, await host.org(org_id))
        case RustAuthStep.OrgByAlias(alias=alias):
            _org(call, await host.org_by_alias(alias))
        case RustAuthStep.JwtEndUser(end_user_id=end_user_id):
            call.row(await host.jwt_end_user(end_user_id))
        case _:
            return False
    return True


async def _jwt_writes(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.SetUserRole(user=user, role=role):
            call.row(await host.set_user_role(user, role))
        case RustAuthStep.PatchTeamMembership(user=user, add=add, remove=remove):
            call.row(await host.patch_team_membership(user, add, remove))
        case RustAuthStep.AddTeamMember(user_id=user_id, team_id=team_id):
            await host.add_team_member(user_id, team_id)
            call.done()
        case _:
            return False
    return True


async def _key_mappings(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.KeyMappingHash(claim_field=claim_field, claim_value=claim_value, issuer=issuer):
            call.text(await host.key_mapping_hash(claim_field, claim_value, issuer))
        case RustAuthStep.KeyByHash(hash=key_hash):
            mapped: Final = await host.key_by_hash(key_hash)
            call.mapped_key(key_facts(mapped), mapped, user_email=mapped.user_email)
        case RustAuthStep.ReusableKeyHash(user_id=user_id, team_id=team_id):
            call.text(await host.reusable_key_hash(user_id, team_id))
        case RustAuthStep.CreateKeyMapping(
            claim_field=claim_field,
            claim_value=claim_value,
            issuer=issuer,
            behavior=behavior,
            existing_hash=existing_hash,
            team_id=team_id,
            user_id=user_id,
            org_id=org_id,
            agent_id=agent_id,
        ):
            token_hash, conflict = await host.create_key_mapping(
                claim_field, claim_value, issuer, behavior, existing_hash, team_id, user_id, org_id, agent_id
            )
            call.key_mapping(token_hash=token_hash, conflict=conflict)
        case _:
            return False
    return True


async def _agents(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.ManagedAgent(claims=claims):
            call.managed_agent(await host.managed_agent(claims))
        case RustAuthStep.RegisteredAgent(claim=claim):
            call.registered_agent(await host.registered_agent(claim))
        case RustAuthStep.RecordAgentAuthentication(agent=agent):
            call.agent_failure(await host.record_agent_authentication(agent))
        case _:
            return False
    return True


async def _budgets(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.EndUser(token=token, resolved=resolved):
            try:
                call.row(await host.end_user(token, resolved))
            except litellm.BudgetExceededError as exceeded:
                call.budget_exceeded(exceeded)
        case RustAuthStep.KeySkipsBudgetChecks(key=budget_key):
            call.found(await host.key_skips_budget_checks(budget_key))
        case RustAuthStep.TeamMemberMaxBudget(user_id=user_id, team_id=team_id):
            call.amount(await host.team_member_max_budget(user_id, team_id))
        case RustAuthStep.KeyModelBudgets(key=budget_key, owner=owner):
            await _budget(call, host.key_model_budgets(budget_key, owner))
        case RustAuthStep.ThrottleKey(key=budget_key, share=share):
            await host.throttle_key(budget_key, share)
            call.done()
        case RustAuthStep.GlobalProxySpend():
            call.amount(await host.global_proxy_spend())
        case RustAuthStep.KeyBudgetAlert(key=budget_key, owner=owner, kind=kind, spend=spend, max_budget=max_budget):
            await host.key_budget_alert(budget_key, owner, kind, spend, max_budget)
            call.done()
        case RustAuthStep.FinishJwt(identity=identity):
            await _finish_jwt(call, host.finish_jwt(identity))
        case _:
            return False
    return True


async def _decisions(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.CustomValidate(claims=claims):
            call.found(await host.custom_validate(claims))
        case RustAuthStep.TeamModelAccess(team=team, model=model):
            call.model_access(allowed=await host.team_model_access(team, model))
        case RustAuthStep.DelegatedAgentTeam(
            user_id=user_id,
            agent_id=agent_id,
            team_id=team_id,
            explicit_team=explicit_team,
            allowed_team_ids=allowed_team_ids,
        ):
            call.text(await host.delegated_agent_team(user_id, agent_id, team_id, explicit_team, allowed_team_ids))
        case _:
            return False
    return True


async def _authorization(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> bool:
    match step:
        case RustAuthStep.UnpricedModels():
            call.names(list(await host.unpriced_models()))
        case RustAuthStep.AgentRequiresTraceId(agent_id=agent_id):
            call.found(await host.agent_requires_trace_id(agent_id))
        case RustAuthStep.HasTraceId(headers_only=headers_only):
            call.found(await host.has_trace_id(headers_only))
        case RustAuthStep.TeamOrganization(team_id=team_id):
            call.text(await host.team_organization(team_id))
        case RustAuthStep.MergeBudgetTags():
            await host.merge_budget_tags()
            call.done()
        case RustAuthStep.CheckTeamModelAccess(membership_read=membership_read, membership=membership):
            await host.check_team_model_access(membership_read, membership)
            call.done()
        case RustAuthStep.CheckModelGrants():
            await host.check_model_grants()
            call.done()
        case RustAuthStep.CheckProject():
            await host.check_project()
            call.done()
        case RustAuthStep.StampAccessGroups(membership_read=membership_read, membership=membership):
            call.names(list(await host.stamp_access_groups(membership_read, membership)))
        case RustAuthStep.Spend(
            scope=scope,
            entity_id=entity_id,
            team_id=team_id,
            window=window,
            recorded=recorded,
            max_budget=max_budget,
        ):
            call.spend(await host.spend(scope, entity_id, team_id, window, recorded, max_budget))
        case RustAuthStep.OrganizationBudget(org_id=org_id):
            organization_budget: Final = await host.organization_budget(org_id)
            if organization_budget is None:
                call.organization_budget()
            else:
                call.organization_budget(*organization_budget)
        case RustAuthStep.TagBudgets():
            call.named_budgets(list(await host.tag_budgets()))
        case RustAuthStep.ModelAccessGroupBudgets(groups=groups):
            call.named_budgets(list(await host.model_access_group_budgets(groups)))
        case RustAuthStep.TeamMemberBudgetState(membership=membership):
            member: Final = await host.team_member_budget_state(membership)
            call.member_budget(spend=member.spend, budget=member.budget, team_default=member.team_default)
        case RustAuthStep.BudgetAlert(kind=kind, org_id=org_id, spend=spend, max_budget=max_budget):
            await host.budget_alert(kind, org_id, spend, max_budget)
            call.done()
        case RustAuthStep.CheckGuardrailModification():
            await host.check_guardrail_modification()
            call.done()
        case RustAuthStep.CheckObjectAccess():
            await host.check_object_access()
            call.done()
        case _:
            return False
    return True


_SECTIONS: Final[tuple[Callable[[RustAuth, RustAuthStep, PythonAuthHost], Awaitable[bool]], ...]] = (
    _key_reads,
    _key_rebuilds,
    _jwt_io,
    _jwt_reads,
    _jwt_writes,
    _key_mappings,
    _agents,
    _budgets,
    _decisions,
    _authorization,
)


async def answer(call: RustAuth, step: RustAuthStep, host: PythonAuthHost) -> None:
    for section in _SECTIONS:
        if await section(call, step, host):
            return
    raise RuntimeError(f"rust auth asked an unknown question: {step!r}")
