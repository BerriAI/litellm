from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import (  # noqa: TID251  # preserve opaque legacy budget mappings without coercing their keys
    Final,
    Protocol,
    cast,
)

from pydantic import BaseModel

import litellm
from litellm._auth.decision import FallbackTarget
from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.fallback_candidate import (
    CandidateBudgetRead,
    CandidateBudgetScope,
    FallbackCandidateDecision,
    FallbackCandidateFacts,
    fallback_candidate,
)
from litellm._auth.model_attribution import ModelGroupAttributionFacts
from litellm._auth.model_budget_data import ModelBudgetEntity
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LiteLLM_ProjectTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.budget_pricing_adapter import read_model_cost_facts
from litellm.proxy.auth.common_budget_lookups import read_access_group_budget_facts
from litellm.proxy.auth.fallback_access_adapter import load_fallback_authorization_facts
from litellm.proxy.auth.model_access_adapter import object_model_grants, project_model_facts
from litellm.proxy.auth.model_attribution_adapter import read_model_group_attribution_facts
from litellm.proxy.auth.model_budget_adapter import ModelBudgetReading
from litellm.proxy.auth.route_policy_adapter import route_context
from litellm.proxy.auth.unpriced_model_adapter import read_unpriced_model_facts
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.model_max_budget_limiter import team_member_budget_entity_id
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router import Router


class ModelBudgetReader(Protocol):
    async def read_model_budget_raw(
        self,
        entity: ModelBudgetEntity,
        entity_id: str | None,
        team_id: str | None,
        model: str,
        budgets: Mapping[str, object] | None,
        key_budgets: Mapping[str, object] | None = None,
    ) -> ModelBudgetReading: ...


class _BudgetFields(BaseModel, frozen=True):
    model_max_budget: object = None
    user_model_max_budget: object = None
    end_user_model_max_budget: object = None
    team_model_max_budget: object = None
    team_member_model_max_budget: object = None


@dataclass(frozen=True, slots=True)
class _BudgetSource:
    entity: ModelBudgetEntity
    identity: str | None
    budgets: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class _Dependencies:
    prisma: PrismaClient | None
    cache: UserApiKeyCache
    logging: ProxyLogging


@dataclass(frozen=True, slots=True)
class _Sources:
    target: FallbackTarget
    token: UserAPIKeyAuth
    limiter: ModelBudgetReader
    router: Router | None
    route: str
    project: LiteLLM_ProjectTable | None
    team: LiteLLM_TeamTable | None
    customer: LiteLLM_EndUserTable | None
    prisma: PrismaClient | None
    cache: UserApiKeyCache | None
    logging: ProxyLogging | None
    dependencies: _Dependencies | None = None
    membership: LiteLLM_TeamMembership | None = None
    budgets: tuple[_BudgetSource, ...] = ()


def _mapping(raw: object) -> Mapping[str, object] | None:
    if not isinstance(raw, Mapping):
        return None
    return cast(Mapping[str, object], raw)  # cast-ok: preserve opaque mapping contents


def _budget_sources(token: UserAPIKeyAuth) -> tuple[_BudgetSource, ...]:
    fields: Final = _BudgetFields.model_validate(token, from_attributes=True)
    member_id: Final = (
        team_member_budget_entity_id(token.user_id, token.team_id)
        if token.user_id is not None and token.team_id is not None
        else None
    )
    return (
        _BudgetSource("Key", token.token, _mapping(fields.model_max_budget)),
        _BudgetSource("User", token.user_id, _mapping(fields.user_model_max_budget)),
        _BudgetSource("EndUser", token.end_user_id, _mapping(fields.end_user_model_max_budget)),
        _BudgetSource("Team", token.team_id, _mapping(fields.team_model_max_budget)),
        _BudgetSource("TeamMember", member_id, _mapping(fields.team_member_model_max_budget)),
    )


async def read_fallback_candidate_facts(
    *,
    target: FallbackTarget,
    valid_token: UserAPIKeyAuth,
    limiter: ModelBudgetReader,
    llm_router: Router | None,
    route: str,
    project_object: LiteLLM_ProjectTable | None = None,
    team_object: LiteLLM_TeamTable | None = None,
    end_user_object: LiteLLM_EndUserTable | None = None,
    prisma_client: PrismaClient | None = None,
    user_api_key_cache: UserApiKeyCache | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
) -> FallbackCandidateFacts:
    facts: Final = FallbackCandidateFacts(
        router_present=llm_router is not None,
        user_present=valid_token.user_id is not None,
        team_present=valid_token.team_id is not None,
        block_unpriced=litellm.block_requests_for_models_without_pricing,
        customer_present=end_user_object is not None,
    )
    return await _read(
        facts,
        _Sources(
            target,
            valid_token,
            limiter,
            llm_router,
            route,
            project_object,
            team_object,
            end_user_object,
            prisma_client,
            user_api_key_cache,
            proxy_logging_obj,
        ),
    )


def _warn(source: _Sources, *, exception: bool) -> None:
    verbose_proxy_logger.warning(
        "Skipping fallback candidate %s after a budget or authorization check error",
        source.target.model.replace("\r", "").replace("\n", ""),
        exc_info=exception,
    )


async def _read(facts: FallbackCandidateFacts, source: _Sources) -> FallbackCandidateFacts:
    decision: Final = fallback_candidate(facts)
    if decision.kind == "Complete":
        if decision.log_failure:
            _warn(source, exception=False)
        return facts
    if decision.kind == "ReadMembership":
        return await _read_membership(facts, source)
    try:
        updated, updated_source = await _effect(decision, facts, source)
    except NativeAuthFailure:
        raise
    except Exception:  # noqa: BLE001  # legacy candidate IO errors become failure facts, never authorization verdicts
        _warn(source, exception=True)
        return replace(facts, io_failed=True)
    return await _read(updated, updated_source)


async def _read_membership(facts: FallbackCandidateFacts, source: _Sources) -> FallbackCandidateFacts:
    from litellm.proxy import proxy_server
    from litellm.proxy.auth.auth_checks import get_team_membership

    if source.token.user_id is None or source.token.team_id is None:
        raise NativeAuthRuntimeError("candidate membership read requested without identity")
    dependencies: Final = _Dependencies(
        source.prisma if source.prisma is not None else proxy_server.prisma_client,
        source.cache if source.cache is not None else proxy_server.user_api_key_cache,
        source.logging if source.logging is not None else proxy_server.proxy_logging_obj,
    )
    try:
        membership: Final = await get_team_membership(
            user_id=source.token.user_id,
            team_id=source.token.team_id,
            prisma_client=dependencies.prisma,
            user_api_key_cache=dependencies.cache,
            proxy_logging_obj=dependencies.logging,
        )
    except NativeAuthFailure:
        raise
    except Exception:  # noqa: BLE001  # preserve silent membership IO failure as a native candidate fact
        return replace(facts, membership="Failed")
    return await _read(
        replace(facts, membership="Loaded"), replace(source, dependencies=dependencies, membership=membership)
    )


async def _effect(
    decision: FallbackCandidateDecision, facts: FallbackCandidateFacts, source: _Sources
) -> tuple[FallbackCandidateFacts, _Sources]:
    from litellm.proxy.auth.auth_checks import key_model_aliases_for_auth_check, team_model_aliases_for_auth_check

    router: Final = source.router
    dependencies: Final = source.dependencies
    if router is None or dependencies is None:
        raise NativeAuthRuntimeError("candidate read requested without router or dependencies")
    match decision.kind:
        case "ReadRoute":
            return replace(facts, route=route_context(source.route)), source
        case "ReadUnpriced":
            return replace(facts, unpriced=read_unpriced_model_facts(source.target.served_model, router)), source
        case "ReadCustomerGrants":
            if source.customer is None:
                raise NativeAuthRuntimeError("candidate customer grants requested without source")
            return replace(facts, customer_grants=object_model_grants(source.customer)), source
        case "ReadCustomer":
            if facts.customer_grants is None:
                raise NativeAuthRuntimeError("candidate customer scope requested before grants")
            customer: Final = project_model_facts(
                source.target.model,
                router,
                facts.customer_grants,
                team_model_aliases_for_auth_check(source.token),
                source.token.team_id,
                key_model_aliases_for_auth_check(source.token),
                "customer",
            )
            return replace(facts, customer=customer), source
        case "ReadPrice":
            return replace(facts, price=read_model_cost_facts(source.target.served_model, router)), source
        case "ReadBudgetModels":
            models: Final = (
                source.target.requested,
                source.target.model,
                source.target.served_model,
                router.registry().resolve(source.target.served_model),
            )
            return replace(facts, budget_models=models), source
        case "ReadBudgetScopes":
            budgets: Final = _budget_sources(source.token)
            scopes: Final = tuple(
                CandidateBudgetScope(budget.entity, budget.identity is not None, bool(budget.budgets))
                for budget in budgets
            )
            return replace(facts, budget_scopes=scopes), replace(source, budgets=budgets)
        case "ReadBudget":
            if facts.budget_models is None:
                raise NativeAuthRuntimeError("candidate budget read requested without model names")
            budget: Final = next(budget for budget in source.budgets if budget.entity == decision.entity)
            key_budget: Final = next(budget.budgets for budget in source.budgets if budget.entity == "Key")
            reading: Final = await source.limiter.read_model_budget_raw(
                decision.entity,
                budget.identity,
                source.token.team_id,
                facts.budget_models[decision.model_index],
                budget.budgets,
                key_budget if decision.entity == "Team" else None,
            )
            answer: Final = CandidateBudgetRead(decision.model_index, decision.entity, reading.snapshot)
            return replace(facts, budgets=(*facts.budgets, answer)), source
        case "ReadAuthorization":
            authorization: Final = await load_fallback_authorization_facts(
                source.target.model,
                source.token,
                router,
                source.prisma,
                source.membership,
                source.project,
                True,
                source.team,
            )
            return replace(facts, authorization=authorization), source
        case "ResetAttribution":
            source.token.matched_model_access_groups = []  # rebind-ok: preserve native-selected request-local reset before attribution IO
            return replace(facts, attribution_reset=True), source
        case "ReadAttribution":
            return replace(facts, attribution=await _attribution(source, dependencies, router)), source
        case "StampAttribution":
            source.token.matched_model_access_groups = list(decision.groups)  # rebind-ok: native attribution effect
            return replace(facts, attribution_stamped=True), source
        case "ReadGroupBudgets":
            readings: Final = await read_access_group_budget_facts(
                decision.groups, dependencies.prisma, dependencies.cache
            )
            return replace(
                facts, group_budgets_read=True, group_budgets=tuple(reading.facts for reading in readings)
            ), source
        case "Complete" | "ReadMembership":
            raise NativeAuthRuntimeError("unexpected candidate data read")


async def _attribution(source: _Sources, dependencies: _Dependencies, router: Router) -> ModelGroupAttributionFacts:
    try:
        return await read_model_group_attribution_facts(
            source.target.model,
            source.token,
            source.team,
            source.project,
            router,
            dependencies.prisma,
            dependencies.cache,
            dependencies.logging,
            source.membership,
            True,
        )
    except NativeAuthFailure:
        raise
    except Exception as error:  # noqa: BLE001  # preserve attribution telemetry IO failure separately from candidate IO failure
        verbose_proxy_logger.debug("model access group attribution failed: %s", error)
        return ModelGroupAttributionFacts(io_failed=True)
