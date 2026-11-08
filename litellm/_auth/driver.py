from typing import Protocol, assert_never

from litellm._auth.budget_pricing import ExemptionFacts
from litellm._auth.decision import (
    AlertKeyBudget,
    AlertKeySoftBudget,
    AlertProxyBudget,
    AlertTeamMemberBudget,
    Allow,
    AuthorizeBudgetFallback,
    BudgetEffect,
    CheckBudgetExemption,
    CheckKeyModelBudget,
    CheckTeamMemberModelBudget,
    CheckUserModelBudget,
    Deny,
    Effect,
    EnforceModelAccess,
    EvictPresentedKey,
    FallbackTarget,
    IdentityEffect,
    InspectKey,
    KeyFacts,
    LoadKey,
    LoadKeyContext,
    LoadKeyOwner,
    ModelBudgetEffect,
    ModelBudgetRead,
    ModelBudgetScope,
    Outcome,
    OwnerFacts,
    PrefetchAuthObjects,
    PrepareKey,
    PresentedKey,
    ReadKeySpend,
    ReadModelBudgetScope,
    ReadProxySpend,
    ReadTeamMemberBudget,
    ReadTeamMemberFallbacks,
    ReadTeamMemberModelBudgetScope,
    RerouteTeamMemberRequest,
    ResolveEndUser,
    ResolvePresentedKey,
    RewriteRequestModel,
    SendKeyBudgetAlerts,
    Step,
    TeamMemberBudget,
    TeamMemberEffect,
    TeamMemberModelBudgetScope,
    TeamMemberStep,
    VetTeamMemberFallback,
)
from litellm._auth.fallback_access import FallbackAuthorizationFacts
from litellm._auth.fallback_candidate import FallbackCandidateFacts
from litellm._auth.key_model_access import KeyAndFallbackModelAccessFacts


class ModelBudgetHandler(Protocol):
    async def read_model_budget_scope(self) -> ModelBudgetScope: ...

    async def check_key_model_budget(self, model: str) -> ModelBudgetRead: ...

    async def authorize_budget_fallback(self, model: str) -> FallbackAuthorizationFacts: ...

    async def rewrite_request_model(self, model: str) -> None: ...

    async def check_user_model_budget(self, model: str) -> ModelBudgetRead: ...


class TeamMemberModelBudgetHandler(Protocol):
    async def read_team_member_model_budget_scope(self) -> TeamMemberModelBudgetScope: ...

    async def check_team_member_model_budget(self, model: str) -> ModelBudgetRead: ...

    async def read_team_member_fallbacks(self, model: str) -> tuple[FallbackTarget, ...]: ...

    async def vet_team_member_fallback(self, target: FallbackTarget) -> FallbackCandidateFacts: ...

    async def reroute_team_member_request(self, model: str) -> None: ...


class EffectHandler(ModelBudgetHandler, Protocol):
    async def resolve_end_user(self, prefetch_key: bool) -> None: ...

    async def resolve_presented_key(self, hashed_token: str) -> PresentedKey | None: ...

    async def evict_presented_key(self, hashed_token: str) -> None: ...

    async def load_key(self, hashed_token: str) -> None: ...

    async def prepare_key(self, loaded_from_db: bool) -> None: ...

    async def inspect_key(self) -> KeyFacts: ...

    async def enforce_model_access(self) -> KeyAndFallbackModelAccessFacts: ...

    async def prefetch_auth_objects(self) -> None: ...

    async def load_key_owner(self, user_id: str) -> OwnerFacts | None: ...

    async def check_budget_exemption(self) -> ExemptionFacts: ...

    async def read_team_member_budget(self, user_id: str, team_id: str) -> TeamMemberBudget | None: ...

    async def alert_team_member_budget(self, user_id: str, team_id: str, budget: TeamMemberBudget) -> None: ...

    async def send_key_budget_alerts(self) -> None: ...

    async def read_key_spend(self, max_budget: float) -> float: ...

    async def alert_key_budget(self, spend: float) -> None: ...

    async def alert_key_soft_budget(self) -> None: ...

    async def load_key_context(self) -> None: ...

    async def read_proxy_spend(self) -> float | None: ...

    async def alert_proxy_budget(self, spend: float) -> None: ...


async def _answer(step: Effect, handler: EffectHandler) -> Step:
    match step:
        case (
            ResolveEndUser()
            | ResolvePresentedKey()
            | EvictPresentedKey()
            | LoadKey()
            | PrepareKey()
            | InspectKey()
            | EnforceModelAccess()
            | PrefetchAuthObjects()
            | LoadKeyOwner()
            | CheckBudgetExemption()
            | LoadKeyContext()
        ):
            return await _answer_identity(step, handler)
        case (
            ReadModelBudgetScope()
            | CheckKeyModelBudget()
            | AuthorizeBudgetFallback()
            | RewriteRequestModel()
            | CheckUserModelBudget()
        ):
            return await _answer_model_budget(step, handler)
        case _:
            return await _answer_budget(step, handler)


async def _answer_identity(step: IdentityEffect, handler: EffectHandler) -> Step:
    match step:
        case ResolveEndUser(prefetch_key=prefetch_key, resume=resume):
            await handler.resolve_end_user(prefetch_key)
            return resume()
        case ResolvePresentedKey(hashed_token=hashed_token, resume=resume):
            return resume(await handler.resolve_presented_key(hashed_token))
        case EvictPresentedKey(hashed_token=hashed_token, resume=resume):
            await handler.evict_presented_key(hashed_token)
            return resume()
        case LoadKey(hashed_token=hashed_token, resume=resume):
            await handler.load_key(hashed_token)
            return resume()
        case PrepareKey(loaded_from_db=loaded_from_db, resume=resume):
            await handler.prepare_key(loaded_from_db)
            return resume()
        case InspectKey(resume=resume):
            return resume(await handler.inspect_key())
        case EnforceModelAccess(resume=resume):
            return resume(await handler.enforce_model_access())
        case PrefetchAuthObjects(resume=resume):
            await handler.prefetch_auth_objects()
            return resume()
        case LoadKeyOwner(user_id=user_id, resume=resume):
            return resume(await handler.load_key_owner(user_id))
        case CheckBudgetExemption(resume=resume):
            return resume(await handler.check_budget_exemption())
        case LoadKeyContext(resume=resume):
            await handler.load_key_context()
            return resume()
        case _:
            assert_never(step)


async def _answer_budget(step: BudgetEffect, handler: EffectHandler) -> Step:
    match step:
        case ReadTeamMemberBudget(user_id=user_id, team_id=team_id, resume=resume):
            return resume(await handler.read_team_member_budget(user_id, team_id))
        case AlertTeamMemberBudget(user_id=user_id, team_id=team_id, budget=budget, resume=resume):
            await handler.alert_team_member_budget(user_id, team_id, budget)
            return resume()
        case SendKeyBudgetAlerts(resume=resume):
            await handler.send_key_budget_alerts()
            return resume()
        case ReadKeySpend(max_budget=max_budget, resume=resume):
            return resume(await handler.read_key_spend(max_budget))
        case AlertKeyBudget(spend=spend, resume=resume):
            await handler.alert_key_budget(spend)
            return resume()
        case AlertKeySoftBudget(resume=resume):
            await handler.alert_key_soft_budget()
            return resume()
        case ReadProxySpend(resume=resume):
            return resume(await handler.read_proxy_spend())
        case AlertProxyBudget(spend=spend, resume=resume):
            await handler.alert_proxy_budget(spend)
            return resume()
        case _:
            assert_never(step)


async def _answer_model_budget(step: ModelBudgetEffect, handler: ModelBudgetHandler) -> Step:
    match step:
        case ReadModelBudgetScope(resume=resume):
            return resume(await handler.read_model_budget_scope())
        case CheckKeyModelBudget(model=model, resume=resume):
            return resume(await handler.check_key_model_budget(model))
        case AuthorizeBudgetFallback(model=model, resume=resume):
            return resume(await handler.authorize_budget_fallback(model))
        case RewriteRequestModel(model=model, resume=resume):
            await handler.rewrite_request_model(model)
            return resume()
        case CheckUserModelBudget(model=model, resume=resume):
            return resume(await handler.check_user_model_budget(model))
        case _:
            assert_never(step)


async def drive_model_budgets(step: Step, handler: ModelBudgetHandler) -> Outcome:
    match step:
        case Allow() | Deny():
            return step
        case (
            ReadModelBudgetScope()
            | CheckKeyModelBudget()
            | AuthorizeBudgetFallback()
            | RewriteRequestModel()
            | CheckUserModelBudget()
        ):
            return await drive_model_budgets(await _answer_model_budget(step, handler), handler)
        case _:
            raise RuntimeError(f"{type(step).__name__} is not a model budget effect")


async def drive(step: Step, handler: EffectHandler) -> Outcome:
    match step:
        case Allow() | Deny():
            return step
        case _:
            return await drive(await _answer(step, handler), handler)


async def _answer_team_member(step: TeamMemberEffect, handler: TeamMemberModelBudgetHandler) -> TeamMemberStep:
    match step:
        case ReadTeamMemberModelBudgetScope(resume=resume):
            return resume(await handler.read_team_member_model_budget_scope())
        case CheckTeamMemberModelBudget(model=model, resume=resume):
            return resume(await handler.check_team_member_model_budget(model))
        case ReadTeamMemberFallbacks(model=model, resume=resume):
            return resume(await handler.read_team_member_fallbacks(model))
        case VetTeamMemberFallback(target=target, resume=resume):
            return resume(await handler.vet_team_member_fallback(target))
        case RerouteTeamMemberRequest(model=model, resume=resume):
            await handler.reroute_team_member_request(model)
            return resume()
        case _:
            assert_never(step)


async def drive_team_member_model_budgets(step: TeamMemberStep, handler: TeamMemberModelBudgetHandler) -> Outcome:
    match step:
        case Allow() | Deny():
            return step
        case _:
            return await drive_team_member_model_budgets(await _answer_team_member(step, handler), handler)
