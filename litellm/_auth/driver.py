from typing import Protocol, assert_never

from litellm._auth.decision import (
    AlertKeyBudget,
    AlertKeySoftBudget,
    AlertProxyBudget,
    AlertTeamMemberBudget,
    Allow,
    BudgetEffect,
    CheckBudgetExemption,
    Deny,
    Effect,
    EnforceModelAccess,
    EnforceModelBudgets,
    EvictPresentedKey,
    IdentityEffect,
    InspectKey,
    KeyFacts,
    LoadKey,
    LoadKeyContext,
    LoadKeyOwner,
    Outcome,
    OwnerFacts,
    PrepareKey,
    PresentedKey,
    ReadKeySpend,
    ReadProxySpend,
    ReadTeamMemberBudget,
    ResolveEndUser,
    ResolvePresentedKey,
    SendKeyBudgetAlerts,
    Step,
    TeamMemberBudget,
)


class EffectHandler(Protocol):
    async def resolve_end_user(self, prefetch_key: bool) -> None: ...

    async def resolve_presented_key(self, hashed_token: str) -> PresentedKey | None: ...

    async def evict_presented_key(self, hashed_token: str) -> None: ...

    async def load_key(self, hashed_token: str) -> None: ...

    async def prepare_key(self, loaded_from_db: bool) -> None: ...

    async def inspect_key(self) -> KeyFacts: ...

    async def enforce_model_access(self) -> None: ...

    async def load_key_owner(self, user_id: str) -> OwnerFacts | None: ...

    async def check_budget_exemption(self) -> bool: ...

    async def read_team_member_budget(self, user_id: str, team_id: str) -> TeamMemberBudget | None: ...

    async def alert_team_member_budget(self, user_id: str, team_id: str, budget: TeamMemberBudget) -> None: ...

    async def send_key_budget_alerts(self) -> None: ...

    async def read_key_spend(self, max_budget: float) -> float: ...

    async def alert_key_budget(self, spend: float) -> None: ...

    async def alert_key_soft_budget(self) -> None: ...

    async def enforce_model_budgets(self) -> None: ...

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
            | LoadKeyOwner()
            | CheckBudgetExemption()
            | LoadKeyContext()
        ):
            return await _answer_identity(step, handler)
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
            await handler.enforce_model_access()
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
        case EnforceModelBudgets(resume=resume):
            await handler.enforce_model_budgets()
            return resume()
        case ReadProxySpend(resume=resume):
            return resume(await handler.read_proxy_spend())
        case AlertProxyBudget(spend=spend, resume=resume):
            await handler.alert_proxy_budget(spend)
            return resume()
        case _:
            assert_never(step)


async def drive(step: Step, handler: EffectHandler) -> Outcome:
    match step:
        case Allow() | Deny():
            return step
        case _:
            return await drive(await _answer(step, handler), handler)
