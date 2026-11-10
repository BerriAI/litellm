import math
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from litellm.proxy.auth.core.answers import AdminKey, KeyFacts, TeamMember
from litellm.proxy.auth.core.core import AuthCore, KeyCore, ModelBudgetCore
from litellm.proxy.auth.core.host import KeyHost, ModelBudgetHost, TeamMemberModelBudgetHost
from litellm.proxy.auth.core.inputs import KeySettings, RequestFacts
from litellm.proxy.auth.core.key import hash_token, is_master_key
from litellm.proxy.auth.core.model_budgets import (
    check_key_and_user_model_budgets,
    check_team_member_model_budgets,
    check_user_model_budgets,
)
from litellm.proxy.auth.core.outcome import (
    Allow,
    CachedAdminKey,
    Deny,
    ExpiredKey,
    KeyBlocked,
    KeyBudgetExceeded,
    KeyWithoutToken,
    MalformedKey,
    MasterKey,
    MasterKeyOnlyRoute,
    MissingKey,
    ModelBudgetExceeded,
    ModelBudgetResult,
    NoDatabase,
    NoMasterKey,
    NotAVirtualKey,
    Outcome,
    OwnerDeactivated,
    TeamMemberBudgetExceeded,
    UserAuthDisabled,
    UserAuthRoute,
    VirtualKey,
    WithinModelBudgets,
)

_USER_AUTH_ROUTE: Final = "/user/auth"
_VIRTUAL_KEY_PREFIX: Final = "sk-"


@dataclass(frozen=True, slots=True)
class _KeyCheck:
    settings: KeySettings
    now: datetime
    is_llm_api_route: bool
    key_label: str


@dataclass(frozen=True, slots=True)
class _WithinKeyBudget:
    budget_throttle_pct: float | None


class PythonCore(AuthCore):
    def keys(self) -> KeyCore:
        return PythonKeyCore()

    def model_budgets(self) -> ModelBudgetCore:
        return PythonModelBudgetCore()


class PythonKeyCore(KeyCore):
    async def authenticate(self, host: KeyHost, facts: RequestFacts, settings: KeySettings, now: datetime) -> Outcome:
        token: Final = facts.token
        if settings.master_key is None:
            return Allow(NoMasterKey(token=token))
        if token is None:
            return Deny(MissingKey())
        if token == "":
            return Deny(MalformedKey())
        if facts.route == _USER_AUTH_ROUTE:
            return Allow(UserAuthRoute()) if settings.allow_user_auth else Deny(UserAuthDisabled())
        if is_master_key(token, settings.master_key):
            await host.resolve_end_user(prefetch_key=False)
            return Allow(MasterKey())
        await host.resolve_end_user(prefetch_key=True)
        return await _presented_key(host, token, facts, settings, now)

    async def check(self, host: KeyHost, facts: RequestFacts, settings: KeySettings, now: datetime) -> Outcome:
        return await _check_key(host, _KeyCheck(settings, now, facts.is_llm_api_route, key_label=facts.token or ""))


class PythonModelBudgetCore(ModelBudgetCore):
    async def check_key_and_user(self, host: ModelBudgetHost) -> ModelBudgetResult:
        return _within_model_budgets(await check_key_and_user_model_budgets(host))

    async def check_user(self, host: ModelBudgetHost) -> ModelBudgetResult:
        return _within_model_budgets(await check_user_model_budgets(host))

    async def check_team_member(self, host: TeamMemberModelBudgetHost) -> ModelBudgetResult:
        return _within_model_budgets(await check_team_member_model_budgets(host))


def _within_model_budgets(exceeded: ModelBudgetExceeded | None) -> ModelBudgetResult:
    return WithinModelBudgets() if exceeded is None else exceeded


async def _presented_key(
    host: KeyHost, token: str, facts: RequestFacts, settings: KeySettings, now: datetime
) -> Outcome:
    hashed_token: Final = hash_token(token)
    presented: Final = await host.resolve_presented_key(hashed_token)
    if isinstance(presented, AdminKey):
        if presented.expires is None or presented.expires >= now:
            return Allow(CachedAdminKey())
        await host.evict_presented_key(hashed_token)
        return Deny(ExpiredKey(expires=presented.expires, now=now, key_label=token))
    if facts.route in settings.master_key_only_routes:
        return Deny(MasterKeyOnlyRoute())
    if not settings.has_db:
        return Deny(NoDatabase())
    if presented is not None:
        await host.prepare_key(loaded_from_db=False)
        return await _check_key(host, _KeyCheck(settings, now, facts.is_llm_api_route, key_label=token))
    if not token.startswith(_VIRTUAL_KEY_PREFIX):
        return Deny(NotAVirtualKey())
    await host.load_key(hashed_token)
    await host.prepare_key(loaded_from_db=True)
    return await _check_key(host, _KeyCheck(settings, now, facts.is_llm_api_route, key_label=hashed_token))


async def _check_key(host: KeyHost, check: _KeyCheck) -> Outcome:
    key: Final = await host.inspect_key()
    if key.blocked:
        return Deny(KeyBlocked())
    await host.enforce_model_access()
    if key.user_id is not None and await _owner_deactivated(host, key.user_id):
        return Deny(OwnerDeactivated(user_id=key.user_id))
    exempt: Final = await host.check_budget_exemption()
    if not exempt and check.settings.has_db and key.team_member is not None:
        team_member_exceeded: Final = await _check_team_member_budget(host, key.team_member)
        if team_member_exceeded is not None:
            return Deny(team_member_exceeded)
    if key.expires is not None and key.expires < check.now:
        return Deny(ExpiredKey(expires=key.expires, now=check.now, key_label=check.key_label))
    if exempt:
        return await _grant(host, check, key, budget_throttle_pct=None)
    await host.send_key_budget_alerts()
    key_budget: Final = await _check_key_budget(host, check, key)
    if isinstance(key_budget, KeyBudgetExceeded):
        return Deny(key_budget)
    if key.soft_budget and key.spend >= key.soft_budget:
        await host.alert_key_soft_budget()
    model_budget_exceeded: Final = await check_key_and_user_model_budgets(host.model_budgets())
    if model_budget_exceeded is not None:
        return Deny(model_budget_exceeded)
    return await _grant(host, check, key, budget_throttle_pct=key_budget.budget_throttle_pct)


async def _owner_deactivated(host: KeyHost, user_id: str) -> bool:
    owner: Final = await host.load_key_owner(user_id)
    return owner is not None and owner.deactivated


async def _check_team_member_budget(host: KeyHost, member: TeamMember) -> TeamMemberBudgetExceeded | None:
    budget: Final = await host.read_team_member_budget(member)
    if budget is None or budget.spend < budget.max_budget:
        return None
    await host.alert_team_member_budget(member, budget)
    return TeamMemberBudgetExceeded(
        user_id=member.user_id, team_id=member.team_id, spend=budget.spend, max_budget=budget.max_budget
    )


async def _check_key_budget(host: KeyHost, check: _KeyCheck, key: KeyFacts) -> _WithinKeyBudget | KeyBudgetExceeded:
    max_budget: Final = key.max_budget
    if not check.is_llm_api_route or max_budget is None:
        return _WithinKeyBudget(budget_throttle_pct=None)
    spend: Final = await host.read_key_spend(max_budget)
    await host.alert_key_budget(spend)
    # A NaN max_budget compares false against any spend, so a non-finite budget means no limit, never a bypass.
    if not math.isfinite(max_budget) or spend < max_budget:
        return _WithinKeyBudget(budget_throttle_pct=None)
    if key.budget_throttle_pct is not None:
        return _WithinKeyBudget(budget_throttle_pct=key.budget_throttle_pct)
    return KeyBudgetExceeded(spend=spend, max_budget=max_budget)


async def _grant(host: KeyHost, check: _KeyCheck, key: KeyFacts, budget_throttle_pct: float | None) -> Outcome:
    await host.load_key_context()
    if check.settings.proxy_max_budget > 0 and check.settings.has_db:
        proxy_spend: Final = await host.read_proxy_spend()
        if proxy_spend is not None:
            await host.alert_proxy_budget(proxy_spend)
    if not key.has_token:
        return Deny(KeyWithoutToken())
    return Allow(VirtualKey(budget_throttle_pct=budget_throttle_pct))
