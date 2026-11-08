import math
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from litellm._auth.decision import (
    AdminKey,
    AlertKeyBudget,
    AlertKeySoftBudget,
    AlertProxyBudget,
    AlertTeamMemberBudget,
    Allow,
    CachedAdminKey,
    CheckBudgetExemption,
    Deny,
    EnforceModelAccess,
    EvictPresentedKey,
    ExpiredKey,
    InspectKey,
    KeyBlocked,
    KeyBudgetExceeded,
    KeyFacts,
    KeyWithoutToken,
    LoadKey,
    LoadKeyContext,
    LoadKeyOwner,
    MalformedKey,
    MasterKey,
    MasterKeyOnlyRoute,
    MissingKey,
    NoDatabase,
    NoMasterKey,
    NonAdminKey,
    NotAVirtualKey,
    OwnerDeactivated,
    OwnerFacts,
    PrepareKey,
    PresentedKey,
    ReadKeySpend,
    ReadProxySpend,
    ReadTeamMemberBudget,
    RequestFacts,
    ResolveEndUser,
    ResolvePresentedKey,
    SendKeyBudgetAlerts,
    Settings,
    Step,
    TeamMemberBudget,
    TeamMemberBudgetExceeded,
    TeamMemberStep,
    UserAuthDisabled,
    UserAuthRoute,
    VirtualKey,
    WithinModelBudgets,
)
from litellm._auth.key import hash_token, is_master_key
from tests.unit._auth.reference_model_budgets import (
    check_model_budgets,
    check_team_member_model_budgets,
    check_user_model_budgets,
)

_USER_AUTH_ROUTE: Final = "/user/auth"
_VIRTUAL_KEY_PREFIX: Final = "sk-"


@dataclass(frozen=True, slots=True)
class _KeyCheck:
    settings: Settings
    now: datetime
    is_llm_api_route: bool
    key_label: str


class PythonCore:
    def start(self, facts: RequestFacts, settings: Settings, now: datetime) -> Step:
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
            return ResolveEndUser(prefetch_key=False, resume=lambda: Allow(MasterKey()))
        hashed_token: Final = hash_token(token)
        return ResolveEndUser(
            prefetch_key=True,
            resume=lambda: ResolvePresentedKey(
                hashed_token=hashed_token,
                resume=lambda presented: _after_presented_key(presented, token, hashed_token, facts, settings, now),
            ),
        )

    def start_resolved_key(self, facts: RequestFacts, settings: Settings, now: datetime) -> Step:
        return _check_key(_KeyCheck(settings, now, facts.is_llm_api_route, key_label=facts.token or ""))

    def check_model_budgets(self) -> Step:
        return check_model_budgets(then=lambda: Allow(WithinModelBudgets()))

    def check_user_model_budgets(self) -> Step:
        return check_user_model_budgets(then=lambda: Allow(WithinModelBudgets()))

    def check_team_member_model_budgets(self) -> TeamMemberStep:
        return check_team_member_model_budgets()


def _after_presented_key(
    presented: PresentedKey | None,
    token: str,
    hashed_token: str,
    facts: RequestFacts,
    settings: Settings,
    now: datetime,
) -> Step:
    match presented:
        case AdminKey(expires=expires) if expires is not None and expires < now:
            return EvictPresentedKey(
                hashed_token=hashed_token,
                resume=lambda: Deny(ExpiredKey(expires=expires, now=now, key_label=token)),
            )
        case AdminKey():
            return Allow(CachedAdminKey())
        case NonAdminKey() | None:
            pass
    if facts.route in settings.master_key_only_routes:
        return Deny(MasterKeyOnlyRoute())
    if not settings.has_db:
        return Deny(NoDatabase())
    if presented is not None:
        cached: Final = _KeyCheck(settings, now, facts.is_llm_api_route, key_label=token)
        return PrepareKey(loaded_from_db=False, resume=lambda: _check_key(cached))
    if not token.startswith(_VIRTUAL_KEY_PREFIX):
        return Deny(NotAVirtualKey())
    loaded: Final = _KeyCheck(settings, now, facts.is_llm_api_route, key_label=hashed_token)
    return LoadKey(
        hashed_token=hashed_token,
        resume=lambda: PrepareKey(loaded_from_db=True, resume=lambda: _check_key(loaded)),
    )


def _check_key(check: _KeyCheck) -> Step:
    return InspectKey(resume=lambda key: _after_inspect(check, key))


def _after_inspect(check: _KeyCheck, key: KeyFacts) -> Step:
    if key.blocked:
        return Deny(KeyBlocked())
    return EnforceModelAccess(resume=lambda: _check_owner(check, key))


def _check_owner(check: _KeyCheck, key: KeyFacts) -> Step:
    user_id: Final = key.user_id
    if user_id is None:
        return _check_budget_exemption(check, key)
    return LoadKeyOwner(user_id=user_id, resume=lambda owner: _after_owner(check, key, user_id, owner))


def _after_owner(check: _KeyCheck, key: KeyFacts, user_id: str, owner: OwnerFacts | None) -> Step:
    if owner is not None and owner.deactivated:
        return Deny(OwnerDeactivated(user_id=user_id))
    return _check_budget_exemption(check, key)


def _check_budget_exemption(check: _KeyCheck, key: KeyFacts) -> Step:
    return CheckBudgetExemption(resume=lambda exempt: _check_team_member_budget(check, key, exempt))


def _check_team_member_budget(check: _KeyCheck, key: KeyFacts, exempt: bool) -> Step:
    user_id: Final = key.user_id
    team_id: Final = key.team_id
    if exempt or key.team_member_spend is None or not check.settings.has_db or user_id is None or team_id is None:
        return _check_expiry(check, key, exempt)
    return ReadTeamMemberBudget(
        user_id=user_id,
        team_id=team_id,
        resume=lambda budget: _after_team_member_budget(check, key, user_id, team_id, budget),
    )


def _after_team_member_budget(
    check: _KeyCheck, key: KeyFacts, user_id: str, team_id: str, budget: TeamMemberBudget | None
) -> Step:
    if budget is None or budget.spend < budget.max_budget:
        return _check_expiry(check, key, exempt=False)
    return AlertTeamMemberBudget(
        user_id=user_id,
        team_id=team_id,
        budget=budget,
        resume=lambda: Deny(
            TeamMemberBudgetExceeded(user_id=user_id, team_id=team_id, spend=budget.spend, max_budget=budget.max_budget)
        ),
    )


def _check_expiry(check: _KeyCheck, key: KeyFacts, exempt: bool) -> Step:
    if key.expires is not None and key.expires < check.now:
        return Deny(ExpiredKey(expires=key.expires, now=check.now, key_label=check.key_label))
    if exempt:
        return _load_context(check, key, budget_throttle_pct=None)
    return SendKeyBudgetAlerts(resume=lambda: _check_key_budget(check, key))


def _check_key_budget(check: _KeyCheck, key: KeyFacts) -> Step:
    max_budget: Final = key.max_budget
    if not check.is_llm_api_route or max_budget is None:
        return _check_soft_budget(check, key, budget_throttle_pct=None)
    return ReadKeySpend(
        max_budget=max_budget,
        resume=lambda spend: AlertKeyBudget(
            spend=spend, resume=lambda: _after_key_spend(check, key, spend, max_budget)
        ),
    )


def _after_key_spend(check: _KeyCheck, key: KeyFacts, spend: float, max_budget: float) -> Step:
    # A NaN max_budget compares false against any spend, so a non-finite budget means no limit, never a bypass.
    if not math.isfinite(max_budget) or spend < max_budget:
        return _check_soft_budget(check, key, budget_throttle_pct=None)
    if key.budget_throttle_pct is not None:
        return _check_soft_budget(check, key, budget_throttle_pct=key.budget_throttle_pct)
    return Deny(KeyBudgetExceeded(spend=spend, max_budget=max_budget))


def _check_soft_budget(check: _KeyCheck, key: KeyFacts, budget_throttle_pct: float | None) -> Step:
    def model_budgets() -> Step:
        return check_model_budgets(then=lambda: _load_context(check, key, budget_throttle_pct))

    if key.soft_budget and key.spend >= key.soft_budget:
        return AlertKeySoftBudget(resume=model_budgets)
    return model_budgets()


def _load_context(check: _KeyCheck, key: KeyFacts, budget_throttle_pct: float | None) -> Step:
    return LoadKeyContext(resume=lambda: _check_proxy_budget(check, key, budget_throttle_pct))


def _check_proxy_budget(check: _KeyCheck, key: KeyFacts, budget_throttle_pct: float | None) -> Step:
    if not (check.settings.proxy_max_budget > 0 and check.settings.has_db):
        return _grant(key, budget_throttle_pct)
    return ReadProxySpend(
        resume=lambda spend: (
            _grant(key, budget_throttle_pct)
            if spend is None
            else AlertProxyBudget(spend=spend, resume=lambda: _grant(key, budget_throttle_pct))
        )
    )


def _grant(key: KeyFacts, budget_throttle_pct: float | None) -> Step:
    if not key.has_token:
        return Deny(KeyWithoutToken())
    return Allow(VirtualKey(budget_throttle_pct=budget_throttle_pct))
