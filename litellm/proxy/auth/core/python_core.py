import math
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from litellm.proxy.auth.core import jwt as jwt_rules
from litellm.proxy.auth.core import oauth2 as oauth2_rules
from litellm.proxy.auth.core import plan as plan_rules
from litellm.proxy.auth.core import request_checks
from litellm.proxy.auth.core.answers import AdminKey, KeyFacts, TeamMember
from litellm.proxy.auth.core.core import (
    AnonymousCore,
    AuthCore,
    AuthorizeCore,
    AuthPlanner,
    CustomAuthCore,
    JwtCore,
    KeyCore,
    ModelBudgetCore,
    OAuth2Core,
)
from litellm.proxy.auth.core.host import (
    AuthorizeHost,
    CustomAuthHost,
    JwtHost,
    KeyHost,
    ModelBudgetHost,
    OAuth2IntrospectionHost,
    OAuth2ProxyHost,
    RequestCheckHost,
    TeamMemberModelBudgetHost,
)
from litellm.proxy.auth.core.identity import (
    Anonymous,
    AnonymousReason,
    CachedAdminKey,
    CustomAuthResult,
    ExpiredKey,
    JwtAuthentication,
    KeyAuthentication,
    KeyBlocked,
    KeyDenial,
    KeyIdentity,
    KeyWithoutToken,
    MalformedKey,
    MasterKey,
    MissingKey,
    NoDatabase,
    NoMasterKey,
    NotAVirtualKey,
    OAuth2Denial,
    OAuth2Identity,
    OAuth2ProxyDenial,
    OAuth2ProxyIdentity,
    OwnerDeactivated,
    UserAuthRoute,
    VirtualKey,
)
from litellm.proxy.auth.core.inputs import (
    AuthorizeSettings,
    CheckFacts,
    CheckSettings,
    JwtSettings,
    KeySettings,
    OAuth2IntrospectionSettings,
    OAuth2ProxySettings,
    PlanFacts,
    PlanSettings,
    RequestFacts,
)
from litellm.proxy.auth.core.key import hash_token, is_master_key
from litellm.proxy.auth.core.model_budgets import (
    check_key_and_user_model_budgets,
    check_team_member_model_budgets,
    check_user_model_budgets,
)
from litellm.proxy.auth.core.outcome import (
    Allow,
    Deny,
    KeyBudgetExceeded,
    MasterKeyOnlyRoute,
    ModelBudgetExceeded,
    ModelBudgetResult,
    Outcome,
    RequestCheck,
    TeamMemberBudgetExceeded,
    UserAuthDisabled,
    WithinModelBudgets,
)
from litellm.proxy.auth.core.validators import CredentialPlan, KeyPlan, Plan

_USER_AUTH_ROUTE: Final = "/user/auth"
_VIRTUAL_KEY_PREFIX: Final = "sk-"


@dataclass(frozen=True, slots=True)
class _WithinKeyBudget:
    budget_throttle_pct: float | None


class PythonCore(AuthCore):
    def keys(self) -> KeyCore:
        return PythonKeyCore()

    def model_budgets(self) -> ModelBudgetCore:
        return PythonModelBudgetCore()

    def plan(self) -> AuthPlanner:
        return PythonAuthPlanner()

    def anonymous(self) -> AnonymousCore:
        return PythonAnonymousCore()

    def jwt(self) -> JwtCore:
        return PythonJwtCore()

    def oauth2(self) -> OAuth2Core:
        return PythonOAuth2Core()

    def custom(self) -> CustomAuthCore:
        return PythonCustomAuthCore()

    def authorize(self) -> AuthorizeCore:
        return PythonAuthorizeCore()


class PythonAuthPlanner(AuthPlanner):
    def plan(self, facts: PlanFacts, settings: PlanSettings) -> Plan:
        return plan_rules.plan(facts, settings)

    def after_custom_auth(self, facts: PlanFacts, settings: PlanSettings, *, issued_key: bool) -> CredentialPlan:
        return plan_rules.after_custom_auth(facts, settings, issued_key=issued_key)

    def after_jwt(self, facts: PlanFacts, settings: PlanSettings, *, issued_key: bool) -> KeyPlan:
        return plan_rules.after_jwt(facts, settings, issued_key=issued_key)

    def oauth2_introspection_applies(self, facts: PlanFacts, settings: PlanSettings) -> bool:
        return plan_rules.oauth2_introspection_applies(facts, settings)


class PythonAnonymousCore(AnonymousCore):
    def authenticate(self, reason: AnonymousReason) -> Anonymous:
        return Anonymous(reason)


class PythonJwtCore(JwtCore):
    async def authenticate(self, host: JwtHost, settings: JwtSettings) -> JwtAuthentication:
        return await jwt_rules.authenticate(host, settings)


class PythonOAuth2Core(OAuth2Core):
    async def introspect(
        self, host: OAuth2IntrospectionHost, token: str, settings: OAuth2IntrospectionSettings
    ) -> OAuth2Identity | OAuth2Denial:
        return await oauth2_rules.introspect(host, token, settings)

    def proxy(self, host: OAuth2ProxyHost, settings: OAuth2ProxySettings) -> OAuth2ProxyIdentity | OAuth2ProxyDenial:
        return oauth2_rules.proxy(host, settings)


class PythonCustomAuthCore(CustomAuthCore):
    async def authenticate(self, host: CustomAuthHost, facts: RequestFacts) -> CustomAuthResult:
        return await host.custom_auth(facts)


class PythonKeyCore(KeyCore):
    async def authenticate(
        self, host: KeyHost, facts: RequestFacts, settings: KeySettings, now: datetime
    ) -> KeyAuthentication:
        token: Final = facts.token
        if settings.master_key is None:
            return NoMasterKey(token=token)
        if token is None:
            return MissingKey()
        if token == "":
            return MalformedKey()
        if facts.route == _USER_AUTH_ROUTE:
            return UserAuthRoute()
        if is_master_key(token, settings.master_key):
            await host.resolve_end_user(prefetch_key=False)
            return MasterKey()
        await host.resolve_end_user(prefetch_key=True)
        return await _presented_key(host, token, settings, now)

    async def verify(self, host: KeyHost, facts: RequestFacts, now: datetime) -> VirtualKey | KeyDenial:
        return await _verify_key(host, now, key_label=facts.token or "")


class PythonAuthorizeCore(AuthorizeCore):
    async def authorize(
        self, host: AuthorizeHost, identity: KeyIdentity, facts: RequestFacts, settings: AuthorizeSettings
    ) -> Outcome:
        match identity:
            case UserAuthRoute():
                return Allow(budget_throttle_pct=None) if settings.allow_user_auth else Deny(UserAuthDisabled())
            case NoMasterKey() | MasterKey() | CachedAdminKey():
                return Allow(budget_throttle_pct=None)
            case VirtualKey(key=key):
                return await _authorize_virtual_key(host, key, facts, settings)

    async def check_request(self, host: RequestCheckHost, facts: CheckFacts, settings: CheckSettings) -> RequestCheck:
        return await request_checks.check_request(host, facts, settings)


class PythonModelBudgetCore(ModelBudgetCore):
    async def check_key_and_user(self, host: ModelBudgetHost) -> ModelBudgetResult:
        return _within_model_budgets(await check_key_and_user_model_budgets(host))

    async def check_user(self, host: ModelBudgetHost) -> ModelBudgetResult:
        return _within_model_budgets(await check_user_model_budgets(host))

    async def check_team_member(self, host: TeamMemberModelBudgetHost) -> ModelBudgetResult:
        return _within_model_budgets(await check_team_member_model_budgets(host))


def _within_model_budgets(exceeded: ModelBudgetExceeded | None) -> ModelBudgetResult:
    return WithinModelBudgets() if exceeded is None else exceeded


async def _presented_key(host: KeyHost, token: str, settings: KeySettings, now: datetime) -> KeyAuthentication:
    hashed_token: Final = hash_token(token)
    presented: Final = await host.resolve_presented_key(hashed_token)
    if isinstance(presented, AdminKey):
        if presented.expires is None or presented.expires >= now:
            return CachedAdminKey()
        await host.evict_presented_key(hashed_token)
        return ExpiredKey(expires=presented.expires, now=now, key_label=token)
    if not settings.has_db:
        return NoDatabase()
    if presented is not None:
        await host.prepare_key(loaded_from_db=False)
        return await _verify_key(host, now, key_label=token)
    if not token.startswith(_VIRTUAL_KEY_PREFIX):
        return NotAVirtualKey()
    await host.load_key(hashed_token)
    await host.prepare_key(loaded_from_db=True)
    return await _verify_key(host, now, key_label=hashed_token)


async def _verify_key(host: KeyHost, now: datetime, key_label: str) -> VirtualKey | KeyDenial:
    key: Final = await host.inspect_key()
    if key.blocked:
        return KeyBlocked()
    if key.user_id is not None and await _owner_deactivated(host, key.user_id):
        return OwnerDeactivated(user_id=key.user_id)
    if key.expires is not None and key.expires < now:
        return ExpiredKey(expires=key.expires, now=now, key_label=key_label)
    if not key.has_token:
        return KeyWithoutToken()
    return VirtualKey(key=key)


async def _owner_deactivated(host: KeyHost, user_id: str) -> bool:
    owner: Final = await host.load_key_owner(user_id)
    return owner is not None and owner.deactivated


async def _authorize_virtual_key(
    host: AuthorizeHost, key: KeyFacts, facts: RequestFacts, settings: AuthorizeSettings
) -> Outcome:
    if facts.route in settings.master_key_only_routes:
        return Deny(MasterKeyOnlyRoute())
    await host.enforce_model_access()
    exempt: Final = await host.check_budget_exemption()
    if not exempt and settings.has_db and key.team_member is not None:
        team_member_exceeded: Final = await _check_team_member_budget(host, key.team_member)
        if team_member_exceeded is not None:
            return Deny(team_member_exceeded)
    if exempt:
        return await _grant(host, settings, budget_throttle_pct=None)
    await host.send_key_budget_alerts()
    key_budget: Final = await _check_key_budget(host, key, is_llm_api_route=facts.is_llm_api_route)
    if isinstance(key_budget, KeyBudgetExceeded):
        return Deny(key_budget)
    if key.soft_budget and key.spend >= key.soft_budget:
        await host.alert_key_soft_budget()
    model_budget_exceeded: Final = await check_key_and_user_model_budgets(host.model_budgets())
    if model_budget_exceeded is not None:
        return Deny(model_budget_exceeded)
    return await _grant(host, settings, budget_throttle_pct=key_budget.budget_throttle_pct)


async def _check_team_member_budget(host: AuthorizeHost, member: TeamMember) -> TeamMemberBudgetExceeded | None:
    budget: Final = await host.read_team_member_budget(member)
    if budget is None or budget.spend < budget.max_budget:
        return None
    await host.alert_team_member_budget(member, budget)
    return TeamMemberBudgetExceeded(
        user_id=member.user_id, team_id=member.team_id, spend=budget.spend, max_budget=budget.max_budget
    )


async def _check_key_budget(
    host: AuthorizeHost, key: KeyFacts, *, is_llm_api_route: bool
) -> _WithinKeyBudget | KeyBudgetExceeded:
    max_budget: Final = key.max_budget
    if not is_llm_api_route or max_budget is None:
        return _WithinKeyBudget(budget_throttle_pct=None)
    spend: Final = await host.read_key_spend(max_budget)
    await host.alert_key_budget(spend)
    # A NaN max_budget compares false against any spend, so a non-finite budget means no limit, never a bypass.
    if not math.isfinite(max_budget) or spend < max_budget:
        return _WithinKeyBudget(budget_throttle_pct=None)
    if key.budget_throttle_pct is not None:
        return _WithinKeyBudget(budget_throttle_pct=key.budget_throttle_pct)
    return KeyBudgetExceeded(spend=spend, max_budget=max_budget)


async def _grant(host: AuthorizeHost, settings: AuthorizeSettings, budget_throttle_pct: float | None) -> Outcome:
    await host.load_key_context()
    if settings.proxy_max_budget > 0 and settings.has_db:
        proxy_spend: Final = await host.read_proxy_spend()
        if proxy_spend is not None:
            await host.alert_proxy_budget(proxy_spend)
    return Allow(budget_throttle_pct=budget_throttle_pct)
