from datetime import datetime
from typing import Protocol

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
    AuthDenial,
    CustomAuthResult,
    Identity,
    KeyAuthentication,
    KeyDenial,
    KeyIdentity,
    OAuth2Denial,
    OAuth2Identity,
    OAuth2ProxyDenial,
    OAuth2ProxyIdentity,
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
from litellm.proxy.auth.core.outcome import ModelBudgetResult, Outcome, RequestCheck
from litellm.proxy.auth.core.validators import CredentialPlan, KeyPlan, Plan


class KeyCore(Protocol):
    async def authenticate(
        self, host: KeyHost, facts: RequestFacts, settings: KeySettings, now: datetime
    ) -> KeyAuthentication: ...

    async def verify(self, host: KeyHost, facts: RequestFacts, now: datetime) -> VirtualKey | KeyDenial: ...


class ModelBudgetCore(Protocol):
    async def check_key_and_user(self, host: ModelBudgetHost) -> ModelBudgetResult: ...

    async def check_user(self, host: ModelBudgetHost) -> ModelBudgetResult: ...

    async def check_team_member(self, host: TeamMemberModelBudgetHost) -> ModelBudgetResult: ...


class AuthPlanner(Protocol):
    def plan(self, facts: PlanFacts, settings: PlanSettings) -> Plan: ...

    def after_custom_auth(self, facts: PlanFacts, settings: PlanSettings, *, issued_key: bool) -> CredentialPlan: ...

    def after_jwt(self, facts: PlanFacts, settings: PlanSettings, *, issued_key: bool) -> KeyPlan: ...

    def oauth2_introspection_applies(self, facts: PlanFacts, settings: PlanSettings) -> bool: ...


class AnonymousCore(Protocol):
    def authenticate(self, reason: AnonymousReason) -> Anonymous: ...


class JwtCore(Protocol):
    async def authenticate(
        self, host: JwtHost, facts: RequestFacts, settings: JwtSettings, now: datetime
    ) -> Identity | AuthDenial: ...


class OAuth2Core(Protocol):
    async def introspect(
        self, host: OAuth2IntrospectionHost, token: str, settings: OAuth2IntrospectionSettings
    ) -> OAuth2Identity | OAuth2Denial: ...

    def proxy(
        self, host: OAuth2ProxyHost, settings: OAuth2ProxySettings
    ) -> OAuth2ProxyIdentity | OAuth2ProxyDenial: ...


class CustomAuthCore(Protocol):
    async def authenticate(self, host: CustomAuthHost, facts: RequestFacts) -> CustomAuthResult: ...


class AuthorizeCore(Protocol):
    async def authorize(
        self, host: AuthorizeHost, identity: KeyIdentity, facts: RequestFacts, settings: AuthorizeSettings
    ) -> Outcome: ...

    async def check_request(
        self, host: RequestCheckHost, facts: CheckFacts, settings: CheckSettings
    ) -> RequestCheck: ...


class AuthCore(Protocol):
    def keys(self) -> KeyCore: ...

    def model_budgets(self) -> ModelBudgetCore: ...

    def plan(self) -> AuthPlanner: ...

    def anonymous(self) -> AnonymousCore: ...

    def jwt(self) -> JwtCore: ...

    def oauth2(self) -> OAuth2Core: ...

    def custom(self) -> CustomAuthCore: ...

    def authorize(self) -> AuthorizeCore: ...
