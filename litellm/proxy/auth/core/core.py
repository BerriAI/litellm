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
    TeamMemberModelBudgetHost,
)
from litellm.proxy.auth.core.inputs import (
    AuthorizeSettings,
    JwtSettings,
    KeySettings,
    OAuth2IntrospectionSettings,
    OAuth2ProxySettings,
    PlanFacts,
    PlanSettings,
    RequestFacts,
)
from litellm.proxy.auth.core.outcome import (
    Anonymous,
    AnonymousReason,
    CredentialPlan,
    CustomAuthResult,
    Denial,
    Identity,
    KeyPlan,
    ModelBudgetResult,
    OAuth2Denial,
    OAuth2Identity,
    OAuth2ProxyDenial,
    OAuth2ProxyIdentity,
    Outcome,
    Plan,
)


class KeyCore(Protocol):
    async def authenticate(
        self, host: KeyHost, facts: RequestFacts, settings: KeySettings, now: datetime
    ) -> Outcome: ...

    async def check(self, host: KeyHost, facts: RequestFacts, settings: KeySettings, now: datetime) -> Outcome: ...


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
    ) -> Identity | Denial: ...


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
        self, host: AuthorizeHost, identity: Identity, facts: RequestFacts, settings: AuthorizeSettings
    ) -> Outcome: ...


class AuthCore(Protocol):
    def keys(self) -> KeyCore: ...

    def model_budgets(self) -> ModelBudgetCore: ...

    def plan(self) -> AuthPlanner: ...

    def anonymous(self) -> AnonymousCore: ...

    def jwt(self) -> JwtCore: ...

    def oauth2(self) -> OAuth2Core: ...

    def custom(self) -> CustomAuthCore: ...

    def authorize(self) -> AuthorizeCore: ...
