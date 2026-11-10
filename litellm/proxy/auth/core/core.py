from datetime import datetime
from typing import Protocol

from litellm.proxy.auth.core.host import (
    AuthorizeHost,
    CustomAuthHost,
    JwtHost,
    KeyHost,
    ModelBudgetHost,
    OAuth2Host,
    TeamMemberModelBudgetHost,
)
from litellm.proxy.auth.core.inputs import (
    AuthorizeSettings,
    JwtSettings,
    KeySettings,
    OAuth2Settings,
    PlanSettings,
    RequestFacts,
)
from litellm.proxy.auth.core.outcome import (
    Anonymous,
    CustomIdentity,
    Denial,
    Identity,
    ModelBudgetResult,
    OAuth2Denial,
    OAuth2Identity,
    Outcome,
    Validator,
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
    def plan(self, facts: RequestFacts, settings: PlanSettings) -> Validator: ...


class AnonymousCore(Protocol):
    def authenticate(self, facts: RequestFacts, settings: PlanSettings) -> Anonymous: ...


class JwtCore(Protocol):
    async def authenticate(
        self, host: JwtHost, facts: RequestFacts, settings: JwtSettings, now: datetime
    ) -> Identity | Denial: ...


class OAuth2Core(Protocol):
    async def authenticate(
        self, host: OAuth2Host, facts: RequestFacts, settings: OAuth2Settings
    ) -> OAuth2Identity | OAuth2Denial: ...


class CustomAuthCore(Protocol):
    async def authenticate(self, host: CustomAuthHost, facts: RequestFacts) -> CustomIdentity: ...


class AuthorizeCore(Protocol):
    async def authorize(
        self, host: AuthorizeHost, identity: Identity, facts: RequestFacts, settings: AuthorizeSettings
    ) -> Outcome: ...


class AuthCore(Protocol):
    def keys(self) -> KeyCore: ...

    def model_budgets(self) -> ModelBudgetCore: ...


class TargetAuthCore(AuthCore, Protocol):
    def plan(self) -> AuthPlanner: ...

    def anonymous(self) -> AnonymousCore: ...

    def jwt(self) -> JwtCore: ...

    def oauth2(self) -> OAuth2Core: ...

    def custom(self) -> CustomAuthCore: ...

    def authorize(self) -> AuthorizeCore: ...
