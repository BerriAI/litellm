from dataclasses import dataclass
from typing import Protocol, TypeAlias

from litellm.proxy.auth.core.answers import (
    FallbackTarget,
    Introspection,
    KeyFacts,
    ModelBudgetScope,
    OrganizationFacts,
    OwnerFacts,
    PresentedKey,
    SigningKey,
    TeamFacts,
    TeamMember,
    TeamMemberBudget,
    TeamMemberModelBudgetScope,
)
from litellm.proxy.auth.core.inputs import RequestFacts
from litellm.proxy.auth.core.outcome import CustomAuthResult, Identity, JwtIdentity, KeyIdentity, ModelBudgetExceeded


class ModelBudgetHost(Protocol):
    async def read_model_budget_scope(self) -> ModelBudgetScope: ...

    async def check_key_model_budget(self, model: str) -> ModelBudgetExceeded | None: ...

    async def authorize_budget_fallback(self, model: str) -> bool: ...

    async def rewrite_request_model(self, model: str) -> None: ...

    async def check_user_model_budget(self, model: str) -> ModelBudgetExceeded | None: ...


class TeamMemberModelBudgetHost(Protocol):
    async def read_team_member_model_budget_scope(self) -> TeamMemberModelBudgetScope: ...

    async def check_team_member_model_budget(self, model: str) -> ModelBudgetExceeded | None: ...

    async def read_team_member_fallbacks(self, model: str) -> tuple[FallbackTarget, ...]: ...

    async def vet_team_member_fallback(self, target: FallbackTarget) -> bool: ...

    async def reroute_team_member_request(self, model: str) -> None: ...


class KeyHost(Protocol):
    def model_budgets(self) -> ModelBudgetHost: ...

    async def resolve_end_user(self, prefetch_key: bool) -> None: ...

    async def resolve_presented_key(self, hashed_token: str) -> PresentedKey | None: ...

    async def evict_presented_key(self, hashed_token: str) -> None: ...

    async def load_key(self, hashed_token: str) -> None: ...

    async def prepare_key(self, loaded_from_db: bool) -> None: ...

    async def inspect_key(self) -> KeyFacts: ...

    async def enforce_model_access(self) -> None: ...

    async def load_key_owner(self, user_id: str) -> OwnerFacts | None: ...

    async def check_budget_exemption(self) -> bool: ...

    async def read_team_member_budget(self, member: TeamMember) -> TeamMemberBudget | None: ...

    async def alert_team_member_budget(self, member: TeamMember, budget: TeamMemberBudget) -> None: ...

    async def send_key_budget_alerts(self) -> None: ...

    async def read_key_spend(self, max_budget: float) -> float: ...

    async def alert_key_budget(self, spend: float) -> None: ...

    async def alert_key_soft_budget(self) -> None: ...

    async def load_key_context(self) -> None: ...

    async def read_proxy_spend(self) -> float | None: ...

    async def alert_proxy_budget(self, spend: float) -> None: ...


@dataclass(frozen=True, slots=True)
class KeySpend:
    hashed_token: str


@dataclass(frozen=True, slots=True)
class TeamSpend:
    team_id: str


@dataclass(frozen=True, slots=True)
class OrganizationSpend:
    org_id: str


@dataclass(frozen=True, slots=True)
class TeamMemberSpend:
    member: TeamMember


SpendCounter: TypeAlias = KeySpend | TeamSpend | OrganizationSpend | TeamMemberSpend


class JwtHost(Protocol):
    async def signing_keys(self, issuer: str) -> tuple[SigningKey, ...]: ...

    async def mapped_key(self, identity: JwtIdentity) -> KeyIdentity | None: ...


class OAuth2IntrospectionHost(Protocol):
    async def introspect(self, token: str) -> Introspection: ...


class OAuth2ProxyHost(Protocol):
    def client_ip(self) -> str | None: ...

    def header(self, name: str) -> str | None: ...


class CustomAuthHost(Protocol):
    async def custom_auth(self, facts: RequestFacts) -> CustomAuthResult: ...


class AuthorizeHost(Protocol):
    async def team(self, team_id: str) -> TeamFacts | None: ...

    async def organization(self, org_id: str) -> OrganizationFacts | None: ...

    async def allowed_models(self, identity: Identity) -> frozenset[str]: ...

    async def spend(self, counter: SpendCounter) -> float: ...
