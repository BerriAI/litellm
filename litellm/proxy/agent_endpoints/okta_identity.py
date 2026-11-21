import re
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import (
    OKTA_CUSTOM_ISSUER_PATTERN,
    AgentIdentityBinding,
    AgentIdentityFailure,
    ManagedAgentContext,
)

if TYPE_CHECKING:
    from litellm.proxy._types import LiteLLM_JWTAuth

_SCOPES: Final = TypeAdapter(list[str], config=ConfigDict(strict=True))
_DISABLED: Final = AgentIdentityFailure(message="Agent is disabled or no longer bound to an identity")


@dataclass(frozen=True, slots=True)
class OktaIssuer:
    org: str
    accepts_access_tokens: bool


def okta_issuers(litellm_jwtauth: "LiteLLM_JWTAuth") -> Mapping[str, OktaIssuer]:
    return MappingProxyType(
        {
            item.issuer: OktaIssuer(item.okta_org, re.fullmatch(OKTA_CUSTOM_ISSUER_PATTERN, item.issuer) is not None)
            for item in litellm_jwtauth.issuers or ()
            if item.agent_identity_provider == "okta" and item.okta_org
        }
    )


class PersonLookup(Protocol):
    def __call__(self, claims: Mapping[str, object], /) -> Awaitable[str | AgentIdentityFailure | None]: ...


@dataclass(frozen=True, slots=True)
class OktaAgents:
    issuers: Mapping[str, OktaIssuer]
    person: PersonLookup


class BindingLookup(Protocol):
    def __call__(
        self, org: str, client_id: str, /, *, check_retired: bool
    ) -> Awaitable[AgentResponse | AgentIdentityFailure | None]: ...


@dataclass(frozen=True, slots=True)
class ClientCredentials:
    client: str


@dataclass(frozen=True, slots=True)
class Person:
    client: str
    uid: str


OktaToken: TypeAlias = ClientCredentials | Person


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def okta_token(claims: Mapping[str, object]) -> OktaToken | None:
    client: Final = _text(claims.get("cid"))
    uid: Final = _text(claims.get("uid"))
    if client is not None and uid is not None and "act" not in claims:
        return Person(client, uid)
    if client is not None and uid is None and claims.get("sub") == client:
        return ClientCredentials(client)
    return None


def _scope_failure(binding: AgentIdentityBinding, claims: Mapping[str, object]) -> AgentIdentityFailure | None:
    try:
        granted: Final = _SCOPES.validate_python(claims.get("scp", []))
    except ValidationError:
        return AgentIdentityFailure(message="Invalid scope claim")
    if not frozenset(binding.required_scopes).issubset(granted):
        return AgentIdentityFailure(message="Token lacks the required scopes")
    return None


def _bound_identity(agent: AgentResponse) -> AgentIdentityBinding | None:
    binding: Final = agent.identity if agent.identity_managed and agent.enabled else None
    return binding if binding is not None and binding.active else None


def _autonomous(
    agent: AgentResponse, claims: Mapping[str, object], client: str
) -> ManagedAgentContext | AgentIdentityFailure:
    binding: Final = _bound_identity(agent)
    if binding is None:
        return _DISABLED
    if (claims.get("iss"), client) != (binding.issuer, binding.client_id):
        return AgentIdentityFailure(message="Token does not match the registered Okta application")
    if agent.execution_mode == "delegated":
        return AgentIdentityFailure(message="Agent is not enabled for this execution mode")
    scope_failure: Final = _scope_failure(binding, claims)
    if scope_failure is not None:
        return scope_failure
    return ManagedAgentContext(
        agent_id=agent.agent_id, binding_revision=binding.revision, mode="autonomous", subject_oid=client
    )


async def _delegated(
    agent: AgentResponse, claims: Mapping[str, object], client: str, uid: str, person: PersonLookup
) -> ManagedAgentContext | AgentIdentityFailure | None:
    binding: Final = _bound_identity(agent)
    if binding is None:
        return _DISABLED
    if (claims.get("iss"), client) != (binding.issuer, binding.client_id):
        return AgentIdentityFailure(message="Token does not match the registered Okta application")
    scope_failure: Final = _scope_failure(binding, claims)
    if scope_failure is not None:
        return scope_failure
    user_id: Final = await person(claims)
    if user_id is None or isinstance(user_id, AgentIdentityFailure):
        return user_id
    return ManagedAgentContext(
        agent_id=agent.agent_id, binding_revision=binding.revision, mode="delegated", user_id=user_id, subject_oid=uid
    )


async def resolve_okta_claims(
    claims: Mapping[str, object], issuer: OktaIssuer, lookup: BindingLookup, person: PersonLookup
) -> ManagedAgentContext | AgentIdentityFailure | None:
    if not issuer.accepts_access_tokens:
        return None
    match okta_token(claims):
        case ClientCredentials(client=client):
            agent: Final = await lookup(issuer.org, client, check_retired=True)
            if agent is None or isinstance(agent, AgentIdentityFailure):
                return agent
            return _autonomous(agent, claims, client)
        case Person(client=client, uid=uid):
            bound: Final = await lookup(issuer.org, client, check_retired=False)
            if bound is None or isinstance(bound, AgentIdentityFailure) or bound.execution_mode == "autonomous":
                return bound if isinstance(bound, AgentIdentityFailure) else None
            return await _delegated(bound, claims, client, uid, person)
        case None:
            return None
