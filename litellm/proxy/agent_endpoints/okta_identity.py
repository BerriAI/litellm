import re
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Protocol, TypeAlias

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from litellm.types.agents import AgentResponse
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.agent_identity import (
    OKTA_CUSTOM_ISSUER_PATTERN,
    AgentIdentityBinding,
    AgentIdentityFailure,
    AgentStamp,
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


@dataclass(frozen=True, slots=True)
class Actor:
    client: str
    uid: str
    actor: str | None
    chain: tuple[str, ...]


OktaToken: TypeAlias = ClientCredentials | Person | Actor
_MAX_ACTORS: Final = 5


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


class _ActClaim(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    sub: Annotated[str, Field(min_length=1)]
    act: object = None


def _act_claim(value: object) -> _ActClaim | None:
    try:
        return _ActClaim.model_validate(value)
    except ValidationError:
        return None


def _next_actor(chain: tuple[tuple[str, ...], object], _level: int) -> tuple[tuple[str, ...], object]:
    actors, node = chain
    claim: Final = _act_claim(node)
    return (actors, None) if claim is None else ((*actors, claim.sub), claim.act)


def _chain(act: object) -> tuple[str, ...]:
    return reduce(_next_actor, range(_MAX_ACTORS), ((), act))[0]


def okta_token(claims: Mapping[str, object]) -> OktaToken | None:
    client: Final = _text(claims.get("cid"))
    uid: Final = _text(claims.get("uid"))
    if client is not None and uid is not None and "act" in claims:
        act: Final = claims.get("act")
        top: Final = _act_claim(act)
        return Actor(client, uid, top.sub if top is not None else None, _chain(act))
    if client is not None and uid is not None:
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


def _delegated_binding(
    agent: AgentResponse, claims: Mapping[str, object], client: str
) -> AgentIdentityBinding | AgentIdentityFailure:
    binding: Final = _bound_identity(agent)
    if binding is None:
        return _DISABLED
    if (claims.get("iss"), client) != (binding.issuer, binding.client_id):
        return AgentIdentityFailure(message="Token does not match the registered Okta application")
    return binding


async def _delegated(
    agent: AgentResponse,
    binding: AgentIdentityBinding,
    claims: Mapping[str, object],
    uid: str,
    person: PersonLookup,
    chain: tuple[str, ...] = (),
) -> ManagedAgentContext | AgentIdentityFailure | None:
    scope_failure: Final = _scope_failure(binding, claims)
    if scope_failure is not None:
        return scope_failure
    user_id: Final = await person(claims)
    if user_id is None or isinstance(user_id, AgentIdentityFailure):
        return user_id
    return ManagedAgentContext(
        agent_id=agent.agent_id,
        binding_revision=binding.revision,
        mode="delegated",
        user_id=user_id,
        subject_oid=uid,
        actor_chain=chain,
    )


async def _delegated_agent(
    lookup: BindingLookup, org: str, client_id: str
) -> AgentResponse | AgentIdentityFailure | None:
    bound: Final = await lookup(org, client_id, check_retired=False)
    if isinstance(bound, AgentIdentityFailure):
        return bound
    return None if bound is None or bound.execution_mode == "autonomous" else bound


async def _person_token(
    claims: Mapping[str, object], issuer: OktaIssuer, lookup: BindingLookup, person: PersonLookup, token: Person | Actor
) -> ManagedAgentContext | AgentIdentityFailure | None:
    agent: Final = await _delegated_agent(lookup, issuer.org, token.client)
    if agent is None or isinstance(agent, AgentIdentityFailure):
        return agent
    binding: Final = _delegated_binding(agent, claims, token.client)
    if isinstance(binding, AgentIdentityFailure):
        return binding
    if isinstance(token, Actor) and token.actor is None:
        return AgentIdentityFailure(message="Invalid actor claim")
    return await _delegated(agent, binding, claims, token.uid, person, token.chain if isinstance(token, Actor) else ())


async def _actor_token(
    claims: Mapping[str, object], issuer: OktaIssuer, lookup: BindingLookup, person: PersonLookup, token: Actor
) -> ManagedAgentContext | AgentIdentityFailure | None:
    if token.actor is None:
        return await _person_token(claims, issuer, lookup, person, token)
    agent: Final = await _delegated_agent(lookup, issuer.org, token.actor)
    if isinstance(agent, AgentIdentityFailure):
        return agent
    if agent is None:
        return await _person_token(claims, issuer, lookup, person, token)
    binding: Final = _delegated_binding(agent, claims, token.actor)
    if isinstance(binding, AgentIdentityFailure):
        return binding
    if token.actor != token.client:
        return AgentIdentityFailure(message="Token actor does not match its client")
    return await _delegated(agent, binding, claims, token.uid, person, token.chain)


async def resolve_okta_claims(
    claims: Mapping[str, object], issuer: OktaIssuer, lookup: BindingLookup, person: PersonLookup
) -> ManagedAgentContext | AgentIdentityFailure | None:
    if not issuer.accepts_access_tokens:
        return None
    token: Final = okta_token(claims)
    match token:
        case ClientCredentials(client=client):
            agent: Final = await lookup(issuer.org, client, check_retired=True)
            if agent is None or isinstance(agent, AgentIdentityFailure):
                return agent
            return _autonomous(agent, claims, client)
        case Person():
            return await _person_token(claims, issuer, lookup, person, token)
        case Actor():
            return await _actor_token(claims, issuer, lookup, person, token)
        case None:
            return None


class AgentGrants(Protocol):
    def __call__(self, user_id: str, team_id: str | None, /) -> Awaitable[frozenset[str]]: ...


class RecordAuthentication(Protocol):
    def __call__(self, context: ManagedAgentContext, /) -> Awaitable[AgentIdentityFailure | None]: ...


class AgentLoader(Protocol):
    def __call__(self, agent_id: str, /) -> Awaitable[AgentResponse | AgentIdentityFailure | None]: ...


@dataclass(frozen=True, slots=True)
class IdJagAgents:
    issuers: Mapping[str, OktaIssuer]
    lookup: BindingLookup
    grants: AgentGrants
    record: RecordAuthentication


async def id_jag_agent(
    claims: Mapping[str, object], client_id: str, user_id: str, team_id: str | None, agents: IdJagAgents
) -> AgentStamp | AgentIdentityFailure | None:
    issuer: Final = agents.issuers.get(str(claims.get("iss")))
    if issuer is None:
        return None
    agent: Final = await _delegated_agent(agents.lookup, issuer.org, client_id)
    if agent is None or isinstance(agent, AgentIdentityFailure):
        return agent
    binding: Final = _bound_identity(agent)
    if binding is None:
        return AgentIdentityFailure(message=f"agent {agent.agent_id} is disabled")
    if agent.agent_id not in await agents.grants(user_id, team_id):
        return AgentIdentityFailure(message=f"user {user_id} has no grant for agent {agent.agent_id}")
    recorded: Final = await agents.record(
        ManagedAgentContext(
            agent_id=agent.agent_id, binding_revision=binding.revision, mode="delegated", user_id=user_id
        )
    )
    if recorded is not None:
        return recorded
    return AgentStamp(agent_id=agent.agent_id, binding_revision=binding.revision)


async def stamped_agent_context(
    load: AgentLoader, agent_id: str, revision: str, user_id: str | None
) -> ManagedAgentContext | AgentIdentityFailure:
    agent: Final = await load(agent_id)
    if isinstance(agent, AgentIdentityFailure):
        return agent
    if agent is None:
        return AgentIdentityFailure(message="Agent no longer exists")
    binding: Final = _bound_identity(agent)
    if binding is None:
        return AgentIdentityFailure(message="Agent execution is disabled")
    if binding.revision != revision:
        return AgentIdentityFailure(message="Agent binding changed; redeem a new assertion")
    if agent.execution_mode == "autonomous":
        return AgentIdentityFailure(message="Agent is not enabled for this execution mode")
    return ManagedAgentContext(agent_id=agent.agent_id, binding_revision=revision, mode="delegated", user_id=user_id)
