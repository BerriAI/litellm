from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.rust_core import encode_facts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class BoundIdentity:
    active: bool
    revision: str


@dataclass(frozen=True, slots=True)
class Agent:
    agent_id: str
    enabled: bool
    identity_managed: bool
    identity: BoundIdentity | None
    execution_mode: str


@dataclass(frozen=True, slots=True)
class Context:
    agent_id: str
    binding_revision: str | None
    mode: str
    user_id: str | None


@dataclass(frozen=True, slots=True)
class Actor:
    agent: Agent
    context: Context | None


@dataclass(frozen=True, slots=True)
class ActorLookup:
    agent_id: str | None
    has_database: bool
    registry_loaded: bool = False
    registry_managed: bool = False
    agent_loaded: bool = False
    agent: Agent | None = None
    retired: bool | None = None
    context: Context | None = None
    jwt_claims: bool = False
    delegation_verified: bool = False
    human_grants: tuple[str, ...] | None = None
    policy_assigned: bool = False


@dataclass(frozen=True, slots=True)
class UnionGrants:
    grants: tuple[tuple[str, ...] | None, ...]


@dataclass(frozen=True, slots=True)
class IntersectGrants:
    left: tuple[str, ...] | None
    right: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class DeclaredGrants:
    direct: tuple[str, ...]
    declared_groups: tuple[str, ...]
    unified_groups: tuple[str, ...]
    group_agents: tuple[str, ...]
    unified_agents: tuple[str, ...]


ManagedDenial: TypeAlias = Literal[
    "Disabled",
    "MissingContext",
    "ChangedIdentity",
    "ExecutionMode",
    "MissingHuman",
    "DatabaseRequired",
    "Removed",
    "VerifiedIdentityRequired",
    "HumanNotPermitted",
]


@dataclass(frozen=True, slots=True)
class ManagedDecision:
    action: Literal[
        "Allow", "Deny", "ReadRegistry", "ReadAgent", "ReadRetired", "AssignPolicy", "ReadHumanGrants", "Grants"
    ]
    denial: ManagedDenial | None
    grants: tuple[str, ...] | None


_DECISION: Final = TypeAdapter(ManagedDecision)


def decide(facts: Actor | ActorLookup | UnionGrants | IntersectGrants | DeclaredGrants) -> ManagedDecision:
    return _DECISION.validate_python(_native.request_auth_managed(encode_facts(facts)))
