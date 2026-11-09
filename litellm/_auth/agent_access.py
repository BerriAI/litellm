from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.managed import Agent
from litellm._auth.rust_core import encode_facts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class AgentAccessFacts:
    registry_managed: bool
    database_present: bool
    target_loaded: bool = False
    target: Agent | None = None
    subject_present: bool = False
    key_hash_present: bool = False
    subject_managed: bool = False
    session: bool = False
    custom: bool = False
    authority_loaded: bool = False
    explicit_grants: tuple[str, ...] | None = None
    legacy_loaded: bool = False
    legacy_grants: tuple[str, ...] | None = None
    target_ids: tuple[str, ...] | None = None


AgentAccessDecision: TypeAlias = Literal[
    "ReadTarget", "ReadAuthority", "ReadExplicitGrants", "ReadLegacyGrants", "ReadTargetIds", "Allow", "Deny"
]
_DECISION: Final[TypeAdapter[AgentAccessDecision]] = TypeAdapter(AgentAccessDecision)


@native_boundary
def decide(facts: AgentAccessFacts) -> AgentAccessDecision:
    return _DECISION.validate_python(_native.request_auth_agent_access(encode_facts(facts)))


@dataclass(frozen=True, slots=True)
class InvocationFacts:
    registered: Agent | None
    has_store: bool
    target_loaded: bool = False
    target: Agent | None = None
    actor_managed: bool = False
    actor_id_present: bool = False
    permissions: AgentAccessFacts | None = None
    assigned: bool = False
    fee_loaded: bool = False
    fee: float | None = None


@dataclass(frozen=True, slots=True)
class InvocationPlan:
    action: Literal["Complete", "ReadTarget", "ReadPermissions", "Assign", "ReadFee", "Deny"]
    denial: Literal["DatabaseRequired", "Removed", "CallerNotPermitted", "InvalidFee"] | None
    use_target: bool
    assign_billing: bool


_INVOCATION: Final = TypeAdapter(InvocationPlan)


@native_boundary
def invocation(facts: InvocationFacts) -> InvocationPlan:
    return _INVOCATION.validate_python(_native.request_auth_agent_invocation(encode_facts(facts)))
