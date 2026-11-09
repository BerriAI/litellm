from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.member_models import MemberModelFacts
from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class ManagedModelFacts:
    purpose: Literal["All", "Ceilings", "Caller"]
    model_present: bool
    token_present: bool
    managed_loaded: bool = False
    managed_present: bool = False
    managed_models_valid: bool = False
    managed_models_count: int = 0
    managed: ModelAccessFacts | None = None
    ceilings: tuple[tuple[str, ...], ...] | None = None
    aliases_loaded: bool = False
    scopes: tuple[ModelAccessFacts, ...] = ()
    caller_loaded: bool = False
    caller_present: bool = False
    team_loaded: bool = False
    team_present: bool = False
    team: GroupedModelScope | None = None
    member: MemberModelFacts | None = None
    user_loaded: bool = False
    user_present: bool = False
    user_grants: tuple[str, ...] = ()
    user: ModelAccessFacts | None = None


@dataclass(frozen=True, slots=True)
class ManagedModelDecision:
    kind: Literal[
        "Allow",
        "Deny",
        "ReadManaged",
        "ReadManagedScope",
        "ReadCeilings",
        "ReadAliases",
        "ReadCeilingScope",
        "ReadCaller",
        "ReadTeam",
        "ReadTeamScope",
        "ReadTeamGroups",
        "ReadMember",
        "ReadUser",
        "ReadUserScope",
    ]
    denial: ModelAccessDecision | None = None
    member_models: tuple[str, ...] | None = None
    index: int = 0


@dataclass(frozen=True, slots=True)
class AgentCeilingSourceFacts:
    agent_present: bool
    managed_loaded: bool = False
    managed_present: bool = False
    loaded: bool = False
    count: int = 0


@dataclass(frozen=True, slots=True)
class AgentCeilingSourceDecision:
    kind: Literal["ReadManaged", "ReadManagedCeilings", "ReadUnmanagedCeiling", "Complete"]
    indices: tuple[int, ...] = ()


_DECISION: Final = TypeAdapter(ManagedModelDecision)
_CEILINGS: Final = TypeAdapter(AgentCeilingSourceDecision)
_GRANTS: Final = TypeAdapter(frozenset[str])


@native_boundary
def managed_models(facts: ManagedModelFacts) -> ManagedModelDecision:
    return _DECISION.validate_python(_native.request_auth_managed_models(asdict(facts)))


@native_boundary
def agent_ceiling_source(facts: AgentCeilingSourceFacts) -> AgentCeilingSourceDecision:
    return _CEILINGS.validate_python(_native.request_auth_agent_ceiling_source(asdict(facts)))


@native_boundary
def managed_model_grants(
    models: tuple[str | None, ...] | None, ceilings: tuple[tuple[str, ...], ...]
) -> frozenset[str]:
    return _GRANTS.validate_python(_native.request_auth_managed_model_grants({"models": models, "ceilings": ceilings}))
