from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class AgentCeilingGroup:
    status: Literal["Found", "Missing", "Unavailable"]
    models: tuple[str, ...] = ()
    mcp_server_ids: tuple[str, ...] = ()
    agent_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentCeilingFacts:
    ids: tuple[str, ...] | None = None
    groups: tuple[AgentCeilingGroup, ...] | None = None
    check_db_only: bool = False


@dataclass(frozen=True, slots=True)
class AgentCeilingDecision:
    kind: Literal["ReadIds", "ReadGroups", "Complete"]
    authoritative: bool = False
    present: bool = False
    ids: tuple[str, ...] = ()
    models: frozenset[str] = frozenset()
    mcp_server_ids: frozenset[str] = frozenset()
    agent_ids: frozenset[str] = frozenset()
    indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentGroupLoadFacts:
    database_present: bool
    check_db_only: bool = False
    status: Literal["Success", "NotFound", "HttpError"] | None = None


@dataclass(frozen=True, slots=True)
class AgentGroupLoadDecision:
    kind: Literal["Read", "Return", "DatabaseUnavailable", "HttpUnavailable", "Raise"]


_DECISION: Final = TypeAdapter(AgentCeilingDecision)
_LOAD: Final = TypeAdapter(AgentGroupLoadDecision)


@native_boundary
def agent_ceiling(facts: AgentCeilingFacts) -> AgentCeilingDecision:
    return _DECISION.validate_python(_native.request_auth_agent_ceiling(asdict(facts)))


@native_boundary
def agent_group_load(facts: AgentGroupLoadFacts) -> AgentGroupLoadDecision:
    return _LOAD.validate_python(_native.request_auth_agent_group_load(asdict(facts)))
