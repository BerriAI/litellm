from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class GroupedModelScope:
    facts: ModelAccessFacts
    group_ids: tuple[str, ...] = ()
    group_models: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class FallbackAuthorizationFacts:
    key: GroupedModelScope | None = None
    refuse_agent_keys: bool = False
    agent_id: str | None = None
    team: GroupedModelScope | None = None
    embedded_team: ModelAccessFacts | None = None
    assigned_models: tuple[str, ...] | None = None
    member: ModelAccessFacts | None = None
    project: ModelAccessFacts | None = None
    team_present: bool = False
    embedded_team_grants: tuple[str, ...] = ()
    member_grants: tuple[str, ...] = ()
    project_grants: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FallbackAuthorizationDecision:
    kind: Literal[
        "Allow",
        "Deny",
        "ReadKey",
        "ReadTeam",
        "ReadEmbeddedTeam",
        "ReadMember",
        "ReadProject",
        "ReadKeyGroups",
        "ReadTeamGroups",
        "ReadAssignedGroups",
    ]
    denial: ModelAccessDecision | None = None


_DECISION: Final = TypeAdapter(FallbackAuthorizationDecision)


@native_boundary
def fallback_authorization(facts: FallbackAuthorizationFacts) -> FallbackAuthorizationDecision:
    return _DECISION.validate_python(_native.request_auth_fallback_access(asdict(facts)))
