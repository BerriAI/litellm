from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class TeamOverrideFacts:
    team: GroupedModelScope | None = None
    assigned_models: tuple[str, ...] | None = None
    assigned_scope: ModelAccessFacts | None = None


@dataclass(frozen=True, slots=True)
class TeamOverrideDecision:
    kind: Literal["Allow", "Deny", "ReadTeam", "ReadTeamGroups", "ReadAssignedGroups", "ReadAssignedScope"]
    denial: ModelAccessDecision | None = None


_DECISION: Final = TypeAdapter(TeamOverrideDecision)


@native_boundary
def team_override(facts: TeamOverrideFacts) -> TeamOverrideDecision:
    return _DECISION.validate_python(_native.request_auth_team_override(asdict(facts)))
