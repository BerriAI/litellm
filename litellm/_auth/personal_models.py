from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import ModelAccessFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class PersonalModelScope:
    user_present: bool
    team_id_present: bool
    team: GroupedModelScope | None = None
    member_grants: tuple[str, ...] | None = None
    member: ModelAccessFacts | None = None
    io_failed: bool = False


@dataclass(frozen=True, slots=True)
class LoadedTeamModelFacts:
    present: tuple[bool, ...]
    scopes: tuple[PersonalModelScope, ...] = ()


@dataclass(frozen=True, slots=True)
class PersonalModelDecision:
    kind: Literal["Complete", "ReadTeam", "ReadGroups", "ReadMembership", "ReadMember", "ReadScopes"]
    allowed: bool = False
    indices: tuple[int, ...] = ()


_DECISION: Final = TypeAdapter(PersonalModelDecision)


@native_boundary
def personal_model_scope(facts: PersonalModelScope) -> PersonalModelDecision:
    return _DECISION.validate_python(_native.request_auth_personal_models({"Scope": asdict(facts)}))


@native_boundary
def loaded_team_models(facts: LoadedTeamModelFacts) -> PersonalModelDecision:
    return _DECISION.validate_python(_native.request_auth_personal_models({"Loaded": asdict(facts)}))
