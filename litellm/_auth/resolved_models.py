from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.member_models import MemberModelFacts
from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts
from litellm._auth.personal_model_request import PersonalModelRequestFacts
from litellm._auth.team_override import TeamOverrideFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class ResolvedTeamRead:
    found: bool = False
    failed: bool = False


@dataclass(frozen=True, slots=True)
class ResolvedModelFacts:
    model: str = ""
    config_nonempty: bool = False
    key_models: tuple[str, ...] = ()
    team_id: str | None = None
    user_id: str | None = None
    project_present: bool = False
    customer_present: bool = False
    database_present: bool = False
    personal_enabled: bool = False
    master: bool = False
    role: str | None = None
    key: GroupedModelScope | None = None
    team_read: ResolvedTeamRead | None = None
    embedded_ready: bool = False
    team: TeamOverrideFacts | None = None
    member: MemberModelFacts | None = None
    owner_loaded: bool = False
    owner_present: bool = False
    owner_failed: bool = False
    personal: PersonalModelRequestFacts | None = None
    project_loaded: bool = False
    project_grants: tuple[str, ...] = ()
    project: ModelAccessFacts | None = None
    customer_loaded: bool = False
    customer_grants: tuple[str, ...] = ()
    customer: ModelAccessFacts | None = None


@dataclass(frozen=True, slots=True)
class ResolvedModelDecision:
    kind: Literal[
        "Allow",
        "Deny",
        "ReadKey",
        "ReadKeyGroups",
        "ReadTeamObject",
        "ReadEmbeddedTeam",
        "ReadTeamScope",
        "ReadMember",
        "ReadOwner",
        "ReadPersonal",
        "ReadProjectObject",
        "ReadProjectScope",
        "ReadCustomerObject",
        "ReadCustomerScope",
    ]
    denial: ModelAccessDecision | None = None
    member_models: tuple[str, ...] | None = None


_DECISION: Final = TypeAdapter(ResolvedModelDecision)


@native_boundary
def resolved_models(facts: ResolvedModelFacts) -> ResolvedModelDecision:
    return _DECISION.validate_python(_native.request_auth_resolved_models(asdict(facts)))
