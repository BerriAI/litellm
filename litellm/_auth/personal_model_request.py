from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.owner_teams import OwnerTeamFacts
from litellm._auth.personal_models import LoadedTeamModelFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class PersonalModelRequestFacts:
    requested: tuple[str, ...]
    owner_present: bool
    teams: OwnerTeamFacts | None = None
    models: tuple[LoadedTeamModelFacts, ...] = ()


@dataclass(frozen=True, slots=True)
class PersonalModelRequestDecision:
    kind: Literal["Allow", "Deny", "ReadTeams", "ReadModel"]
    model: str = ""
    index: int = 0
    teams: tuple[int, ...] = ()


_DECISION: Final = TypeAdapter(PersonalModelRequestDecision)


@native_boundary
def personal_model_request(facts: PersonalModelRequestFacts) -> PersonalModelRequestDecision:
    return _DECISION.validate_python(_native.request_auth_personal_model_request(asdict(facts)))
