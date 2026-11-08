from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.personal_models import LoadedTeamModelFacts
from litellm._auth.personal_owner import PersonalOwnerFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class PersonalModelListingFacts:
    owner: PersonalOwnerFacts
    names: tuple[str, ...] | None = None
    models: tuple[LoadedTeamModelFacts, ...] = ()


@dataclass(frozen=True, slots=True)
class PersonalModelListingDecision:
    kind: Literal["ReadOwner", "ReadNames", "ReadModel", "Complete"]
    index: int = 0
    teams: tuple[int, ...] = ()
    selected: tuple[int, ...] = ()


_DECISION: Final = TypeAdapter(PersonalModelListingDecision)


@native_boundary
def personal_model_listing(facts: PersonalModelListingFacts) -> PersonalModelListingDecision:
    return _DECISION.validate_python(_native.request_auth_personal_model_listing(asdict(facts)))
