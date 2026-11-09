from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.owner_teams import OwnerTeamFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class PersonalOwnerFacts:
    user_present: bool
    database_present: bool
    cache_present: bool
    logging_present: bool
    owner_loaded: bool = False
    owner_present: bool = False
    owner_failed: bool = False
    teams: OwnerTeamFacts | None = None


@dataclass(frozen=True, slots=True)
class PersonalOwnerDecision:
    kind: Literal["ReadOwner", "ReadTeams", "Complete"]
    selected: tuple[int, ...] = ()
    withheld: tuple[int, ...] = ()
    unreadable: bool = False


_DECISION: Final = TypeAdapter(PersonalOwnerDecision)


@native_boundary
def personal_owner(facts: PersonalOwnerFacts) -> PersonalOwnerDecision:
    return _DECISION.validate_python(_native.request_auth_personal_owner(asdict(facts)))
