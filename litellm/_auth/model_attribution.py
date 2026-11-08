from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class ModelGroupAttributionFacts:
    model_present: bool = False
    token_present: bool = False
    router_present: bool = False
    database_present: bool = False
    team_present: bool = False
    user_present: bool = False
    membership_loaded: bool = False
    token_org: str | None = None
    team_org: str | None = None
    registry_loaded: bool = False
    registry: tuple[str, ...] | None = None
    covering: tuple[str, ...] | None = None
    grants: tuple[tuple[str, ...], ...] = ()
    org_failed: bool = False
    io_failed: bool = False


@dataclass(frozen=True, slots=True)
class ModelGroupAttributionDecision:
    kind: Literal[
        "Complete", "ReadRegistry", "ReadCovering", "ReadKey", "ReadTeam", "ReadMember", "ReadProject", "ReadOrg"
    ]
    groups: tuple[str, ...] = ()
    stamp: bool = False
    lookup: bool = False
    include: bool = False
    id: str | None = None


_DECISION: Final = TypeAdapter(ModelGroupAttributionDecision)


@native_boundary
def model_group_attribution(facts: ModelGroupAttributionFacts) -> ModelGroupAttributionDecision:
    return _DECISION.validate_python(_native.request_auth_model_attribution(asdict(facts)))
