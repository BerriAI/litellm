from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class MemberModelFacts:
    user_present: bool
    team_present: bool
    membership_loaded: bool = False
    grants: tuple[str, ...] | None = None
    scope: ModelAccessFacts | None = None


@dataclass(frozen=True, slots=True)
class MemberModelDecision:
    kind: Literal["Allow", "Deny", "ReadMembership", "ReadGrants", "ReadScope"]
    denial: ModelAccessDecision | None = None
    member_models: tuple[str, ...] | None = None


_DECISION: Final = TypeAdapter(MemberModelDecision)


@native_boundary
def member_models(facts: MemberModelFacts) -> MemberModelDecision:
    return _DECISION.validate_python(_native.request_auth_member_models(asdict(facts)))
