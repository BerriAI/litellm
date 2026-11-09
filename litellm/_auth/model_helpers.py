from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.model_access import ModelProvider
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class ModelHelperFacts:
    kind: Literal["Group", "Helper"]
    model: str
    grants: tuple[str, ...] | None
    team_id: str | None = None
    team_aliases: tuple[str, ...] = ()
    router_present: bool = False
    groups: tuple[str, ...] | None = None
    names: tuple[str, ...] | None = None
    provider_loaded: bool = False
    provider: ModelProvider | None = None
    provider_reads: tuple[ModelProvider | None, ...] | None = None


@dataclass(frozen=True, slots=True)
class ModelHelperDecision:
    kind: Literal["Complete", "ReadGroups", "ReadNames", "ReadProvider"]
    allowed: bool = False


_DECISION: Final = TypeAdapter(ModelHelperDecision)


@native_boundary
def model_helper(facts: ModelHelperFacts) -> ModelHelperDecision:
    return _DECISION.validate_python(_native.request_auth_model_helpers(asdict(facts)))
