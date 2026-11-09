from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts, ModelReading, ModelReadKind
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class ModelProjectionDecision:
    kind: Literal["Complete", "Read"]
    read: ModelReadKind | None = None
    model: str | None = None
    decision: ModelAccessDecision | None = None


_DECISION: Final = TypeAdapter(ModelProjectionDecision)


@native_boundary
def model_projection(facts: ModelAccessFacts, reads: tuple[ModelReading, ...]) -> ModelProjectionDecision:
    return _DECISION.validate_python(
        _native.request_auth_model_projection(asdict(facts), tuple(asdict(read) for read in reads))
    )
