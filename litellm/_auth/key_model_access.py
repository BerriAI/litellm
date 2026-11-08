from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import ModelAccessDecision
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class KeyAndFallbackModelAccessFacts:
    config_nonempty: bool
    key_models: tuple[str, ...]
    primary_loaded: bool = False
    primary: GroupedModelScope | None = None
    fallback_names: tuple[str, ...] | None = None
    fallback_scopes: tuple[GroupedModelScope, ...] = ()
    routing: tuple[Literal["Routed", "Failed"], ...] = ()


@dataclass(frozen=True, slots=True)
class KeyAndFallbackModelAccessDecision:
    kind: Literal[
        "Allow",
        "Deny",
        "ReadPrimary",
        "ReadPrimaryGroups",
        "ReadFallbackNames",
        "ReadFallbackScope",
        "ReadFallbackGroups",
        "RouteFallback",
        "RoutingFailure",
    ]
    index: int = 0
    denial: ModelAccessDecision | None = None


_DECISION: Final = TypeAdapter(KeyAndFallbackModelAccessDecision)


def key_and_fallback_model_access(facts: KeyAndFallbackModelAccessFacts) -> KeyAndFallbackModelAccessDecision:
    return _DECISION.validate_python(_native.request_auth_key_model_access(asdict(facts)))
