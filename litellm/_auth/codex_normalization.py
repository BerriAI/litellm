from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.codex_models import CodexModelFacts
from litellm._auth.codex_turn import CodexTurnFacts
from litellm._auth.failure import native_boundary
from litellm._auth.routes import RouteContext
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class CodexNormalizationFacts:
    kind: Literal["Alias", "Prefix"]
    route: str
    request_present: bool
    router_present: bool | None = None
    context: RouteContext | None = None
    eu_jev: bool | None = None
    resolved: bool | None = None
    marked: bool = False
    model_loaded: bool = False
    model: str | None = None
    explicit: bool | None = None
    provider: bool | None = None
    routing: CodexModelFacts | None = None
    body_present: bool | None = None
    settings_loaded: bool = False
    target: str | None = None
    provider_after: bool | None = None
    turn: CodexTurnFacts | None = None
    prefix: CodexModelFacts | None = None


@dataclass(frozen=True, slots=True)
class CodexNormalizationDecision:
    kind: Literal[
        "Complete",
        "ReadRouter",
        "ReadContext",
        "ReadEuJev",
        "ReadResolved",
        "MarkResolved",
        "ReadModel",
        "ReadExplicit",
        "ReadProvider",
        "ReadRouting",
        "ReadBody",
        "ReadSettings",
        "ReadProviderAfter",
        "ReadTurn",
        "ReadPrefix",
        "Rewrite",
    ]
    target: str | None = None


_DECISION: Final = TypeAdapter(CodexNormalizationDecision)


@native_boundary
def codex_normalization(facts: CodexNormalizationFacts) -> CodexNormalizationDecision:
    return _DECISION.validate_python(_native.request_auth_codex_normalization(asdict(facts)))
