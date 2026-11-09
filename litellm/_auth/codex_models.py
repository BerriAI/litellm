from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm._auth.model_helpers import ModelHelperFacts
from litellm._auth.rust_core import encode_facts
from litellm.rust_bridge import _native

CodexModelKind: TypeAlias = Literal["Routes", "NativeRoutes", "Chatgpt", "Grants"]
CandidateField: TypeAlias = Literal["Provider", "Model", "Name", "PublicName"]


@dataclass(frozen=True, slots=True)
class CodexCandidate:
    provider_loaded: bool = False
    provider: str | None = None
    model: str | None = None
    name: str | None = None
    public_loaded: bool = False
    public_name: str | None = None


@dataclass(frozen=True, slots=True)
class CodexGrants:
    key: tuple[str, ...]
    team: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CodexModelFacts:
    kind: CodexModelKind
    model: str | None
    router_present: bool
    team_id: str | None = None
    bundled: tuple[str, ...] | None = None
    priced: bool | None = None
    candidates: tuple[CodexCandidate, ...] | None = None
    alias_loaded: bool = False
    alias: str | None = None
    grants: CodexGrants | None = None
    prefixed_grants: CodexGrants | None = None
    bare: tuple[ModelHelperFacts, ...] = ()
    prefixed: tuple[ModelHelperFacts, ...] = ()
    route: str | None = None


@dataclass(frozen=True, slots=True)
class CodexModelDecision:
    kind: Literal[
        "Complete",
        "ReadBundled",
        "ReadCost",
        "ReadCandidates",
        "ReadAlias",
        "ReadCandidate",
        "ReadGrants",
        "ReadHelper",
    ]
    allowed: bool = False
    target: str | None = None
    model: str | None = None
    index: int = 0
    field: CandidateField | None = None
    prefixed: bool = False


_DECISION: Final = TypeAdapter(CodexModelDecision)


@native_boundary
def codex_model(facts: CodexModelFacts) -> CodexModelDecision:
    return _DECISION.validate_python(_native.request_auth_codex_model(encode_facts(facts)))
