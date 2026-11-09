from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class ClaudeMembershipFacts:
    name: str
    router_present: bool
    alias_count: int
    global_alias_exists: bool | None = None
    alias_entries: tuple[bool | None, ...] = ()
    registry_loaded: bool = False
    registry_alias_exists: bool | None = None
    deployment: bool | None = None
    names: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ClaudeMembershipDecision:
    kind: Literal["Complete", "ReadGlobal", "ReadAlias", "ReadRegistry", "ReadAliases", "ReadDeployment", "ReadNames"]
    present: bool = False
    index: int = 0


@dataclass(frozen=True, slots=True)
class ClaudeNormalizationFacts:
    route: str
    request_present: bool
    router_present: bool | None = None
    normalized: bool | None = None
    model_loaded: bool = False
    requested: str | None = None
    body_loaded: bool = False
    body_model: str | None = None
    settings_loaded: bool = False
    membership: tuple[ClaudeMembershipFacts, ...] = ()
    marked: bool = False


@dataclass(frozen=True, slots=True)
class ClaudeNormalizationDecision:
    kind: Literal[
        "Complete",
        "ReadRouter",
        "ReadNormalized",
        "ReadModel",
        "ReadBodyModel",
        "ReadSettings",
        "ReadMembership",
        "MarkNormalized",
        "Rewrite",
    ]
    name: str | None = None
    index: int = 0
    target: str | None = None


_MEMBERSHIP: Final = TypeAdapter(ClaudeMembershipDecision)
_NORMALIZATION: Final = TypeAdapter(ClaudeNormalizationDecision)


@native_boundary
def claude_membership(facts: ClaudeMembershipFacts) -> ClaudeMembershipDecision:
    return _MEMBERSHIP.validate_python(_native.request_auth_claude_membership(asdict(facts)))


@native_boundary
def claude_normalization(facts: ClaudeNormalizationFacts) -> ClaudeNormalizationDecision:
    return _NORMALIZATION.validate_python(_native.request_auth_claude_normalization(asdict(facts)))
