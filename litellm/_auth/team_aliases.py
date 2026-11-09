from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class TeamAliasFacts:
    requested: tuple[str, ...]
    aliases: Mapping[str, str]
    team_id: str | None
    router_present: bool
    names: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True, slots=True)
class TeamAliasDecision:
    kind: Literal["ReadNames", "Complete"]
    count: int = 0
    models: tuple[str, ...] = ()


_DECISION: Final = TypeAdapter(TeamAliasDecision)


@native_boundary
def team_aliases(facts: TeamAliasFacts) -> TeamAliasDecision:
    return _DECISION.validate_python(
        _native.request_auth_team_aliases(asdict(replace(facts, aliases=dict(facts.aliases))))
    )
