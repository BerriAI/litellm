from dataclasses import asdict, dataclass
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native


@dataclass(frozen=True, slots=True)
class GroupResources:
    assigned_team_ids: tuple[str, ...]
    assigned_key_ids: tuple[str, ...]
    resources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupLookup:
    kind: Literal["Found", "Missing", "Failed"]
    group: GroupResources | None = None


@dataclass(frozen=True, slots=True)
class ResourceGroupFacts:
    subject_present: bool
    database_present: bool
    cache_present: bool
    group_ids: tuple[str, ...]
    team_id: str | None
    key_id: str | None
    check_db_only: bool = False
    strict_errors: bool = False
    loaded: tuple[GroupLookup, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceGroupDecision:
    kind: Literal["Read", "Raise", "Complete"]
    index: int = 0
    resources: tuple[str, ...] = ()


_DECISION: Final = TypeAdapter(ResourceGroupDecision)


@native_boundary
def resource_group_decision(facts: ResourceGroupFacts) -> ResourceGroupDecision:
    return _DECISION.validate_python(_native.request_auth_model_groups(asdict(facts)))
