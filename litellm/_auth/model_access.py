from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Final, Literal, TypeAlias

from pydantic import TypeAdapter

from litellm._auth.failure import native_boundary
from litellm.rust_bridge import _native

ModelObject: TypeAlias = Literal["user", "customer", "team", "key", "org", "project", "agent"]
ModelReadKind: TypeAlias = Literal["Compaction", "Aliases", "AliasTarget", "Deployment", "Groups", "Names", "Provider"]


@dataclass(frozen=True, slots=True)
class ModelProvider:
    stripped_model: str
    provider: str


@dataclass(frozen=True, slots=True)
class ModelRecord:
    access_groups: tuple[str, ...] = ()
    provider: ModelProvider | None = None
    deployment_team: str | None = None
    deployment_public_name: str | None = None


@dataclass(frozen=True, slots=True)
class ModelReading:
    kind: ModelReadKind
    model: str
    values: tuple[str, ...] = ()
    value: str | None = None
    team: str | None = None
    provider: ModelProvider | None = None


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    grants: tuple[str, ...]
    reads: tuple[ModelReading, ...]
    aliases: ModelAliases | None = None


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    present: bool = False
    names: tuple[str, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict[str, str])
    records: Mapping[str, ModelRecord] = field(default_factory=dict[str, ModelRecord])
    compaction_parents: Mapping[str, str] = field(default_factory=dict[str, str])
    helper_present: bool | None = None
    evaluations: tuple[ModelEvaluation, ...] | None = None


@dataclass(frozen=True, slots=True)
class ModelAliases:
    team: Mapping[str, str] = field(default_factory=dict[str, str])
    key: Mapping[str, str] = field(default_factory=dict[str, str])
    global_: Mapping[str, str] = field(default_factory=dict[str, str])
    team_order: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class ModelAccessFacts:
    requested: tuple[str, ...]
    grants: tuple[str, ...] = ()
    team_id: str | None = None
    aliases: ModelAliases = field(default_factory=ModelAliases)
    registry: ModelRegistry = field(default_factory=ModelRegistry)
    object_type: ModelObject = "user"
    fallback_depth: int = 0
    max_depth: int = 5
    is_list: bool = False


@dataclass(frozen=True, slots=True)
class ModelAccessDecision:
    kind: Literal[
        "Allow",
        "Deny",
        "Depth",
        "NoDefaultModels",
        "NoAgentModels",
        "EmptyAgentCeiling",
        "PersonalTeam",
        "ReadGroups",
        "ReadScope",
        "ReadGroupScope",
    ]
    model: str | None = None
    grants: tuple[str, ...] = ()
    object_type: ModelObject = "user"


@dataclass(frozen=True, slots=True)
class AssignedModelGroup:
    assigned_team_ids: tuple[str, ...]
    assigned_key_ids: tuple[str, ...]
    models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PersonalModelTeam:
    blocked: bool
    available: bool
    team: ModelAccessFacts
    member: ModelAccessFacts | None = None


_DECISION: Final = TypeAdapter(ModelAccessDecision)
_MODELS: Final = TypeAdapter(tuple[str, ...])
_BOOL: Final = TypeAdapter(bool)


@native_boundary
def model_access(facts: ModelAccessFacts, *, kind: Literal["Object", "Customer"] = "Object") -> ModelAccessDecision:
    return _DECISION.validate_python(_native.request_auth_model_access({kind: asdict(facts)}))


@native_boundary
def grouped_model_access(
    facts: ModelAccessFacts,
    group_ids: tuple[str, ...],
    group_models: tuple[str, ...] | None = None,
    *,
    merge: bool = False,
) -> ModelAccessDecision:
    return _DECISION.validate_python(
        _native.request_auth_model_access(
            {
                "Grouped": {
                    "facts": asdict(facts),
                    "group_ids": group_ids,
                    "group_models": group_models,
                    "merge": merge,
                }
            }
        )
    )


@native_boundary
def user_model_access(facts: ModelAccessFacts, *, present: bool) -> ModelAccessDecision:
    return _DECISION.validate_python(
        _native.request_auth_model_access({"User": {"facts": asdict(facts), "present": present}})
    )


@native_boundary
def user_model_precheck(*, present: bool, grants: tuple[str, ...]) -> ModelAccessDecision:
    return _DECISION.validate_python(
        _native.request_auth_model_access({"UserPrecheck": {"present": present, "grants": grants}})
    )


@native_boundary
def agent_model_access(
    facts: ModelAccessFacts, managed_grants: tuple[str, ...] | None, ceilings: tuple[tuple[str, ...], ...]
) -> ModelAccessDecision:
    return _DECISION.validate_python(
        _native.request_auth_model_access(
            {
                "Agent": {
                    "facts": asdict(facts),
                    "managed_grants": managed_grants,
                    "ceilings": ceilings,
                }
            }
        )
    )


@native_boundary
def personal_model_access(requested: tuple[str, ...], teams: tuple[PersonalModelTeam, ...]) -> ModelAccessDecision:
    return _DECISION.validate_python(
        _native.request_auth_model_access(
            {
                "Personal": {
                    "requested": requested,
                    "teams": tuple(asdict(team) for team in teams),
                }
            }
        )
    )


@native_boundary
def model_candidates(facts: ModelAccessFacts) -> tuple[str, ...]:
    return _MODELS.validate_python(_native.request_auth_model_access({"Candidates": asdict(facts)}))


@native_boundary
def model_access_helper(model: str, facts: ModelAccessFacts) -> bool:
    return _BOOL.validate_python(
        _native.request_auth_model_access({"Helper": {"model": model, "facts": asdict(facts)}})
    )


@native_boundary
def key_model_grants(grants: tuple[str, ...], team_id: str | None, team_grants: tuple[str, ...]) -> tuple[str, ...]:
    return _MODELS.validate_python(
        _native.request_auth_model_access(
            {
                "KeyGrants": {
                    "grants": grants,
                    "team_id": team_id,
                    "team_grants": team_grants,
                }
            }
        )
    )


@native_boundary
def assigned_model_grants(
    groups: tuple[AssignedModelGroup, ...], team_id: str | None, key_id: str | None
) -> tuple[str, ...]:
    return _MODELS.validate_python(
        _native.request_auth_model_access(
            {
                "AssignedGroups": {
                    "groups": tuple(asdict(group) for group in groups),
                    "team_id": team_id,
                    "key_id": key_id,
                }
            }
        )
    )


@native_boundary
def personal_model_access_applies(
    *,
    enabled: bool,
    token_present: bool,
    team_id: str | None,
    user_id: str | None,
    master: bool,
    role: str | None,
    api_key_present: bool,
    inheritance: bool = False,
) -> bool:
    return _BOOL.validate_python(
        _native.request_auth_model_access(
            {
                "PersonalApplies": {
                    "enabled": enabled,
                    "token_present": token_present,
                    "team_id": team_id,
                    "user_id": user_id,
                    "master": master,
                    "role": role,
                    "api_key_present": api_key_present,
                    "inheritance": inheritance,
                }
            }
        )
    )
