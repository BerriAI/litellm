"""Project a team row (plus the caller's membership in it) onto the ``team_*`` fields of ``UserAPIKeyAuth``.

The virtual-key path gets these fields for free from the combined-view SQL join. Every other auth path
starts from a ``LiteLLM_TeamTable`` object instead and has to copy them over by hand, which is how JWT
callers kept losing grants (aliases, permissions, limits) one field at a time. Build the badge through
``team_grants`` and the two paths cannot drift.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final

from pydantic import BaseModel, BeforeValidator, ConfigDict, TypeAdapter, ValidationError
from pydantic.main import IncEx
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    Member,
)

_MODEL_ALIASES_ADAPTER: Final = TypeAdapter(dict[str, str])
_JSON_COLUMNS: Final[Mapping[str, IncEx | bool]] = MappingProxyType(
    {"metadata": True, "litellm_model_table": MappingProxyType({"model_aliases": True})}
)


def _decode_model_aliases(value: object) -> object:
    """``LiteLLM_ModelTable.model_aliases`` is typed ``str | dict``; writers hand Prisma ``json.dumps(...)``, so take both."""
    if not isinstance(value, str):
        return value
    try:
        return _MODEL_ALIASES_ADAPTER.validate_json(value)
    except ValidationError:
        return None


class TeamModelAliasTable(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_aliases: Annotated[Mapping[str, str] | None, BeforeValidator(_decode_model_aliases)] = None


class _TeamJsonColumns(BaseModel):
    """The two loosely typed columns on ``LiteLLM_TeamTable``, re-read with the shape the badge needs."""

    metadata: Mapping[str, object] | None = None
    litellm_model_table: TeamModelAliasTable | None = None


class TeamGrants(TypedDict, total=False):
    """Keyword arguments for ``UserAPIKeyAuth``. Empty when the caller has no team, so the model's own defaults apply."""

    team_alias: ReadOnly[str | None]
    team_tpm_limit: ReadOnly[int | None]
    team_rpm_limit: ReadOnly[int | None]
    team_max_budget: ReadOnly[float | None]
    team_soft_budget: ReadOnly[float | None]
    team_spend: ReadOnly[float | None]
    team_models: ReadOnly[Sequence[str]]
    team_blocked: ReadOnly[bool]
    team_metadata: ReadOnly[Mapping[str, object] | None]
    team_model_aliases: ReadOnly[Mapping[str, str] | None]
    team_object_permission_id: ReadOnly[str | None]
    team_object_permission: ReadOnly[LiteLLM_ObjectPermissionTable | None]
    team_member: ReadOnly[Member | None]
    team_member_spend: ReadOnly[float | None]
    team_member_tpm_limit: ReadOnly[int | None]
    team_member_rpm_limit: ReadOnly[int | None]


def _json_columns(team_object: LiteLLM_TeamTable) -> _TeamJsonColumns:
    try:
        return _TeamJsonColumns.model_validate(team_object.model_dump(include=_JSON_COLUMNS))
    except ValidationError:
        return _TeamJsonColumns()


def team_model_aliases(team_object: LiteLLM_TeamTable | None) -> Mapping[str, str] | None:
    if team_object is None:
        return None
    alias_table: Final = _json_columns(team_object).litellm_model_table
    return alias_table.model_aliases if alias_table is not None else None


def team_grants(
    team_object: LiteLLM_TeamTable | None,
    team_membership: LiteLLM_TeamMembership | None,
    user_id: str | None,
) -> TeamGrants:
    if team_object is None:
        return TeamGrants()
    json_columns: Final = _json_columns(team_object)
    return TeamGrants(
        team_alias=team_object.team_alias,
        team_tpm_limit=team_object.tpm_limit,
        team_rpm_limit=team_object.rpm_limit,
        team_max_budget=team_object.max_budget,
        team_soft_budget=team_object.soft_budget,
        team_spend=team_object.spend,
        team_models=tuple(team_object.models),
        team_blocked=team_object.blocked,
        team_metadata=json_columns.metadata,
        team_model_aliases=(
            json_columns.litellm_model_table.model_aliases if json_columns.litellm_model_table is not None else None
        ),
        team_object_permission_id=team_object.object_permission_id,
        team_object_permission=team_object.object_permission,
        team_member=next(
            (m for m in team_object.members_with_roles if user_id is not None and m.user_id == user_id),
            None,
        ),
        team_member_spend=team_membership.spend if team_membership is not None else None,
        team_member_tpm_limit=(
            team_membership.safe_get_team_member_tpm_limit() if team_membership is not None else None
        ),
        team_member_rpm_limit=(
            team_membership.safe_get_team_member_rpm_limit() if team_membership is not None else None
        ),
    )
