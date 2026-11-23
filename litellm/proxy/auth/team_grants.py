"""Project a team row (plus the caller's membership in it) onto the ``team_*`` fields of ``UserAPIKeyAuth``.

A fresh virtual-key read gets these fields from the combined-view SQL join. Everything else starts from a
``LiteLLM_TeamTable`` object: the JWT path, and a virtual key whose cached copy is older than its team.
Copying them over by hand is how callers kept losing grants (aliases, permissions, limits) one field at a
time. Build the badge through ``team_grants`` or ``with_team_row`` and the paths cannot drift.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final

from pydantic import BeforeValidator, ConfigDict, TypeAdapter, ValidationError
from pydantic.main import IncEx
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    Member,
    UserAPIKeyAuth,
)
from litellm.types.llms.base import LiteLLMBaseModel

_MODEL_ALIASES_ADAPTER: Final = TypeAdapter(dict[str, str])
_METADATA_COLUMN: Final[Mapping[str, IncEx | bool]] = MappingProxyType({"metadata": True})
_ALIAS_COLUMN: Final[Mapping[str, IncEx | bool]] = MappingProxyType(
    {"litellm_model_table": MappingProxyType({"model_aliases": True})}
)


def _decode_model_aliases(value: object) -> object:
    """``LiteLLM_ModelTable.model_aliases`` is typed ``str | dict``; writers hand Prisma ``json.dumps(...)``, so take both."""
    if not isinstance(value, str):
        return value
    try:
        return _MODEL_ALIASES_ADAPTER.validate_json(value)
    except ValidationError:
        return None


class TeamModelAliasTable(LiteLLMBaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_aliases: Annotated[Mapping[str, str] | None, BeforeValidator(_decode_model_aliases)] = None


class _TeamJsonColumns(LiteLLMBaseModel):
    """The two loosely typed columns on ``LiteLLM_TeamTable``, re-read with the shape the badge needs."""

    metadata: Mapping[str, object] | None = None
    litellm_model_table: TeamModelAliasTable | None = None


class TeamRowGrants(TypedDict, total=False):
    """What the team's own row grants: every field but the caller's membership spend and limits."""

    team_alias: ReadOnly[str | None]
    team_tpm_limit: ReadOnly[int | None]
    team_rpm_limit: ReadOnly[int | None]
    team_tpd_limit: ReadOnly[int | None]
    team_max_budget: ReadOnly[float | None]
    team_soft_budget: ReadOnly[float | None]
    team_model_max_budget: ReadOnly[dict[str, object] | None]
    team_spend: ReadOnly[float | None]
    team_models: ReadOnly[Sequence[str]]
    team_blocked: ReadOnly[bool]
    team_metadata: ReadOnly[Mapping[str, object] | None]
    team_model_aliases: ReadOnly[Mapping[str, str] | None]
    team_object_permission_id: ReadOnly[str | None]
    team_object_permission: ReadOnly[LiteLLM_ObjectPermissionTable | None]
    team_member: ReadOnly[Member | None]


class TeamGrants(TeamRowGrants, total=False):
    """Keyword arguments for ``UserAPIKeyAuth``. Empty when the caller has no team, so the model's own defaults apply."""

    team_member_spend: ReadOnly[float | None]
    team_member_tpm_limit: ReadOnly[int | None]
    team_member_rpm_limit: ReadOnly[int | None]


_TEAM_GRANT_AUTH_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "team_alias",
        "team_tpm_limit",
        "team_rpm_limit",
        "team_tpd_limit",
        "team_max_budget",
        "team_soft_budget",
        "team_spend",
        "team_metadata",
        "team_member",
        "team_member_spend",
        "team_member_tpm_limit",
        "team_member_rpm_limit",
    }
)


def team_grant_auth_update(
    team_object: LiteLLM_TeamTable,
    team_membership: LiteLLM_TeamMembership | None,
    user_id: str | None,
    *,
    key_org_id: str | None,
) -> Mapping[str, object]:
    grant_values: Final = team_grants(team_object, team_membership, user_id)
    return MappingProxyType(
        {
            "team_id": team_object.team_id,
            "org_id": key_org_id or team_object.organization_id,
            **{field: value for field, value in grant_values.items() if field in _TEAM_GRANT_AUTH_FIELDS},
        }
    )


async def team_grant_billing_auth_update(
    auth: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable,
    team_membership: LiteLLM_TeamMembership | None,
) -> Mapping[str, object]:
    team_auth_update: Final = team_grant_auth_update(
        team_object,
        team_membership,
        auth.user_id,
        key_org_id=auth.org_id,
    )
    if auth.org_id is not None or team_object.organization_id is None:
        return team_auth_update

    organization_auth_fields: Final[Mapping[str, object]] = MappingProxyType(
        {
            "organization_alias": None,
            "organization_metadata": None,
            "organization_max_budget": None,
            "organization_tpm_limit": None,
            "organization_rpm_limit": None,
        }
    )
    from litellm.proxy.auth.auth_checks import get_org_object_for_request
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        return MappingProxyType({**team_auth_update, **organization_auth_fields})
    org_object: Final = await get_org_object_for_request(
        org_id=team_object.organization_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=auth.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    if org_object is None:
        return MappingProxyType({**team_auth_update, **organization_auth_fields})
    budget: Final = org_object.litellm_budget_table
    return MappingProxyType(
        {
            **team_auth_update,
            "organization_alias": org_object.organization_alias,
            "organization_metadata": org_object.metadata,
            "organization_max_budget": budget.max_budget if budget is not None else None,
            "organization_tpm_limit": budget.tpm_limit if budget is not None else None,
            "organization_rpm_limit": budget.rpm_limit if budget is not None else None,
        }
    )


def _json_column(team_object: LiteLLM_TeamTable, column: Mapping[str, IncEx | bool]) -> _TeamJsonColumns:
    """One column at a time, so an alias table that fails to parse never takes the metadata with it."""
    try:
        return _TeamJsonColumns.model_validate(team_object.model_dump(include=column))
    except ValidationError:
        return _TeamJsonColumns()


def team_model_aliases(team_object: LiteLLM_TeamTable | None) -> Mapping[str, str] | None:
    if team_object is None:
        return None
    alias_table: Final = _json_column(team_object, _ALIAS_COLUMN).litellm_model_table
    return alias_table.model_aliases if alias_table is not None else None


def _team_member(team_object: LiteLLM_TeamTable, user_id: str | None) -> Member | None:
    """A user listed twice gets the last entry, which is the one the key read's SQL join picks."""
    if user_id is None:
        return None
    entries: Final = tuple(member for member in team_object.members_with_roles if member.user_id == user_id)
    return entries[-1] if entries else None


def _team_row_grants(team_object: LiteLLM_TeamTable, user_id: str | None) -> TeamRowGrants:
    alias_table: Final = _json_column(team_object, _ALIAS_COLUMN).litellm_model_table
    return TeamRowGrants(
        team_alias=team_object.team_alias,
        team_tpm_limit=team_object.tpm_limit,
        team_rpm_limit=team_object.rpm_limit,
        team_tpd_limit=team_object.tpd_limit,
        team_max_budget=team_object.max_budget,
        team_soft_budget=team_object.soft_budget,
        team_model_max_budget=team_object.model_max_budget,
        team_spend=team_object.spend,
        team_models=list(team_object.models),
        team_blocked=team_object.blocked,
        team_metadata=_json_column(team_object, _METADATA_COLUMN).metadata,
        team_model_aliases=alias_table.model_aliases if alias_table is not None else None,
        team_object_permission_id=team_object.object_permission_id,
        team_object_permission=team_object.object_permission,
        team_member=_team_member(team_object, user_id),
    )


def team_grants(
    team_object: LiteLLM_TeamTable | None,
    team_membership: LiteLLM_TeamMembership | None,
    user_id: str | None,
) -> TeamGrants:
    if team_object is None:
        return TeamGrants()
    return TeamGrants(
        **_team_row_grants(team_object, user_id),
        team_member_spend=team_membership.spend if team_membership is not None else None,
        team_member_tpm_limit=(
            team_membership.safe_get_team_member_tpm_limit() if team_membership is not None else None
        ),
        team_member_rpm_limit=(
            team_membership.safe_get_team_member_rpm_limit() if team_membership is not None else None
        ),
    )


def team_row_update(token: UserAPIKeyAuth, team_object: LiteLLM_TeamTable) -> Mapping[str, object]:
    """The fields a key takes from its team row's current values. Its membership spend and limits stay as the
    key read them, and so do its aliases when the row names an alias table it was cached without."""
    grants: Final = _team_row_grants(team_object, token.user_id)
    aliases_unread: Final = team_object.model_id is not None and team_object.litellm_model_table is None
    return {**grants, "team_model_aliases": token.team_model_aliases} if aliases_unread else grants


def newer_team_row_update(token: UserAPIKeyAuth, team_object: LiteLLM_TeamTableCachedObj) -> Mapping[str, object]:
    """The team row's values when the team was read after the key, else nothing: an older team must not
    overwrite what a newer key read holds."""
    if team_object.last_refreshed_at is None or token.last_refreshed_at is None:
        return MappingProxyType({})
    if team_object.last_refreshed_at <= token.last_refreshed_at:
        return MappingProxyType({})
    return team_row_update(token, team_object)


def with_team_row(token: UserAPIKeyAuth, team_object: LiteLLM_TeamTable) -> UserAPIKeyAuth:
    return token.model_copy(update=team_row_update(token, team_object))
