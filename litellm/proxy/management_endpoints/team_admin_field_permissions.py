"""Proxy-wide allow-list of what a team admin may do on the teams they administer: team-settings fields on
/team/update, plus the ``projects`` permission for /project/new and /project/update."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy._types import (
    LiteLLM_ManagementEndpoint_MetadataFields,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    UpdateTeamRequest,
)

TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING: Final = "team_admin_editable_team_fields"

# TODO(LIT-5722): add the remaining team settings one per PR, each with its value-diff tests and dashboard field
SUPPORTED_TEAM_ADMIN_EDITABLE_TEAM_FIELDS: Final[frozenset[str]] = frozenset({"tpm_limit", "rpm_limit", "max_budget"})
TEAM_ADMIN_PROJECTS_PERMISSION: Final = "projects"
SUPPORTED_TEAM_ADMIN_PERMISSIONS: Final[frozenset[str]] = SUPPORTED_TEAM_ADMIN_EDITABLE_TEAM_FIELDS | {
    TEAM_ADMIN_PROJECTS_PERMISSION
}

_FIELD_LIST: Final = TypeAdapter(list[str])
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_METADATA_FOLDED_FIELDS: Final[frozenset[str]] = frozenset(
    (*LiteLLM_ManagementEndpoint_MetadataFields, *LiteLLM_ManagementEndpoint_MetadataFields_Premium)
)
_SYSTEM_MANAGED_METADATA_KEYS: Final[frozenset[str]] = frozenset({"team_member_budget_id"})
_NOT_COLUMNS: Final[frozenset[str]] = frozenset({"team_id", "metadata"})
_SETTINGS_LOCATION: Final = "Settings > UI > Team admin editable fields"


@dataclass(frozen=True, slots=True)
class TeamAdminEditAllowed:
    request: UpdateTeamRequest
    kind: Literal["allowed"] = "allowed"


@dataclass(frozen=True, slots=True)
class TeamAdminEditingDisabled:
    kind: Literal["disabled"] = "disabled"


@dataclass(frozen=True, slots=True)
class TeamAdminFieldNotPermitted:
    field: str
    kind: Literal["field_not_permitted"] = "field_not_permitted"


TeamAdminEditVerdict: TypeAlias = TeamAdminEditAllowed | TeamAdminEditingDisabled | TeamAdminFieldNotPermitted


def resolve_team_admin_editable_fields(
    general_settings: Mapping[str, object],
    supported: frozenset[str],
) -> frozenset[str]:
    raw: Final = general_settings.get(TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING)
    if raw is None:
        return frozenset()
    try:
        configured: Final = frozenset(_FIELD_LIST.validate_python(raw))
    except ValidationError:
        verbose_proxy_logger.warning(
            "%s must be a list of field names; ignoring %r", TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING, raw
        )
        return frozenset()
    unsupported: Final = configured - supported - SUPPORTED_TEAM_ADMIN_PERMISSIONS
    if unsupported:
        verbose_proxy_logger.warning(
            "%s ignores unsupported field(s) %s; supported: %s",
            TEAM_ADMIN_EDITABLE_TEAM_FIELDS_SETTING,
            sorted(unsupported),
            sorted(supported | SUPPORTED_TEAM_ADMIN_PERMISSIONS),
        )
    return configured & supported


def team_admin_may_manage_projects(general_settings: Mapping[str, object]) -> bool:
    return TEAM_ADMIN_PROJECTS_PERMISSION in resolve_team_admin_editable_fields(
        general_settings, frozenset({TEAM_ADMIN_PROJECTS_PERMISSION})
    )


def _as_object(value: object) -> Mapping[str, object]:
    try:
        return _JSON_OBJECT.validate_json(value) if isinstance(value, str) else _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return _EMPTY


def _stored_metadata(existing: Mapping[str, object]) -> Mapping[str, object]:
    return _as_object(existing.get("metadata"))


def _submitted_metadata(
    data: UpdateTeamRequest, submitted: Mapping[str, object], existing: Mapping[str, object]
) -> Mapping[str, object]:
    """Metadata as it would be stored: the caller's dict (or the stored one) with top-level folded fields laid over."""
    base: Final = (
        _as_object(submitted.get("metadata")) if "metadata" in data.model_fields_set else _stored_metadata(existing)
    )
    folded: Final = data.model_fields_set & _METADATA_FOLDED_FIELDS
    return MappingProxyType({key: submitted[key] if key in folded else base[key] for key in base.keys() | folded})


def _metadata_changes(
    data: UpdateTeamRequest, submitted: Mapping[str, object], existing: Mapping[str, object]
) -> frozenset[str]:
    merged: Final = _submitted_metadata(data, submitted, existing)
    stored: Final = _stored_metadata(existing)
    return frozenset(
        key if key in _METADATA_FOLDED_FIELDS else "metadata"
        for key in (merged.keys() | stored.keys()) - _SYSTEM_MANAGED_METADATA_KEYS
        if merged.get(key) != stored.get(key)
    )


def _stored_model_aliases(existing_row: LiteLLM_TeamTable) -> Mapping[str, object]:
    table: Final = existing_row.litellm_model_table
    return _as_object(_JSON_OBJECT.validate_json(table.model_dump_json()).get("model_aliases")) if table else _EMPTY


def _column_changed(
    field: str, submitted: Mapping[str, object], existing: Mapping[str, object], existing_row: LiteLLM_TeamTable
) -> bool:
    if field == "model_aliases":
        return _as_object(submitted.get(field)) != _stored_model_aliases(existing_row)
    if field in LiteLLM_TeamTable.model_fields:
        return submitted.get(field) != existing.get(field)
    return True


def changed_team_fields(data: UpdateTeamRequest, existing_row: LiteLLM_TeamTable) -> frozenset[str]:
    """Logical field names whose stored value the request would change.

    Request and stored row are compared as JSON values so both sides share one representation. Fields the
    server folds into metadata are attributed to their own name whether they arrive top-level or inside
    ``metadata``; anything else in ``metadata`` is attributed to ``metadata``. Fields with no stored
    counterpart on the team row count as changed whenever they are sent.
    """
    submitted: Final = _JSON_OBJECT.validate_json(data.model_dump_json(exclude_unset=True))
    existing: Final = _JSON_OBJECT.validate_json(existing_row.model_dump_json())
    column_fields: Final = frozenset(data.model_fields_set) - _NOT_COLUMNS - _METADATA_FOLDED_FIELDS
    column_changes: Final = frozenset(
        field for field in column_fields if _column_changed(field, submitted, existing, existing_row)
    )
    return column_changes | _metadata_changes(data, submitted, existing)


def _only_changes(data: UpdateTeamRequest, changed: frozenset[str]) -> UpdateTeamRequest:
    """The request without the values it resends unchanged, which would otherwise still trigger derived writes
    such as a resent budget_duration pushing budget_reset_at back."""
    sent: Final = frozenset(data.model_fields_set)
    via_metadata: Final = frozenset({"metadata"}) if changed - sent else frozenset[str]()
    kept: Final = frozenset({"team_id"}) | (changed & sent) | via_metadata
    return UpdateTeamRequest.model_validate(data.model_dump(include=MappingProxyType({field: True for field in kept})))


def team_admin_edit_verdict(
    data: UpdateTeamRequest,
    existing: LiteLLM_TeamTable,
    permitted: frozenset[str],
) -> TeamAdminEditVerdict:
    if not permitted:
        return TeamAdminEditingDisabled()
    changed: Final = changed_team_fields(data, existing)
    blocked: Final = sorted(changed - permitted)
    if blocked:
        return TeamAdminFieldNotPermitted(field=blocked[0])
    return TeamAdminEditAllowed(request=_only_changes(data, changed))


def team_admin_request_or_raise(verdict: TeamAdminEditVerdict) -> UpdateTeamRequest:
    match verdict:
        case TeamAdminEditAllowed():
            return verdict.request
        case TeamAdminEditingDisabled():
            raise HTTPException(
                status_code=403,
                detail=(
                    "Team admins on this proxy cannot edit team settings. "
                    f"Ask a proxy admin to enable fields under {_SETTINGS_LOCATION}."
                ),
            )
        case TeamAdminFieldNotPermitted(field=field):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Team admins on this proxy do not have permission to update '{field}'. "
                    f"Ask a proxy admin to add it under {_SETTINGS_LOCATION}."
                ),
            )
        case _:
            assert_never(verdict)
