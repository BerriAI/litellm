"""
Team table models.

Canonical definitions for ``litellm_teamtable`` (plus the shared Member and
budget-window value types and the team-model alias table). Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Final, Literal, Optional

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase

TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY: Final = "team_member_max_budget_alert_emails"


def _parse_team_member_budget_alert_threshold(raw: object) -> str:
    if isinstance(raw, str) and raw.isdigit() and len(raw) <= 3 and 1 <= int(raw) <= 100:
        return str(int(raw))
    raise ValueError(
        f"metadata.{TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY} thresholds must be whole-number percentages "
        f"from 1 to 100, got {raw!r}"
    )


def _is_plausible_email(raw: object) -> bool:
    if not isinstance(raw, str):
        return False
    local, at, domain = raw.strip().partition("@")
    return bool(local and at and domain) and not any(c.isspace() or c == "@" for c in local + domain)


def _parse_team_member_budget_alert_recipients(threshold: str, raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"metadata.{TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY}[{threshold!r}] must be a list of email addresses"
        )
    invalid: Final = [email for email in raw if not _is_plausible_email(email)]
    if invalid:
        raise ValueError(
            f"metadata.{TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY}[{threshold!r}] has invalid email addresses: {invalid!r}"
        )
    return list(dict.fromkeys(email.strip() for email in raw))


def validate_team_request_metadata(metadata: dict) -> dict:
    """
    Reject a malformed team_member_max_budget_alert_emails on write and store it canonically,
    e.g. {"50": [], "100": ["finance@x.com"]}, so a bad threshold is a 422 instead of an alert that never fires.
    """
    raw: Final = metadata.get(TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY)
    if raw is None:
        return metadata
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"metadata.{TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY} must map percentages to email lists, "
            'e.g. {"50": [], "100": ["finance@example.com"]}'
        )
    parsed: Final[dict[str, list[str]]] = {}
    for raw_threshold, raw_recipients in raw.items():
        threshold = _parse_team_member_budget_alert_threshold(raw_threshold)
        if threshold in parsed:
            raise ValueError(
                f"metadata.{TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY} lists the {threshold}% threshold more than once"
            )
        parsed[threshold] = _parse_team_member_budget_alert_recipients(threshold, raw_recipients)
    return {**metadata, TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY: parsed}


# Request-side only: LiteLLM_TeamTable keeps plain `dict` so reading a stored row never fails validation.
TeamRequestMetadata = Annotated[dict, AfterValidator(validate_team_request_metadata)]


class MemberBase(LiteLLMPydanticObjectBase):
    user_id: str | None = Field(
        default=None,
        description="The unique ID of the user to add. Either user_id or user_email must be provided",
    )
    user_email: str | None = Field(
        default=None,
        description="The email address of the user to add. Either user_id or user_email must be provided",
    )

    @model_validator(mode="before")
    @classmethod
    def check_user_info(cls, values):
        if not isinstance(values, dict):
            raise ValueError("input needs to be a dictionary")
        if values.get("user_id") is None and values.get("user_email") is None:
            raise ValueError("Either user id or user email must be provided")
        return values


class Member(MemberBase):
    role: Literal["admin", "user"] = Field(
        description="The role of the user within the team. 'admin' users can manage team settings and members, 'user' is a regular team member"
    )


class BudgetLimitEntry(LiteLLMPydanticObjectBase):
    """A single budget window with its own limit and independent reset schedule."""

    budget_duration: str
    max_budget: float
    reset_at: datetime | None = None


class LiteLLM_ModelTable(LiteLLMPydanticObjectBase):
    id: int | None = None
    model_aliases: str | dict | None = None
    created_by: str
    updated_by: str
    team: Optional["LiteLLM_TeamTable"] = None

    model_config = ConfigDict(protected_namespaces=())


class TeamBase(LiteLLMPydanticObjectBase):
    team_alias: str | None = None
    team_id: str | None = None
    organization_id: str | None = None
    admins: list[str] = []
    members: list[str] = []
    members_with_roles: list[Member] = []
    team_member_permissions: list[str] | None = None
    metadata: dict | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    tpd_limit: int | None = None
    max_budget: float | None = None
    soft_budget: float | None = None
    budget_duration: str | None = None
    budget_limits: list[BudgetLimitEntry] | None = None
    models: list[str] = []
    blocked: bool = False
    router_settings: dict | None = None
    access_group_ids: list[str] | None = None
    default_team_member_models: list[str] | None = None


class LiteLLM_TeamTable(TeamBase):
    team_id: str
    spend: float | None = None
    max_parallel_requests: int | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    model_id: int | None = None
    model_spend: dict | None = {}
    model_max_budget: dict | None = {}
    policies: list[str] | None = None
    allow_team_guardrail_config: bool | None = False
    litellm_model_table: LiteLLM_ModelTable | None = None
    object_permission: LiteLLM_ObjectPermissionTable | None = None
    object_permission_id: str | None = None
    updated_at: datetime | None = None
    created_at: datetime | None = None

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        dict_fields: Final = [
            "metadata",
            "aliases",
            "config",
            "permissions",
            "model_max_budget",
            "model_aliases",
            "router_settings",
            "budget_limits",
        ]

        if isinstance(values, BaseModel):
            values = values.model_dump()

        if isinstance(values.get("members_with_roles"), dict) and not values["members_with_roles"]:
            values["members_with_roles"] = []

        for field in dict_fields:
            value = values.get(field)
            if value is not None and isinstance(value, str):
                try:
                    values[field] = json.loads(value)
                except json.JSONDecodeError:
                    raise ValueError(f"Field {field} should be a valid dictionary")

        return values


class LiteLLM_TeamTableCachedObj(LiteLLM_TeamTable):
    last_refreshed_at: float | None = None


class LiteLLM_DeletedTeamTable(LiteLLM_TeamTable):
    """Audit record for deleted teams; mirrors the team plus deletion metadata."""

    id: str | None = None
    deleted_at: datetime | None = None
    deleted_by: str | None = None
    deleted_by_api_key: str | None = None
    litellm_changed_by: str | None = None

    model_config = ConfigDict(protected_namespaces=())


LiteLLM_ModelTable.model_rebuild()
