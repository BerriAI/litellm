"""
Team table models.

Canonical definitions for ``litellm_teamtable`` (plus the shared Member and
budget-window value types and the team-model alias table). Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

import json
from datetime import datetime
from typing import Final, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


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
    admins: list = []
    members: list = []
    members_with_roles: list[Member] = []
    team_member_permissions: list[str] | None = None
    metadata: dict | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    max_budget: float | None = None
    soft_budget: float | None = None
    budget_duration: str | None = None
    budget_limits: list[BudgetLimitEntry] | None = None
    models: list = []
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
