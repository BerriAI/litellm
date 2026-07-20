"""
Team table models.

Canonical definitions for ``litellm_teamtable`` (plus the shared Member and
budget-window value types and the team-model alias table). Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

import json
from datetime import datetime
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.types.llms.base import LiteLLMPydanticObjectBase


class MemberBase(LiteLLMPydanticObjectBase):
    user_id: Optional[str] = Field(
        default=None,
        description="The unique ID of the user to add. Either user_id or user_email must be provided",
    )
    user_email: Optional[str] = Field(
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
    reset_at: Optional[datetime] = None


class LiteLLM_ModelTable(LiteLLMPydanticObjectBase):
    id: Optional[int] = None
    model_aliases: Optional[Union[str, dict]] = None
    created_by: str
    updated_by: str
    team: Optional["LiteLLM_TeamTable"] = None

    model_config = ConfigDict(protected_namespaces=())


class TeamBase(LiteLLMPydanticObjectBase):
    team_alias: Optional[str] = None
    team_id: Optional[str] = None
    organization_id: Optional[str] = None
    admins: list = []
    members: list = []
    members_with_roles: List[Member] = []
    team_member_permissions: Optional[List[str]] = None
    metadata: Optional[dict] = None
    tpm_limit: Optional[int] = None
    rpm_limit: Optional[int] = None
    max_budget: Optional[float] = None
    soft_budget: Optional[float] = None
    budget_duration: Optional[str] = None
    budget_limits: Optional[List[BudgetLimitEntry]] = None
    models: list = []
    blocked: bool = False
    router_settings: Optional[dict] = None
    access_group_ids: Optional[List[str]] = None
    default_team_member_models: Optional[List[str]] = None
    model_max_budget: Optional[dict] = None


class LiteLLM_TeamTable(TeamBase):
    team_id: str  # type: ignore
    spend: Optional[float] = None
    max_parallel_requests: Optional[int] = None
    budget_duration: Optional[str] = None
    budget_reset_at: Optional[datetime] = None
    model_id: Optional[int] = None
    model_spend: Optional[dict] = {}
    model_max_budget: Optional[dict] = {}
    policies: Optional[List[str]] = None
    allow_team_guardrail_config: Optional[bool] = False
    litellm_model_table: Optional[LiteLLM_ModelTable] = None
    object_permission: Optional[LiteLLM_ObjectPermissionTable] = None
    object_permission_id: Optional[str] = None
    updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def set_model_info(cls, values):
        dict_fields = [
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
    last_refreshed_at: Optional[float] = None


class LiteLLM_DeletedTeamTable(LiteLLM_TeamTable):
    """Audit record for deleted teams; mirrors the team plus deletion metadata."""

    id: Optional[str] = None
    deleted_at: Optional[datetime] = None
    deleted_by: Optional[str] = None
    deleted_by_api_key: Optional[str] = None
    litellm_changed_by: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


LiteLLM_ModelTable.model_rebuild()
