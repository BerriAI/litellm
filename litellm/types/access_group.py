from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AccessGroupCreateRequest(BaseModel):
    access_group_name: str
    description: str | None = None
    access_model_names: list[str] | None = None
    access_mcp_server_ids: list[str] | None = None
    access_agent_ids: list[str] | None = None
    assigned_team_ids: list[str] | None = None
    assigned_key_ids: list[str] | None = None


class AccessGroupUpdateRequest(BaseModel):
    access_group_name: str | None = None
    description: str | None = None
    access_model_names: list[str] | None = None
    access_mcp_server_ids: list[str] | None = None
    access_agent_ids: list[str] | None = None
    assigned_team_ids: list[str] | None = None
    assigned_key_ids: list[str] | None = None


class AccessGroupPatchRequest(AccessGroupUpdateRequest):
    """`PATCH /management/v1/access-groups/{id}` body: a JSON merge patch, so an unknown key is a 422 rather
    than a silent no-op."""

    model_config = ConfigDict(extra="forbid")


class AccessGroupResponse(BaseModel):
    access_group_id: str
    access_group_name: str
    description: str | None = None
    access_model_names: list[str]
    access_mcp_server_ids: list[str]
    access_agent_ids: list[str]
    assigned_team_ids: list[str]
    assigned_key_ids: list[str]
    created_at: datetime
    created_by: str | None = None
    updated_at: datetime
    updated_by: str | None = None
