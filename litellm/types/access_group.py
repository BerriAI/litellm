from datetime import datetime

from pydantic import BaseModel


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
