from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class AccessGroupCreateRequest(BaseModel):
    access_group_name: str
    description: Optional[str] = None
    access_model_names: Optional[List[str]] = None
    access_mcp_server_ids: Optional[List[str]] = None
    access_agent_ids: Optional[List[str]] = None
    assigned_team_ids: Optional[List[str]] = None
    assigned_key_ids: Optional[List[str]] = None


class AccessGroupUpdateRequest(BaseModel):
    access_group_name: Optional[str] = None
    description: Optional[str] = None
    access_model_names: Optional[List[str]] = None
    access_mcp_server_ids: Optional[List[str]] = None
    access_agent_ids: Optional[List[str]] = None
    assigned_team_ids: Optional[List[str]] = None
    assigned_key_ids: Optional[List[str]] = None


class AccessGroupResponse(BaseModel):
    access_group_id: str
    access_group_name: str
    description: Optional[str] = None
    access_model_names: List[str]
    access_mcp_server_ids: List[str]
    access_agent_ids: List[str]
    assigned_team_ids: List[str]
    assigned_key_ids: List[str]
    created_at: datetime
    created_by: Optional[str] = None
    updated_at: datetime
    updated_by: Optional[str] = None
