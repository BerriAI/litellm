from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AuditLogResponse(BaseModel):
    """Response model for a single audit log entry"""

    id: str
    updated_at: datetime
    changed_by: str
    changed_by_api_key: str
    action: str
    table_name: str
    object_id: str
    before_value: Optional[Dict[str, Any]] = None
    updated_values: Optional[Dict[str, Any]] = None
    object_alias: str | None = None
    object_team_id: str | None = None
    object_team_alias: str | None = None
    changed_by_user_email: str | None = None
    changed_by_key_alias: str | None = None


class PaginatedAuditLogResponse(BaseModel):
    """Response model for paginated audit logs"""

    audit_logs: List[AuditLogResponse]
    total: int = Field(..., description="Total number of audit logs matching the filters")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
