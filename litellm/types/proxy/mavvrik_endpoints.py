"""
Mavvrik endpoint Pydantic models for LiteLLM Proxy admin API.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MavvrikInitRequest(BaseModel):
    """Request body for POST /mavvrik/init — stores encrypted settings in LiteLLM_Config."""

    api_key: str = Field(..., description="Mavvrik API key (x-api-key header value)")
    api_endpoint: str = Field(
        ..., description="Mavvrik API base URL (e.g. https://api.mavvrik.dev)"
    )
    tenant: str = Field(..., description="Mavvrik tenant slug used in upload-url path")
    instance_id: str = Field(
        ...,
        description="Cluster/instance ID used in upload-url path (placeholder: any string for testing)",
    )
    timezone: str = Field(default="UTC", description="Timezone for date handling")


class MavvrikInitResponse(BaseModel):
    """Response for POST /mavvrik/init."""

    message: str
    status: str


class MavvrikExportRequest(BaseModel):
    """Request body for POST /mavvrik/export and POST /mavvrik/dry-run."""

    limit: Optional[int] = Field(
        None,
        description="Max spend rows to fetch (default: MAVVRIK_MAX_FETCHED_DATA_RECORDS)",
    )
    start_time_utc: Optional[datetime] = Field(
        None,
        description="Override window start (default: last marker or 2× interval ago)",
    )
    end_time_utc: Optional[datetime] = Field(
        None, description="Override window end (default: now)"
    )


class MavvrikExportResponse(BaseModel):
    """Response for POST /mavvrik/export."""

    message: str
    status: str
    records_exported: Optional[int] = None


class MavvrikDryRunResponse(BaseModel):
    """Response for POST /mavvrik/dry-run — returns transformed data without uploading."""

    message: str
    status: str
    dry_run_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Sample of raw spend rows and CSV preview (first 5000 chars)",
    )
    summary: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregate stats: total_records, total_cost, total_tokens, unique_models, unique_teams",
    )


class MavvrikSettingsView(BaseModel):
    """Response for GET /mavvrik/settings — API key is masked."""

    api_key_masked: Optional[str] = Field(
        None, description="Masked API key (first 4 + last 4 chars)"
    )
    api_endpoint: Optional[str] = None
    tenant: Optional[str] = None
    instance_id: Optional[str] = None
    timezone: Optional[str] = None
    marker: Optional[str] = Field(
        None, description="Last successfully uploaded interval timestamp"
    )
    status: Optional[str] = None


class MavvrikSettingsUpdate(BaseModel):
    """Request body for PUT /mavvrik/settings — all fields optional."""

    api_key: Optional[str] = Field(None, description="New Mavvrik API key")
    api_endpoint: Optional[str] = Field(None, description="New Mavvrik API base URL")
    tenant: Optional[str] = None
    instance_id: Optional[str] = None
    timezone: Optional[str] = None
