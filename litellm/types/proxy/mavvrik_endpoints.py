"""
Mavvrik endpoint Pydantic models for LiteLLM Proxy admin API.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class MavvrikInitRequest(BaseModel):
    """Request body for POST /mavvrik/init — stores encrypted settings in LiteLLM_Config."""

    api_key: str = Field(
        ..., description="Mavvrik API key (x-api-key header value)", repr=False
    )
    api_endpoint: str = Field(
        ...,
        description="Mavvrik API base URL including tenant (e.g. https://api.mavvrik.dev/my-tenant)",
    )
    connection_id: str = Field(
        ...,
        description="Connection/instance ID used in the agent path",
    )

    @field_validator("api_key", "api_endpoint", "connection_id")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v


class MavvrikInitResponse(BaseModel):
    """Response for POST /mavvrik/init."""

    message: str
    status: str


class MavvrikDeleteResponse(BaseModel):
    """Response for DELETE /mavvrik/delete."""

    message: str
    status: str


class MavvrikExportRequest(BaseModel):
    """Request body for POST /mavvrik/export and POST /mavvrik/dry-run."""

    date_str: Optional[str] = Field(
        None,
        description="Date to export in YYYY-MM-DD format (default: yesterday). "
        "Re-uploading the same date overwrites the previous upload — idempotent.",
    )
    limit: Optional[int] = Field(
        None,
        description="Max spend rows to fetch (default: MAVVRIK_MAX_FETCHED_DATA_RECORDS)",
    )

    @field_validator("date_str")
    @classmethod
    def must_be_valid_date_if_set(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            from datetime import date

            date.fromisoformat(v)
        except ValueError:
            raise ValueError("date_str must be a valid date in YYYY-MM-DD format")
        return v


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
    """Response for GET /mavvrik/settings — API key is masked.

    The export marker (cursor) is owned by the Mavvrik API and is not
    stored or exposed locally.
    """

    api_key_masked: Optional[str] = Field(None, description="Masked API key")
    api_endpoint: Optional[str] = None
    connection_id: Optional[str] = None
    status: Optional[str] = None


class MavvrikSettingsUpdate(BaseModel):
    """Request body for PUT /mavvrik/settings — all fields optional.

    Only credentials can be updated. The export marker is owned by the
    Mavvrik API and is not settable here.
    """

    api_key: Optional[str] = Field(None, description="New Mavvrik API key")
    api_endpoint: Optional[str] = Field(
        None, description="New Mavvrik API base URL (includes tenant)"
    )
    connection_id: Optional[str] = None

    @field_validator("api_key", "api_endpoint", "connection_id")
    @classmethod
    def must_not_be_empty_if_set(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("must not be empty if provided")
        return v
