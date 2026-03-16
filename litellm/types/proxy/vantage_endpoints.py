"""
Vantage endpoint types for LiteLLM Proxy
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class VantageInitRequest(BaseModel):
    """Request model for initializing Vantage settings"""

    api_key: str = Field(..., description="Vantage API key for authentication")
    integration_token: str = Field(
        ..., description="Vantage integration token for the cost-import endpoint"
    )
    base_url: str = Field(
        default="https://api.vantage.sh",
        description="Vantage API base URL (default: https://api.vantage.sh)",
    )

    @field_validator("api_key", "integration_token", "base_url")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be a non-empty string")
        return v


class VantageInitResponse(BaseModel):
    """Response model for Vantage initialization"""

    message: str
    status: str


class VantageExportRequest(BaseModel):
    """Request model for Vantage export operations (actual export, no default limit)"""

    limit: Optional[int] = Field(
        None,
        description="Optional limit on number of records to export (default: no limit)",
    )
    start_time_utc: Optional[datetime] = Field(
        None, description="Start time for data export in UTC"
    )
    end_time_utc: Optional[datetime] = Field(
        None, description="End time for data export in UTC"
    )


class VantageDryRunRequest(BaseModel):
    """Request model for Vantage dry-run operations (capped for preview)"""

    limit: Optional[int] = Field(
        500, description="Limit on number of records to preview (default: 500)"
    )


class VantageExportResponse(BaseModel):
    """Response model for Vantage export operations"""

    message: str
    status: str
    dry_run_data: Optional[Dict[str, Any]] = Field(
        None, description="Dry run data including usage data and FOCUS transformed data"
    )
    summary: Optional[Dict[str, Any]] = Field(
        None, description="Summary statistics for dry run"
    )


class VantageSettingsView(BaseModel):
    """Response model for viewing Vantage settings with masked API key"""

    api_key_masked: Optional[str] = Field(
        None,
        description="Masked API key showing only first 4 and last 4 characters",
    )
    integration_token_masked: Optional[str] = Field(
        None,
        description="Masked integration token showing only first 4 and last 4 characters",
    )
    base_url: Optional[str] = Field(None, description="Vantage API base URL")
    status: Optional[str] = Field(None, description="Configuration status")


class VantageSettingsUpdate(BaseModel):
    """Request model for updating Vantage settings"""

    api_key: Optional[str] = Field(
        None, description="New Vantage API key for authentication"
    )
    integration_token: Optional[str] = Field(
        None, description="New Vantage integration token"
    )
    base_url: Optional[str] = Field(None, description="New Vantage API base URL")

    @field_validator("api_key", "integration_token", "base_url")
    @classmethod
    def must_be_non_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("must be a non-empty string")
        return v
