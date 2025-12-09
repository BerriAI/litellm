"""
FOCUS endpoint types for LiteLLM Proxy.

Types for FOCUS (FinOps Open Cost & Usage Specification) format export endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FOCUSExportRequest(BaseModel):
    """Request model for FOCUS export operations."""

    limit: Optional[int] = Field(
        None, 
        description="Optional limit on number of records to export"
    )
    start_time_utc: Optional[datetime] = Field(
        None, 
        description="Start time for data export in UTC"
    )
    end_time_utc: Optional[datetime] = Field(
        None, 
        description="End time for data export in UTC"
    )
    include_tags: bool = Field(
        True, 
        description="Whether to include tags in the FOCUS output"
    )
    include_token_breakdown: bool = Field(
        True, 
        description="Whether to include token breakdown in tags"
    )


class FOCUSSummary(BaseModel):
    """Summary statistics for FOCUS export."""

    total_records: int = Field(..., description="Total number of records exported")
    total_billed_cost: float = Field(..., description="Total billed cost across all records")
    total_consumed_quantity: int = Field(..., description="Total consumed quantity (tokens)")
    unique_providers: int = Field(..., description="Number of unique LLM providers")
    unique_sub_accounts: int = Field(..., description="Number of unique sub-accounts (teams)")


class FOCUSExportResponse(BaseModel):
    """Response model for FOCUS export operations."""

    message: str
    status: str
    format: str = Field(
        default="json",
        description="Export format (json, csv)"
    )
    data: Optional[Dict[str, Any]] = Field(
        None, 
        description="FOCUS formatted data"
    )
    summary: Optional[FOCUSSummary] = Field(
        None, 
        description="Summary statistics"
    )


class FOCUSExportJSONResponse(BaseModel):
    """Response model for FOCUS JSON export."""

    focus_version: str = Field(
        default="1.0",
        description="FOCUS specification version"
    )
    export_timestamp: str = Field(
        ...,
        description="ISO timestamp of export"
    )
    record_count: int = Field(
        ...,
        description="Number of records in export"
    )
    records: List[Dict[str, Any]] = Field(
        ...,
        description="FOCUS formatted records"
    )


class FOCUSDryRunResponse(BaseModel):
    """Response model for FOCUS dry run operations."""

    message: str
    status: str
    raw_data_sample: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Sample of raw database records (first 50)"
    )
    focus_data: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="FOCUS formatted data"
    )
    summary: Optional[FOCUSSummary] = Field(
        None,
        description="Summary statistics"
    )
