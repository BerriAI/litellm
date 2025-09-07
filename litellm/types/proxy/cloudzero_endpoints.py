"""
CloudZero endpoint types for LiteLLM Proxy
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CloudZeroInitRequest(BaseModel):
    """Request model for initializing CloudZero settings"""
    
    api_key: str = Field(..., description="CloudZero API key for authentication")
    connection_id: str = Field(..., description="CloudZero connection ID for data submission")
    timezone: str = Field(default="UTC", description="Timezone for date handling (default: UTC)")


class CloudZeroInitResponse(BaseModel):
    """Response model for CloudZero initialization"""
    
    message: str
    status: str


class CloudZeroExportRequest(BaseModel):
    """Request model for CloudZero export operations"""
    
    limit: Optional[int] = Field(None, description="Optional limit on number of records to export")
    operation: str = Field(default="replace_hourly", description="CloudZero operation type (replace_hourly or sum)")
    start_time_utc: Optional[datetime] = Field(None, description="Start time for data export in UTC")
    end_time_utc: Optional[datetime] = Field(None, description="End time for data export in UTC")


class CloudZeroExportResponse(BaseModel):
    """Response model for CloudZero export operations"""
    
    message: str
    status: str
    records_exported: Optional[int] = None
    dry_run_data: Optional[Dict[str, Any]] = Field(None, description="Dry run data including usage data and CBF transformed data")
    summary: Optional[Dict[str, Any]] = Field(None, description="Summary statistics for dry run")


class CloudZeroSettingsView(BaseModel):
    """Response model for viewing CloudZero settings with masked API key"""
    
    api_key_masked: str = Field(..., description="Masked API key showing only first 4 and last 4 characters")
    connection_id: str = Field(..., description="CloudZero connection ID for data submission")
    timezone: str = Field(..., description="Timezone for date handling")
    status: str = Field(..., description="Configuration status")


class CloudZeroSettingsUpdate(BaseModel):
    """Request model for updating CloudZero settings"""
    
    api_key: Optional[str] = Field(None, description="New CloudZero API key for authentication")
    connection_id: Optional[str] = Field(None, description="New CloudZero connection ID for data submission")
    timezone: Optional[str] = Field(None, description="New timezone for date handling") 