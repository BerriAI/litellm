"""Pydantic types for XCT Apps (S4-03)."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class XCTAppCreate(BaseModel):
    app_name: str = Field(..., description="Internal slug, unique. Used as login hint.")
    display_name: str
    description: Optional[str] = None
    icon_url: Optional[str] = None
    redirect_uris: List[str] = Field(default_factory=list)
    default_team_id: Optional[str] = None
    default_scopes: List[str] = Field(default_factory=list)
    capability_scope_id: Optional[str] = None
    rpm_limit: Optional[int] = None
    daily_budget: Optional[float] = None
    is_active: bool = True


class XCTAppPatch(BaseModel):
    display_name: Optional[str] = None
    description: Optional[str] = None
    icon_url: Optional[str] = None
    redirect_uris: Optional[List[str]] = None
    default_team_id: Optional[str] = None
    default_scopes: Optional[List[str]] = None
    capability_scope_id: Optional[str] = None
    rpm_limit: Optional[int] = None
    daily_budget: Optional[float] = None
    is_active: Optional[bool] = None


class XCTApp(BaseModel):
    """Public read shape — NEVER includes the client secret."""

    app_id: str
    app_name: str
    display_name: str
    description: Optional[str] = None
    icon_url: Optional[str] = None
    oauth_client_id: str
    redirect_uris: List[str] = Field(default_factory=list)
    default_team_id: Optional[str] = None
    default_scopes: List[str] = Field(default_factory=list)
    capability_scope_id: Optional[str] = None
    rpm_limit: Optional[int] = None
    daily_budget: Optional[float] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None


class XCTAppCreateResponse(XCTApp):
    """One-time response after create / rotate-secret.

    Carries ``client_secret`` — the cleartext, returned exactly once.
    """

    client_secret: str
