"""Pydantic types for webhook subscriptions (S6-04)."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Known event names that the dispatcher emits. Subscribers may register any
# string; unknown events simply never fire. Kept here for SDK type generation.
KNOWN_WEBHOOK_EVENTS = (
    "capability.invoked",
    "budget.exhausted",
    "agent.healthcheck.failed",
    "mcp.tool.called",
)


class WebhookSubscriptionCreate(BaseModel):
    target_url: str
    events: List[str] = Field(..., min_length=1)
    app_id: Optional[str] = None
    team_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    is_active: bool = True


class WebhookSubscriptionPatch(BaseModel):
    target_url: Optional[str] = None
    events: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class WebhookSubscription(BaseModel):
    subscription_id: str
    app_id: Optional[str] = None
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    events: List[str] = Field(default_factory=list)
    target_url: str
    filters: Optional[Dict[str, Any]] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None
    consecutive_failures: int = 0


class WebhookSubscriptionCreateResponse(WebhookSubscription):
    """Response from POST /v1/webhooks.

    Carries the unhashed ``secret`` ONE TIME — the caller must store it
    immediately, the proxy keeps only the bcrypt hash.
    """

    secret: str
