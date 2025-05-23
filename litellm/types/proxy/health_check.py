from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class HealthCheckResponseElement(BaseModel):
    model: str
    model_id: str
    cache: Optional[Dict[str, Any]] = {}
    x_ratelimit_remaining_requests: Optional[int] = Field(
        None, alias="x-ratelimit-remaining-requests"
    )
    x_ratelimit_remaining_tokens: Optional[int] = Field(
        None, alias="x-ratelimit-remaining-tokens"
    )
    status: Optional[Literal["healthy", "unhealthy"]] = "healthy"

    #########################################################
    # Error handling
    #########################################################
    error: Optional[str] = None
    raw_request_typed_dict: Optional[Dict[str, Any]] = None
    raw_request_body: Optional[str] = None
    raw_request_headers: Optional[Dict[str, Any]] = None
