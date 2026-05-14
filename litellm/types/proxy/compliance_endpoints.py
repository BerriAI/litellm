from typing import List, Optional

from pydantic import BaseModel


class ComplianceCheckResult(BaseModel):
    """Result of a single compliance check."""

    check_name: str
    article: str
    passed: bool
    detail: str


class ComplianceResponse(BaseModel):
    """Response from a compliance check endpoint."""

    compliant: bool
    regulation: str
    checks: List[ComplianceCheckResult]


class ComplianceCheckRequest(BaseModel):
    """Request payload for compliance check endpoints.

    Mirrors the spend log fields needed for compliance evaluation.

    `user_id` carries the LiteLLM key owner (spend_log.user, sourced from
    `metadata.user_api_key_user_id`). `end_user_id` carries the OpenAI-spec
    `user` field of the request body (spend_log.end_user). Either one is
    sufficient to identify the data subject for EU AI Act Art. 12 and
    GDPR Art. 30 audit completeness.
    """

    request_id: str
    user_id: Optional[str] = None
    end_user_id: Optional[str] = None
    model: Optional[str] = None
    timestamp: Optional[str] = None
    guardrail_information: Optional[List[dict]] = None
