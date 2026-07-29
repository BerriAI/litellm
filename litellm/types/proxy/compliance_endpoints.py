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
    """

    request_id: str
    user_id: Optional[str] = None
    model: Optional[str] = None
    timestamp: Optional[str] = None
    guardrail_information: Optional[List[dict]] = None
