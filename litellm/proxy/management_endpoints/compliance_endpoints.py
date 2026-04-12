"""
COMPLIANCE CHECK ENDPOINTS

Endpoints for checking regulatory compliance of LLM request logs.

/compliance/eu-ai-act - Check EU AI Act compliance
/compliance/gdpr      - Check GDPR compliance
"""

from fastapi import APIRouter, Depends, Request

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.compliance_checks import ComplianceChecker
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.types.proxy.compliance_endpoints import (
    ComplianceCheckRequest,
    ComplianceResponse,
)

router = APIRouter()


@router.post(
    "/compliance/eu-ai-act",
    tags=["compliance"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ComplianceResponse,
)
@management_endpoint_wrapper
async def check_eu_ai_act_compliance(
    data: ComplianceCheckRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> ComplianceResponse:
    """
    Check EU AI Act compliance for a spend log entry.

    Checks:
    - Art. 9: Guardrails applied (any guardrail)
    - Art. 5: Content screened before LLM (pre-call guardrails)
    - Art. 12: Audit record complete (user_id, model, timestamp, guardrail_results)
    """
    checker = ComplianceChecker(data)
    checks = checker.check_eu_ai_act()
    return ComplianceResponse(
        compliant=all(c.passed for c in checks),
        regulation="EU AI Act",
        checks=checks,
    )


@router.post(
    "/compliance/gdpr",
    tags=["compliance"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ComplianceResponse,
)
@management_endpoint_wrapper
async def check_gdpr_compliance(
    data: ComplianceCheckRequest,
    http_request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> ComplianceResponse:
    """
    Check GDPR compliance for a spend log entry.

    Checks:
    - Art. 32: Data protection applied (pre-call guardrails)
    - Art. 5(1)(c): Sensitive data protected (masked/blocked or no issues)
    - Art. 30: Audit record complete (user_id, model, timestamp, guardrail_results)
    """
    checker = ComplianceChecker(data)
    checks = checker.check_gdpr()
    return ComplianceResponse(
        compliant=all(c.passed for c in checks),
        regulation="GDPR",
        checks=checks,
    )
