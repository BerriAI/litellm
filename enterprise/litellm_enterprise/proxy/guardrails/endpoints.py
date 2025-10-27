"""
Enterprise Guardrail Routes on LiteLLM Proxy

To see all free guardrails see litellm/proxy/guardrails/*


Exposed Routes:
- /mask_pii
"""
from typing import Optional

from fastapi import APIRouter, Depends

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.guardrails.guardrail_endpoints import GUARDRAIL_REGISTRY
from litellm.types.guardrails import ApplyGuardrailRequest, ApplyGuardrailResponse

router = APIRouter(tags=["guardrails"], prefix="/guardrails")


@router.post("/apply_guardrail", response_model=ApplyGuardrailResponse)
async def apply_guardrail(
    request: ApplyGuardrailRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Mask PII from a given text, requires a guardrail to be added to litellm.
    """
    active_guardrail: Optional[
        CustomGuardrail
    ] = GUARDRAIL_REGISTRY.get_initialized_guardrail_callback(
        guardrail_name=request.guardrail_name
    )
    if active_guardrail is None:
        raise Exception(f"Guardrail {request.guardrail_name} not found")

    response_text = await active_guardrail.apply_guardrail(
        text=request.text, language=request.language, entities=request.entities
    )

    return ApplyGuardrailResponse(response_text=response_text)
