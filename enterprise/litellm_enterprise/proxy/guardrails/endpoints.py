"""
Enterprise Guardrail Routes on LiteLLM Proxy

To see all free guardrails see litellm/proxy/guardrails/*


Exposed Routes:
- /mask_pii
- /virtual_key/guardrails
"""
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.guardrails.guardrail_endpoints import GUARDRAIL_REGISTRY
from litellm.proxy.guardrails.guardrail_registry import IN_MEMORY_GUARDRAIL_HANDLER
from litellm.types.guardrails import ApplyGuardrailRequest, ApplyGuardrailResponse, Guardrail

# Models for virtual key guardrail management
class VirtualKeyGuardrailRequest(BaseModel):
    virtual_key_id: str
    guardrail_id: str

class VirtualKeyGuardrailsResponse(BaseModel):
    virtual_key_id: str
    guardrails: List[Guardrail]

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

    return await active_guardrail.apply_guardrail(
        text=request.text, language=request.language, entities=request.entities
    )

@router.post("/virtual_key/associate", response_model=Dict[str, str])
async def associate_guardrail_with_virtual_key(
    request: VirtualKeyGuardrailRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Associate a guardrail with a virtual key
    """
    # Check if guardrail exists
    guardrail = IN_MEMORY_GUARDRAIL_HANDLER.get_guardrail_by_id(request.guardrail_id)
    if not guardrail:
        raise HTTPException(status_code=404, detail=f"Guardrail {request.guardrail_id} not found")
    
    # Associate guardrail with virtual key
    IN_MEMORY_GUARDRAIL_HANDLER.associate_guardrail_with_virtual_key(
        virtual_key_id=request.virtual_key_id,
        guardrail_id=request.guardrail_id
    )
    
    return {"message": f"Guardrail {request.guardrail_id} associated with virtual key {request.virtual_key_id}"}

@router.post("/virtual_key/disassociate", response_model=Dict[str, str])
async def disassociate_guardrail_from_virtual_key(
    request: VirtualKeyGuardrailRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Disassociate a guardrail from a virtual key
    """
    # Disassociate guardrail from virtual key
    IN_MEMORY_GUARDRAIL_HANDLER.disassociate_guardrail_from_virtual_key(
        virtual_key_id=request.virtual_key_id,
        guardrail_id=request.guardrail_id
    )
    
    return {"message": f"Guardrail {request.guardrail_id} disassociated from virtual key {request.virtual_key_id}"}

@router.get("/virtual_key/{virtual_key_id}", response_model=VirtualKeyGuardrailsResponse)
async def get_guardrails_for_virtual_key(
    virtual_key_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get all guardrails associated with a virtual key
    """
    verbose_proxy_logger.debug(f"Getting guardrails for virtual key: {virtual_key_id}")
    guardrails = IN_MEMORY_GUARDRAIL_HANDLER.get_guardrails_for_virtual_key(virtual_key_id)
    verbose_proxy_logger.debug(f"Found {len(guardrails)} guardrails for virtual key {virtual_key_id}")
    
    return VirtualKeyGuardrailsResponse(
        virtual_key_id=virtual_key_id,
        guardrails=guardrails
    )
