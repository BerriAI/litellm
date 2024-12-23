"""
CRUD ENDPOINTS FOR GUARDRAILS
"""

from typing import Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, status

from litellm.proxy._types import CommonProxyErrors
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

#### GUARDRAILS ENDPOINTS ####

router = APIRouter()


def _get_guardrail_names_from_config(guardrails_config: List[Dict]) -> List[str]:
    return [guardrail["guardrail_name"] for guardrail in guardrails_config]


@router.get(
    "/guardrails/list",
    tags=["Guardrails"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_guardrails():
    """
    List the guardrails that are available on the proxy server
    """
    from litellm.proxy.proxy_server import premium_user, proxy_config

    if not premium_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": CommonProxyErrors.not_premium_user.value,
            },
        )

    config = proxy_config.config

    _guardrails_config = cast(Optional[list[dict]], config.get("guardrails"))

    if _guardrails_config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "No guardrails found in config"},
        )

    return _get_guardrail_names_from_config(config["guardrails"])
