"""A pre_call guardrail loaded from the operator's own module, the way a deployment's custom guardrail is."""

import os
from typing import Final

import httpx
from fastapi import HTTPException

from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral


class OperatorGuardrail(CustomGuardrail):
    """Posts the requested model to ``OPERATOR_GUARDRAIL_URL`` and rejects with 403 when the reply says block."""

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: CallTypesLiteral
    ) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            response: Final = await client.post(os.environ["OPERATOR_GUARDRAIL_URL"], json={"model": data.get("model")})
        if response.json()["action"] == "block":
            raise HTTPException(status_code=403, detail="blocked by the operator guardrail")
