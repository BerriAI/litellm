# +-------------------------------------------------------------+
#
#           Use Lasso Security Guardrails for your LLM calls
#                   https://www.lasso.security/
#
# +-------------------------------------------------------------+

import os
import sys

from litellm._logging import verbose_proxy_logger

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path
from typing import Literal, Optional, List, Dict

from fastapi import HTTPException

from litellm import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth


class LassoGuardrailMissingSecrets(Exception):
    pass


class LassoGuardrail(CustomGuardrail):
    def __init__(self, api_key: Optional[str] = None, guard_name: Optional[str] = None, **kwargs):
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_key = api_key or os.environ.get("LASSO_API_KEY")
        self.guardrails_ai_guard_name = guard_name
        if not self.api_key:
            msg = (
                "Couldn't get Lasso api key, either set the `AIM_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise LassoGuardrailMissingSecrets(msg)
        self.api_base = "https://server.lasso.security/gateway/v1"
        super().__init__(**kwargs)

    async def async_pre_call_hook(
            self,
            user_api_key_dict: UserAPIKeyAuth,
            cache: DualCache,
            data: dict,
            call_type: Literal[
                "completion",
                "text_completion"
            ],
    ) -> Exception | str | dict | None:
        verbose_proxy_logger.debug("Inside Lasso Pre-Call Hook")
        messages: List[Dict[str, str]] = data.get("messages", [])
        # check if messages are present
        if not messages:
            return data
        # take the last content as the prompt
        single_massage_obj: Dict[str,str] = messages[-1]
        # check if the role is system, if yes return data
        if single_massage_obj.get("role") == "system":
            return data
        prompt = single_massage_obj.get("content")
        headers = {"lasso-x-api-key": self.api_key}
        # TODO:
        response = await self.async_handler.post(
            url=self.api_base,
            headers=headers,
            json={"prompt": prompt},
        )
        response.raise_for_status()
        res = response.json()
        if res:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "lasso_response": res
                }
            )
        return data
