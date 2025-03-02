# +-------------------------------------------------------------+
#
#           Use Lasso Security Guardrails for your LLM calls
#                   https://www.lasso.security/
#
# +-------------------------------------------------------------+

import os
import sys
from typing import Dict, List, Literal, Optional, Any

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth


class LassoGuardrailMissingSecrets(Exception):
    pass


class LassoGuardrailAPIError(Exception):
    """Exception raised when there's an error calling the Lasso API."""

    pass


class LassoGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_key = api_key or os.environ.get("LASSO_API_KEY")
        self.user_id = user_id or os.environ.get("LASSO_USER_ID")
        self.conversation_id = conversation_id or os.environ.get("LASSO_CONVERSATION_ID")

        if not self.api_key:
            msg = (
                "Couldn't get Lasso api key, either set the `LASSO_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise LassoGuardrailMissingSecrets(msg)

        self.api_base = api_base or "https://server.lasso.security/gateway/v2/classify"
        super().__init__(**kwargs)

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal["completion", "text_completion"],
    ) -> Exception | str | dict | None:
        verbose_proxy_logger.debug("Inside Lasso Pre-Call Hook")
        messages: List[Dict[str, str]] = data.get("messages", [])

        # check if messages are present
        if not messages:
            return data

        # Prepare headers
        headers = {"lasso-api-key": self.api_key, "Content-Type": "application/json"}

        # Add optional headers if provided
        if self.user_id:
            headers["lasso-user-id"] = self.user_id

        if self.conversation_id:
            headers["lasso-conversation-id"] = self.conversation_id

        # Prepare request payload - send all messages
        payload = {"messages": messages}

        try:
            verbose_proxy_logger.debug(f"Sending request to Lasso API: {payload}")
            response = await self.async_handler.post(
                url=self.api_base,
                headers=headers,
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()
            res = response.json()

            verbose_proxy_logger.debug(f"Lasso API response: {res}")

            # Check for violations directly in the response
            if res and res.get("violations_detected") is True:
                # Find which deputies detected violations
                violated_deputies = []
                if "deputies" in res:
                    for deputy, is_violated in res["deputies"].items():
                        if is_violated:
                            violated_deputies.append(deputy)

                verbose_proxy_logger.warning(f"Lasso guardrail detected violations: {violated_deputies}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Violated Lasso guardrail policy",
                        "detection_message": f"Guardrail violations detected: {', '.join(violated_deputies)}",
                        "lasso_response": res,
                    },
                )
            return data
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            verbose_proxy_logger.error(f"Error calling Lasso API: {str(e)}")
            # Instead of allowing the request to proceed, raise an exception
            raise LassoGuardrailAPIError(f"Failed to verify request safety with Lasso API: {str(e)}")

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "text_completion"],
    ) -> Exception | str | dict | None:
        """
        This is used for during_call moderation
        """
        # Use a cache instance from the parent class or create a new one if needed
        cache = getattr(self, "cache", DualCache())

        return await self.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=data,
            call_type=call_type,
        )
