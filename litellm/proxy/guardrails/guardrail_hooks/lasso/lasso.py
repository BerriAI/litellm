# +-------------------------------------------------------------+
#
#           Use Lasso Security Guardrails for your LLM calls
#                   https://www.lasso.security/
#
# +-------------------------------------------------------------+

import os
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class LassoGuardrailMissingSecrets(Exception):
    pass


class LassoGuardrailAPIError(Exception):
    """Exception raised when there's an error calling the Lasso API."""

    pass


class LassoGuardrail(CustomGuardrail):
    def __init__(
        self,
        lasso_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.lasso_api_key = lasso_api_key or api_key or os.environ.get("LASSO_API_KEY")
        self.user_id = user_id or os.environ.get("LASSO_USER_ID")
        self.conversation_id = conversation_id or os.environ.get(
            "LASSO_CONVERSATION_ID"
        )

        if self.lasso_api_key is None:
            msg = (
                "Couldn't get Lasso api key, either set the `LASSO_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise LassoGuardrailMissingSecrets(msg)

        self.api_base = (
            api_base or os.getenv("LASSO_API_BASE") or "https://server.lasso.security"
        )
        super().__init__(**kwargs)

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside Lasso Pre-Call Hook")
        return await self.run_lasso_guardrail(data)

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
        ],
    ):
        """
        This is used for during_call moderation
        """
        verbose_proxy_logger.debug("Inside Lasso Moderation Hook")
        return await self.run_lasso_guardrail(data)

    async def run_lasso_guardrail(
        self,
        data: dict,
    ):
        """
        Run the Lasso guardrail

        Raises:
            LassoGuardrailAPIError: If the Lasso API call fails
        """
        messages: List[Dict[str, str]] = data.get("messages", [])
        # check if messages are present
        if not messages:
            return data

        try:
            headers = self._prepare_headers()
            payload = self._prepare_payload(messages)

            response = await self._call_lasso_api(
                headers=headers,
                payload=payload,
            )
            self._process_lasso_response(response)

            return data
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            verbose_proxy_logger.error(f"Error calling Lasso API: {str(e)}")
            # Instead of allowing the request to proceed, raise an exception
            raise LassoGuardrailAPIError(
                f"Failed to verify request safety with Lasso API: {str(e)}"
            )

    def _prepare_headers(self) -> dict[str, str]:
        """Prepare headers for the Lasso API request."""
        if not self.lasso_api_key:
            msg = (
                "Couldn't get Lasso api key, either set the `LASSO_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise LassoGuardrailMissingSecrets(msg)

        headers: dict[str, str] = {
            "lasso-api-key": self.lasso_api_key,
            "Content-Type": "application/json",
        }

        # Add optional headers if provided
        if self.user_id:
            headers["lasso-user-id"] = self.user_id

        if self.conversation_id:
            headers["lasso-conversation-id"] = self.conversation_id

        return headers

    def _prepare_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Prepare the payload for the Lasso API request."""
        return {"messages": messages}

    async def _call_lasso_api(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call the Lasso API and return the response."""
        verbose_proxy_logger.debug(f"Sending request to Lasso API: {payload}")
        response = await self.async_handler.post(
            url=f"{self.api_base}/gateway/v2/classify",
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        res = response.json()
        verbose_proxy_logger.debug(f"Lasso API response: {res}")
        return res

    def _process_lasso_response(self, response: Dict[str, Any]) -> None:
        """Process the Lasso API response and raise exceptions if violations are detected."""
        if response and response.get("violations_detected") is True:
            violated_deputies = self._parse_violated_deputies(response)
            verbose_proxy_logger.warning(
                f"Lasso guardrail detected violations: {violated_deputies}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated Lasso guardrail policy",
                    "detection_message": f"Guardrail violations detected: {', '.join(violated_deputies)}",
                    "lasso_response": response,
                },
            )

    def _parse_violated_deputies(self, response: Dict[str, Any]) -> List[str]:
        """Parse the response to extract violated deputies."""
        violated_deputies = []
        if "deputies" in response:
            for deputy, is_violated in response["deputies"].items():
                if is_violated:
                    violated_deputies.append(deputy)
        return violated_deputies

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.lasso import (
            LassoGuardrailConfigModel,
        )

        return LassoGuardrailConfigModel
