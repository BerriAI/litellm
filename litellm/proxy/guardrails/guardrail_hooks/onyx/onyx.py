# +-------------------------------------------------------------+
#
#           Use Onyx Guardrails for your LLM calls
#                   https://onyx.security/
#
# +-------------------------------------------------------------+
from enum import Enum
import os
from typing import TYPE_CHECKING, Any, Optional, Type, Union
import uuid

from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypesLiteral, LLMResponseTypes

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class HookType(Enum):
    """Enum for processing hook types."""

    PRE_CALL = "pre_call"
    POST_CALL = "post_call"
    MODERATION = "moderation"


class OnyxGuardrail(CustomGuardrail):
    def __init__(self, api_base: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_base = api_base or os.getenv(
            "ONYX_API_BASE",
            "https://ai-guard.onyx.security",
        )
        self.api_key = api_key or os.getenv("ONYX_API_KEY")
        if not self.api_key:
            raise ValueError("ONYX_API_KEY environment variable is not set")
        self.optional_params = kwargs
        super().__init__(**kwargs)
        verbose_proxy_logger.info(f"OnyxGuard initialized with server: {self.api_base}")

    async def _validate_with_guard_server(
        self,
        payload: Any,
        hook_type: HookType,
        conversation_id: str,
    ) -> dict:
        """
        Call external Onyx Guard server for validation
        """
        response = await self.async_handler.post(
            f"{self.api_base}/guard/evaluate/v1/{self.api_key}/litellm",
            json={
                "payload": payload,
                "hook_type": hook_type.value,
                "conversation_id": conversation_id,
            },
            headers={
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        result = response.json()
        if not result.get("allowed", True):
            detection_message = "Unknown violation"
            if "violated_rules" in result:
                detection_message = ", ".join(result["violated_rules"])
            verbose_proxy_logger.warning(f"Request blocked by Onyx Guard. Violations: {detection_message}.")
            raise HTTPException(
                status_code=400,
                detail=f"Request blocked by Onyx Guard. Violations: {detection_message}.",
            )
        return result

    def _handle_conversation_id(self, data: dict) -> str:
        """
        Handle the conversation ID for the request
        """

        conversation_id = data.get("litellm_call_id")
        if conversation_id:
            return conversation_id

        conversation_id = data.get("_onyx_internal", {}).get("conversation_id")
        if conversation_id:
            return conversation_id

        conversation_id = str(uuid.uuid4())
        data.setdefault("_onyx_internal", {})["conversation_id"] = conversation_id
        return conversation_id

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        """
        Validate and modify input before sending to LLM
        """
        verbose_proxy_logger.info("Running Onyx Guard pre-call hook")

        messages = data.get("messages", [])
        if not messages:
            return data

        conversation_id = self._handle_conversation_id(data)

        try:
            await self._validate_with_guard_server(messages, HookType.PRE_CALL, conversation_id)
            return data
        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_proxy_logger.error(f"Error in pre-call guard: {str(e)}")
            return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        """
        Reject requests that violate Onyx Guard policies
        """
        verbose_proxy_logger.info("Running Onyx Guard moderation hook")

        messages = data.get("messages", [])
        if not messages:
            return data

        conversation_id = self._handle_conversation_id(data)

        try:
            await self._validate_with_guard_server(messages, HookType.MODERATION, conversation_id)
            return data
        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_proxy_logger.error(f"Error in moderation Onyx Guard: {str(e)}")
            return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ):
        """
        Validate LLM response before returning to client
        """
        verbose_proxy_logger.info("Running post-call guard hook")

        conversation_id = self._handle_conversation_id(data)

        try:
            # Convert response to dict format for validation
            if isinstance(response, dict):
                # TypedDict or plain dict
                payload = response
            elif hasattr(response, "model_dump"):
                # Pydantic model
                payload = response.model_dump()
            else:
                # Fallback: use the response as-is
                payload = response

            await self._validate_with_guard_server(
                payload=payload,
                hook_type=HookType.POST_CALL,
                conversation_id=conversation_id,
            )

            return response
        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_proxy_logger.error(f"Error in post-call Onyx Guard: {str(e)}")
            return response

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.onyx import (
            OnyxGuardrailConfigModel,
        )

        return OnyxGuardrailConfigModel
