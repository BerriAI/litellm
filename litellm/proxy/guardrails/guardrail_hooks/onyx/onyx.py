# +-------------------------------------------------------------+
#
#           Use Onyx Guardrails for your LLM calls
#                   https://onyx.security/
#
# +-------------------------------------------------------------+
import os
import uuid
from typing import TYPE_CHECKING, Any, Literal, Optional, Type

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import GenericGuardrailAPIInputs, ModelResponse

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class OnyxGuardrail(CustomGuardrail):
    def __init__(
        self, api_base: Optional[str] = None, api_key: Optional[str] = None, **kwargs
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
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
        input_type: Literal["request", "response"],
        conversation_id: str,
    ) -> dict:
        """
        Call external Onyx Guard server for validation
        """
        response = await self.async_handler.post(
            f"{self.api_base}/guard/evaluate/v1/{self.api_key}/litellm",
            json={
                "payload": payload,
                "input_type": input_type,
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
            verbose_proxy_logger.warning(
                f"Request blocked by Onyx Guard. Violations: {detection_message}."
            )
            raise HTTPException(
                status_code=400,
                detail=f"Request blocked by Onyx Guard. Violations: {detection_message}.",
            )
        return result

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:

        conversation_id = (
            logging_obj.litellm_call_id if logging_obj else str(uuid.uuid4())
        )

        verbose_proxy_logger.info(
            "Running Onyx Guard apply_guardrail hook",
            extra={"conversation_id": conversation_id, "input_type": input_type},
        )
        payload = {}
        if input_type == "request":
            payload = request_data.get("proxy_server_request", {})
        else:
            try:
                response = ModelResponse(**request_data)
                parsed = response.json()
                payload = parsed.get("response", {})
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Error in converting request_data to ModelResponse: {str(e)}",
                    extra={
                        "conversation_id": conversation_id,
                        "input_type": input_type,
                    },
                )
                payload = request_data

        try:
            await self._validate_with_guard_server(payload, input_type, conversation_id)
            return inputs
        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in apply_guardrail guard: {str(e)}",
                extra={"conversation_id": conversation_id, "input_type": input_type},
            )
            return inputs

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.onyx import (
            OnyxGuardrailConfigModel,
        )

        return OnyxGuardrailConfigModel
