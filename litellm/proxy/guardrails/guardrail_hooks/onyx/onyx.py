# +-------------------------------------------------------------+
#
#           Use Onyx Guardrails for your LLM calls
#                   https://onyx.security/
#
# +-------------------------------------------------------------+
import os
import uuid
from typing import TYPE_CHECKING, Any, Final, Literal, Optional

import httpx
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs, ModelResponse

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class OnyxGuardrail(CustomGuardrail):
    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = 10.0,
        **kwargs,
    ):
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        timeout = timeout or int(os.getenv("ONYX_TIMEOUT", 10.0))
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"timeout": httpx.Timeout(timeout=timeout, connect=5.0)},
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
        verbose_proxy_logger.info("OnyxGuard initialized with server: %s", self.api_base)

    async def _validate_with_guard_server(
        self,
        payload: Any,
        input_type: Literal["request", "response"],
        conversation_id: str,
    ) -> dict:
        """
        Call external Onyx Guard server for validation
        """
        response: Final = await self.async_handler.post(
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
        result: Final = response.json()
        if not result.get("allowed", True):
            detection_message = "Unknown violation"
            if "violated_rules" in result:
                detection_message = ", ".join(result["violated_rules"])
            verbose_proxy_logger.warning("Request blocked by Onyx Guard. Violations: %s.", detection_message)
            raise HTTPException(
                status_code=400,
                detail=f"Request blocked by Onyx Guard. Violations: {detection_message}.",
            )
        return result

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        conversation_id: Final = logging_obj.litellm_call_id if logging_obj else str(uuid.uuid4())

        verbose_proxy_logger.info(
            "Running Onyx Guard apply_guardrail hook",
            extra={"conversation_id": conversation_id, "input_type": input_type},
        )
        payload = {}
        if input_type == "request":
            payload = request_data.get("proxy_server_request", {})
        else:
            try:
                response: Final = ModelResponse(**request_data)
                parsed: Final = response.json()
                payload = parsed.get("response", {})
            except Exception as e:
                verbose_proxy_logger.error(
                    "Error in converting request_data to ModelResponse: %s",
                    e,
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
                "Error in apply_guardrail guard: %s",
                e,
                extra={"conversation_id": conversation_id, "input_type": input_type},
            )
            return inputs

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.onyx import (
            OnyxGuardrailConfigModel,
        )

        return OnyxGuardrailConfigModel
