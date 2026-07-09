from typing import TYPE_CHECKING, Literal, Optional, Type

import httpx
from litellm.integrations.custom_guardrail import log_guardrail_information
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import GenericGuardrailAPIInputs
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

GUARDRAIL_NAME = "thirdlaw"


class ThirdlawGuardrailMissingConfig(ValueError):
    pass


class ThirdlawGuardrail(GenericGuardrailAPI):
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        additional_headers: Optional[str] = None,
        guardrail_timeout: Optional[int] = 60,
        streaming_end_of_stream_only: bool = True,
        streaming_sampling_rate: int = 5,
        **kwargs,
    ):
        resolved_base = api_base or get_secret_str("THIRDLAW_API_BASE")
        if not resolved_base:
            raise ThirdlawGuardrailMissingConfig(
                "ThirdLaw api_base is required. Set api_base in the guardrail "
                "config or the THIRDLAW_API_BASE environment variable."
            )
        resolved_key = api_key or get_secret_str("THIRDLAW_API_KEY")
        thirdlaw_headers = []
        if additional_headers:
            thirdlaw_headers.extend(
                [h.strip() for h in additional_headers.split(",") if h.strip()]
            )
        existing = kwargs.get("extra_headers") or []
        kwargs["extra_headers"] = thirdlaw_headers + [
            h for h in existing if h not in thirdlaw_headers
        ]
        self.guardrail_timeout = httpx.Timeout(timeout=guardrail_timeout, connect=5.0)
        self.streaming_end_of_stream_only = streaming_end_of_stream_only
        self.streaming_sampling_rate = streaming_sampling_rate
        super().__init__(
            api_base=resolved_base,
            api_key=resolved_key,
            **kwargs,
        )
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
            params={"timeout": self.guardrail_timeout},
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        inputs["structured_messages"] = (
            inputs.get("structured_messages", request_data.get("messages", [])) or []
        )
        return await super().apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type=input_type,
            logging_obj=logging_obj,
        )

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.thirdlaw import (
            ThirdlawGuardrailConfigModel,
        )

        return ThirdlawGuardrailConfigModel
