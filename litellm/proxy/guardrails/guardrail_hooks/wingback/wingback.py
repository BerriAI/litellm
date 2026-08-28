# +-------------------------------------------------------------+
#
#           Use Wingback for your LLM calls
#                   https://wingback.ai/
#
# +-------------------------------------------------------------+

import os
from typing import TYPE_CHECKING, Any, Final, Literal

from litellm.integrations.custom_guardrail import log_guardrail_information
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.types.proxy.guardrails.guardrail_hooks.wingback import (
    WingbackGuardrailConfigModel,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME: Final = "wingback"
DEFAULT_WINGBACK_API_BASE: Final = "https://api.wingback.ai/connectors"


class WingbackGuardrail(GenericGuardrailAPI):
    """Wingback runtime security via the LiteLLM Generic Guardrail API contract."""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        wingback_app_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_api_base: Final = api_base or os.environ.get("WINGBACK_API_BASE") or DEFAULT_WINGBACK_API_BASE
        resolved_api_key: Final = api_key or os.environ.get("WINGBACK_INTEGRATION_API_KEY")

        existing_params = kwargs.pop("additional_provider_specific_params", None) or {}
        additional_params: Final = (
            {**existing_params, "wingback_app_id": wingback_app_id}
            if wingback_app_id and "wingback_app_id" not in existing_params
            else existing_params
        )

        kwargs.setdefault("guardrail_name", GUARDRAIL_NAME)
        kwargs.setdefault("unreachable_fallback", "fail_closed")

        super().__init__(
            api_base=resolved_api_base,
            api_key=resolved_api_key,
            additional_provider_specific_params=additional_params,
            **kwargs,
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply Wingback to the given inputs.

        NOTE: This override must live on this class so LiteLLM unified guardrail
        routing detects apply_guardrail in type(callback).__dict__.
        """
        return await super().apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type=input_type,
            logging_obj=logging_obj,
        )

    @classmethod
    def get_config_model(cls) -> type[WingbackGuardrailConfigModel] | None:
        return WingbackGuardrailConfigModel
