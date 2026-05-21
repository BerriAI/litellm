"""
Silmaril Firewall LiteLLM guardrail integration.
"""

import os
from typing import TYPE_CHECKING, Literal, Optional, Type

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import log_guardrail_information
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.types.proxy.guardrails.guardrail_hooks.silmaril import (
    SilmarilGuardrailConfigModel,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME = "silmaril"
LITELLM_GUARDRAIL_PATH = "/beta/litellm_basic_guardrail_api"


def get_silmaril_api_base(api_base: Optional[str]) -> str:
    resolved_api_base = api_base or os.environ.get("SILMARIL_GUARDRAIL_URL")

    if not resolved_api_base:
        raise ValueError(
            "api_base is required for Silmaril Firewall. Set "
            "SILMARIL_GUARDRAIL_URL to the full "
            "/beta/litellm_basic_guardrail_api endpoint or pass api_base in "
            "litellm_params."
        )

    resolved_api_base = resolved_api_base.rstrip("/")
    if not resolved_api_base.endswith(LITELLM_GUARDRAIL_PATH):
        raise ValueError(
            "Silmaril Firewall api_base must be the full /beta/litellm_basic_guardrail_api endpoint."
        )

    return resolved_api_base


class SilmarilGuardrail(GenericGuardrailAPI):
    def __init__(
        self,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        kwargs["guardrail_name"] = kwargs.get("guardrail_name") or GUARDRAIL_NAME
        kwargs["unreachable_fallback"] = (
            kwargs.get("unreachable_fallback") or "fail_open"
        )

        super().__init__(
            api_base=get_silmaril_api_base(api_base),
            **kwargs,
        )

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply Silmaril Firewall to the given inputs.

        This override must be defined directly on the class so LiteLLM's
        unified guardrail routing sees Silmaril as a first-class provider.
        """
        try:
            return await super().apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=logging_obj,
            )
        except GuardrailRaisedException as e:
            raise GuardrailRaisedException(
                guardrail_name=getattr(self, "guardrail_name", None) or GUARDRAIL_NAME,
                message=e.message,
                should_wrap_with_default_message=False,
            ) from e

    @classmethod
    def get_config_model(cls) -> Optional[Type[SilmarilGuardrailConfigModel]]:
        return SilmarilGuardrailConfigModel
