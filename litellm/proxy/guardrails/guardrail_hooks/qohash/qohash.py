import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Type

from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
    GenericGuardrailAPI,
)
from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
    QohashGuardrailConfigModel,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_NAME = "qohash_qaigs"


class QohashGuardrail(GenericGuardrailAPI):
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        api_base = api_base or os.environ.get("QAIGS_API_BASE", "http://qaigs:8800")
        api_key = api_key or os.environ.get("QAIGS_API_KEY")

        kwargs["guardrail_name"] = kwargs.get("guardrail_name", GUARDRAIL_NAME)

        super().__init__(
            api_base=api_base,
            api_key=api_key,
            **kwargs,
        )

    def get_guardrail_dynamic_request_body_params(self, request_data: dict) -> Dict[str, Any]:
        """
        Extract extra_body params from the request guardrails list, without requiring
        a LiteLLM premium license. QAIGS-specific params (e.g. identifiers for advanced
        mode) are not a LiteLLM premium feature.

        The parent passes request_data["body"] here, but the proxy stores the original
        request fields (including "guardrails") directly on request_data — not under
        request_data["body"]. We check both so this works regardless of call path.
        """
        requested_guardrails = self.get_guardrail_from_metadata(request_data)
        for guardrail in requested_guardrails:
            if isinstance(guardrail, dict) and self.guardrail_name in guardrail:
                return guardrail[self.guardrail_name].get("extra_body", {})
        return {}

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply the Qohash QAIGS guardrail.

        Overrides GenericGuardrailAPI.apply_guardrail to make per-request extra_body params
        (e.g. QAIGS identifiers for advanced mode) visible to get_guardrail_dynamic_request_body_params.

        The parent extracts dynamic params from request_data["body"], but the proxy stores
        the original request fields (including "guardrails") directly on request_data. We
        inject them into request_data["body"] so our get_guardrail_dynamic_request_body_params
        override can find them when the parent calls it.

        This override is also necessary for LiteLLM's detection logic (proxy/utils.py):
            use_unified = "apply_guardrail" in type(callback).__dict__
        Without it the method would only exist on the parent class and not be detected.
        """
        # The proxy moves "guardrails" from the request body into metadata before calling
        # apply_guardrail (see move_guardrails_to_metadata in litellm_pre_call_utils.py).
        # get_guardrail_from_metadata knows to look in both data["guardrails"] and
        # data["metadata"]["guardrails"] / data["litellm_metadata"]["guardrails"].
        # We inject them into request_data["body"] so the parent's call to
        # get_guardrail_dynamic_request_body_params(request_body) finds them.
        guardrails = self.get_guardrail_from_metadata(request_data)
        if guardrails:
            request_data = {
                **request_data,
                "body": {**(request_data.get("body") or {}), "guardrails": guardrails},
            }

        return await super().apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type=input_type,
            logging_obj=logging_obj,
        )

    @classmethod
    def get_config_model(cls) -> Optional[Type[QohashGuardrailConfigModel]]:
        """
        Returns the config model for QAIGS.
        This is used to render the config form in the UI.
        """
        return QohashGuardrailConfigModel
