# +-------------------------------------------------------------+
#
#           Use GuardrailsAI for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you for using Litellm! - Krrish & Ishaan

import json
from typing import Optional, TypedDict

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_content_from_model_response,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks


class GuardrailsAIResponse(TypedDict):
    callId: str
    rawLlmOutput: str
    validatedOutput: str
    validationPassed: bool


class GuardrailsAI(CustomGuardrail):
    def __init__(
        self,
        guard_name: str,
        api_base: Optional[str] = None,
        **kwargs,
    ):
        if guard_name is None:
            raise Exception(
                "GuardrailsAIException - Please pass the Guardrails AI guard name via 'litellm_params::guard_name'"
            )
        # store kwargs as optional_params
        self.guardrails_ai_api_base = api_base or "http://0.0.0.0:8000"
        self.guardrails_ai_guard_name = guard_name
        self.optional_params = kwargs
        supported_event_hooks = [GuardrailEventHooks.post_call]
        super().__init__(supported_event_hooks=supported_event_hooks, **kwargs)

    async def make_guardrails_ai_api_request(self, llm_output: str, request_data: dict):
        from httpx import URL

        data = {
            "llmOutput": llm_output,
            **self.get_guardrail_dynamic_request_body_params(request_data=request_data),
        }
        _json_data = json.dumps(data)
        response = await litellm.module_level_aclient.post(
            url=str(
                URL(self.guardrails_ai_api_base).join(
                    f"guards/{self.guardrails_ai_guard_name}/validate"
                )
            ),
            data=_json_data,
            headers={
                "Content-Type": "application/json",
            },
        )
        verbose_proxy_logger.debug("guardrails_ai response: %s", response)
        _json_response = GuardrailsAIResponse(**response.json())  # type: ignore
        if _json_response.get("validationPassed") is False:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "guardrails_ai_response": _json_response,
                },
            )
        return _json_response

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs on response from LLM API call

        It can be used to reject a response
        """
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        if not isinstance(response, litellm.ModelResponse):
            return

        response_str: str = get_content_from_model_response(response)
        if response_str is not None and len(response_str) > 0:
            await self.make_guardrails_ai_api_request(
                llm_output=response_str, request_data=data
            )

            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )

        return
