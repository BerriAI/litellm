# +-------------------------------------------------------------+
#
#           Use GuardrailsAI for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you for using Litellm! - Krrish & Ishaan

import json
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.prompt_templates.common_utils import (
    convert_openai_message_to_only_content_messages,
    get_content_from_model_response,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues


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

    async def make_guardrails_ai_api_request(self, llm_output: str):
        from httpx import URL

        data = {"llmOutput": llm_output}
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
            await self.make_guardrails_ai_api_request(llm_output=response_str)

            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )

        return
