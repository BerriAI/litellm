# +-------------------------------------------------------------+
#
#           Use GuardrailsAI for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you for using Litellm! - Krrish & Ishaan

import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
)

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
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class GuardrailsAIResponse(TypedDict):
    callId: str
    rawLlmOutput: str
    validatedOutput: str
    validationPassed: bool


class InferenceData(TypedDict):
    name: str
    shape: List[int]
    data: List
    datatype: str


class GuardrailsAIResponsePreCall(TypedDict):
    modelname: str
    modelversion: str
    outputs: List[InferenceData]


class GuardrailsAI(CustomGuardrail):
    def __init__(
        self,
        guard_name: str,
        api_base: Optional[str] = None,
        guardrails_ai_api_input_format: Literal["inputs", "llmOutput"] = "llmOutput",
        **kwargs,
    ):
        if guard_name is None:
            raise Exception(
                "GuardrailsAIException - Please pass the Guardrails AI guard name via 'litellm_params::guard_name'"
            )
        # store kwargs as optional_params
        self.guardrails_ai_api_base = (
            api_base or os.getenv("GUARDRAILS_AI_API_BASE") or "http://0.0.0.0:8000"
        )
        self.guardrails_ai_guard_name = guard_name
        self.optional_params = kwargs
        self.guardrails_ai_api_input_format = guardrails_ai_api_input_format
        supported_event_hooks = [
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.logging_only,
        ]
        super().__init__(supported_event_hooks=supported_event_hooks, **kwargs)

    async def make_guardrails_ai_api_request(
        self, llm_output: str, request_data: dict
    ) -> GuardrailsAIResponse:
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

    async def make_guardrails_ai_api_request_pre_call_request(
        self, text_input: str, request_data: dict
    ) -> str:
        from httpx import URL

        # This branch of code does not work with current version of GuardrailsAI API (as of July 2025), and it is unclear if it ever worked.
        # Use guardrails_ai_api_input_format: "llmOutput" config line for all guardrails (which is the default anyway)
        # We can still use the "pre_call" mode to validate the inputs even if the API input format is technicallt "llmOutput"

        data = {
            "inputs": [
                {
                    "name": "text",
                    "shape": [1],
                    "data": [text_input],
                    "datatype": "BYTES",  # not sure what this should be, but Guardrail's response sets BYTES for text response - https://github.com/guardrails-ai/detect_pii/blob/e4719a95a26f6caacb78d46ebb4768317032bee5/app.py#L40C31-L40C36
                }
            ]
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
        if response.status_code == 400:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated guardrail policy",
                    "guardrails_ai_response": response.json(),
                },
            )

        _json_response = GuardrailsAIResponsePreCall(**response.json())  # type: ignore
        response = _json_response.get("outputs", [])[0].get("data", [])[0]
        return response

    async def process_input(self, data: dict, call_type: str) -> dict:
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
            set_last_user_message,
        )

        # Only process completion-related call types
        if call_type not in ["completion", "acompletion"]:
            return data

        if "messages" not in data:  # invalid request
            return data

        text = get_last_user_message(data["messages"])
        if text is None:
            return data
        if self.guardrails_ai_api_input_format == "inputs":
            updated_text = await self.make_guardrails_ai_api_request_pre_call_request(
                text_input=text, request_data=data
            )
        else:
            _result = await self.make_guardrails_ai_api_request(
                llm_output=text, request_data=data
            )
            updated_text = (
                _result.get("validatedOutput") or _result.get("rawLlmOutput") or text
            )
        data["messages"] = set_last_user_message(data["messages"], updated_text)

        return data

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: litellm.DualCache,
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
            "mcp_call",
        ],
    ) -> Optional[
        Union[Exception, str, dict]
    ]:  # raise exception if invalid, return a str for the user to receive - if rejected, or return a modified dictionary for passing into litellm
        return await self.process_input(data=data, call_type=call_type)

    async def async_logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        if call_type == "acompletion" or call_type == "completion":
            kwargs = await self.process_input(data=kwargs, call_type=call_type)

        return kwargs, result

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
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

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

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.guardrails_ai import (
            GuardrailsAIGuardrailConfigModel,
        )

        return GuardrailsAIGuardrailConfigModel
