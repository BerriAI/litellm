"""
Unified Guardrail, leveraging LiteLLM's /applyGuardrail endpoint

1. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_pre_call_hook
2. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_success_hook
3. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_streaming_iterator_hook
"""

from typing import Any, AsyncGenerator, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.cost_calculator import _infer_call_type
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms import load_guardrail_translation_mappings
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, CallTypesLiteral, ModelResponseStream

GUARDRAIL_NAME = "unified_llm_guardrails"
endpoint_guardrail_translation_mappings = None


class UnifiedLLMGuardrails(CustomLogger):
    def __init__(
        self,
        **kwargs,
    ):

        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

        verbose_proxy_logger.debug(
            "UnifiedLLMGuardrails initialized with optional_params: %s",
            self.optional_params,
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """
        global endpoint_guardrail_translation_mappings
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        verbose_proxy_logger.debug("Running UnifiedLLMGuardrails pre-call hook")

        guardrail_to_apply: CustomGuardrail = data.pop("guardrail_to_apply", None)
        if guardrail_to_apply is None:
            return data

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if (
            guardrail_to_apply.should_run_guardrail(data=data, event_type=event_type)
            is not True
        ):
            verbose_proxy_logger.debug(
                "UnifiedLLMGuardrails: Pre-call scanning disabled for %s",
                guardrail_to_apply.guardrail_name,
            )
            return data

        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = (
                load_guardrail_translation_mappings()
            )
        if CallTypes(call_type) not in endpoint_guardrail_translation_mappings:
            return data

        endpoint_translation = endpoint_guardrail_translation_mappings[
            CallTypes(call_type)
        ]()

        data = await endpoint_translation.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail_to_apply,
        )

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=guardrail_to_apply.guardrail_name
        )
        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ) -> Any:
        """
        Runs on response from LLM API call

        It can be used to reject a response

        Uses Enkrypt AI guardrails to check the response for policy violations, PII, and injection attacks
        """
        global endpoint_guardrail_translation_mappings
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        guardrail_to_apply: CustomGuardrail = data.pop("guardrail_to_apply", None)
        if guardrail_to_apply is None:
            return

        if (
            guardrail_to_apply.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return

        verbose_proxy_logger.debug(
            "async_post_call_success_hook response: %s", response
        )

        call_type = _infer_call_type(call_type=None, completion_response=response)
        if call_type is None:
            return response

        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = (
                load_guardrail_translation_mappings()
            )

        if CallTypes(call_type) not in endpoint_guardrail_translation_mappings:
            return response

        endpoint_translation = endpoint_guardrail_translation_mappings[
            CallTypes(call_type)
        ]()

        response = await endpoint_translation.process_output_response(
            response=response,  # type: ignore
            guardrail_to_apply=guardrail_to_apply,
        )
        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=guardrail_to_apply.guardrail_name
        )
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Passes the entire stream to the guardrail

        This is useful for guardrails that need to see the entire response, such as PII masking.

        See Aim guardrail implementation for an example - https://github.com/BerriAI/litellm/blob/d0e022cfacb8e9ebc5409bb652059b6fd97b45c0/litellm/proxy/guardrails/guardrail_hooks/aim.py#L168

        Triggered by mode: 'post_call'
        """
        async for item in response:
            yield item
