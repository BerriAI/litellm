"""
Unified Guardrail, leveraging LiteLLM's /applyGuardrail endpoint

1. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_pre_call_hook
2. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_success_hook
3. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_streaming_iterator_hook
"""

import copy
import json
from typing import Any, AsyncGenerator, List, Optional, Union

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.cost_calculator import _infer_call_type
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
from litellm.llms import load_guardrail_translation_mappings
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, CallTypesLiteral

# Call types that use NDJSON streaming (A2A); guardrail HTTPException is emitted as in-stream error
A2A_CALL_TYPES = (CallTypes.asend_message, CallTypes.send_message)

GUARDRAIL_NAME = "unified_llm_guardrails"


def _get_a2a_request_id(
    responses_so_far: List[Any], request_data: dict
) -> Optional[str]:
    """Get JSON-RPC request id from first A2A chunk or request body for in-stream error reporting."""
    for item in responses_so_far:
        if isinstance(item, dict) and "id" in item:
            return item.get("id")
        if isinstance(item, str):
            try:
                obj = json.loads(item.strip())
                if isinstance(obj, dict) and "id" in obj:
                    return obj.get("id")
            except (json.JSONDecodeError, TypeError):
                continue
    body = request_data.get("body") or request_data.get("data") or {}
    if isinstance(body, dict):
        return body.get("id")
    return None


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
        if call_type == CallTypes.call_mcp_tool.value:
            event_type = GuardrailEventHooks.pre_mcp_call

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

        try:
            if CallTypes(call_type) not in endpoint_guardrail_translation_mappings:
                return data
        except ValueError:
            return data  # handle unmapped call types

        endpoint_translation = endpoint_guardrail_translation_mappings[
            CallTypes(call_type)
        ]()

        data = await endpoint_translation.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=data.get("litellm_logging_obj"),
        )

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=guardrail_to_apply.guardrail_name
        )
        return data

    async def async_moderation_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, call_type: CallTypesLiteral
    ) -> Any:
        """
        Runs in parallel to LLM API call
        Runs on only Input

        This can NOT modify the input, only used to reject or accept a call before going to LLM API
        """
        global endpoint_guardrail_translation_mappings

        verbose_proxy_logger.debug("Running UnifiedLLMGuardrails moderation hook")

        guardrail_to_apply: CustomGuardrail = data.pop("guardrail_to_apply", None)
        if guardrail_to_apply is None:
            return data

        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if call_type == CallTypes.call_mcp_tool.value:
            event_type = GuardrailEventHooks.during_mcp_call

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
        if (
            call_type is not None
            and CallTypes(call_type) not in endpoint_guardrail_translation_mappings
        ):
            return data

        endpoint_translation = endpoint_guardrail_translation_mappings[
            CallTypes(call_type)
        ]()

        return await endpoint_translation.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=data.get("litellm_logging_obj"),
        )

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

        call_type: Optional[CallTypesLiteral] = None
        if user_api_key_dict.request_route is not None:
            call_types = get_call_types_for_route(user_api_key_dict.request_route)
            if call_types is not None and len(call_types) > 0:  # type: ignore
                call_type = call_types[0]  # type: ignore
        if call_type is None:
            call_type = _infer_call_type(call_type=None, completion_response=response)  # type: ignore

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
            litellm_logging_obj=data.get("litellm_logging_obj"),
            user_api_key_dict=user_api_key_dict,
        )
        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=guardrail_to_apply.guardrail_name
        )

        return response

    async def async_post_call_streaming_iterator_hook(  # noqa: PLR0915
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """
        Passes the entire stream to the guardrail

        This is useful for guardrails that need to see the entire response, such as PII masking.

        See Aim guardrail implementation for an example - https://github.com/BerriAI/litellm/blob/d0e022cfacb8e9ebc5409bb652059b6fd97b45c0/litellm/proxy/guardrails/guardrail_hooks/aim.py#L168

        Triggered by mode: 'post_call'

        Supports sampling_rate parameter to control how often chunks are processed.
        sampling_rate=1 means every chunk, sampling_rate=5 means every 5th chunk, etc.
        """

        global endpoint_guardrail_translation_mappings

        guardrail_to_apply: CustomGuardrail = request_data.pop(
            "guardrail_to_apply", None
        )

        # Get streaming configuration from guardrail or optional_params
        sampling_rate = 5
        end_of_stream_only = False  # If True, only apply guardrail at end of stream

        if guardrail_to_apply is not None:
            # Check direct attributes on guardrail first
            sampling_rate = getattr(
                guardrail_to_apply, "streaming_sampling_rate", sampling_rate
            )
            end_of_stream_only = getattr(
                guardrail_to_apply, "streaming_end_of_stream_only", end_of_stream_only
            )

            # Also check guardrail_config dict if present
            guardrail_config = getattr(guardrail_to_apply, "guardrail_config", {})
            if isinstance(guardrail_config, dict):
                sampling_rate = guardrail_config.get(
                    "streaming_sampling_rate", sampling_rate
                )
                end_of_stream_only = guardrail_config.get(
                    "streaming_end_of_stream_only", end_of_stream_only
                )

        # Also check optional_params as fallback
        sampling_rate = self.optional_params.get(
            "streaming_sampling_rate", sampling_rate
        )
        end_of_stream_only = self.optional_params.get(
            "streaming_end_of_stream_only", end_of_stream_only
        )

        if guardrail_to_apply is None:
            async for item in response:
                yield item
            return

        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if (
            guardrail_to_apply.should_run_guardrail(
                data=request_data, event_type=event_type
            )
            is not True
        ):
            verbose_proxy_logger.debug(
                "UnifiedLLMGuardrails: Post-call streaming scanning disabled for %s",
                guardrail_to_apply.guardrail_name,
            )
            async for item in response:
                yield item
            return

        # Initialize translation mappings if needed
        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = (
                load_guardrail_translation_mappings()
            )

        # Infer call type from first chunk
        call_type = None
        chunk_counter = 0
        responses_so_far: List[Any] = []

        async for item in response:
            chunk_counter += 1
            responses_so_far.append(item)

            # Infer call type from first chunk if not already done
            if call_type is None and user_api_key_dict.request_route is not None:
                call_types = get_call_types_for_route(user_api_key_dict.request_route)
                if call_types is not None:
                    call_type = call_types[0].value

            if call_type is None:
                call_type = _infer_call_type(call_type=None, completion_response=item)  # type: ignore

            # If call type not supported, just pass through all chunks
            if (
                call_type is None
                or CallTypes(call_type) not in endpoint_guardrail_translation_mappings
            ):
                yield item
                async for remaining_item in response:
                    yield remaining_item
                return

            # If end_of_stream_only mode, yield chunks without processing
            if end_of_stream_only:
                yield item
                continue

            # Process chunk based on sampling rate
            if chunk_counter % sampling_rate == 0:
                verbose_proxy_logger.debug(
                    "Processing streaming chunk %s (sampling_rate=%s) with guardrail %s",
                    chunk_counter,
                    sampling_rate,
                    guardrail_to_apply.guardrail_name,
                )

                # Deep-copy the current chunk before guardrail processing.
                # process_output_streaming_response modifies responses_so_far
                # in-place: it puts the combined guardrailed text in the first
                # chunk and clears all subsequent chunks to "". Without this
                # copy, yielding processed_items[-1] would yield an empty
                # string, permanently losing this chunk's content.
                original_item = copy.deepcopy(item)

                endpoint_translation = endpoint_guardrail_translation_mappings[
                    CallTypes(call_type)
                ]()

                try:
                    await endpoint_translation.process_output_streaming_response(
                        responses_so_far=responses_so_far,
                        guardrail_to_apply=guardrail_to_apply,
                        litellm_logging_obj=request_data.get("litellm_logging_obj"),
                        user_api_key_dict=user_api_key_dict,
                    )
                except HTTPException as e:
                    # Response already started (we already yielded chunks); cannot send 400.
                    # For A2A (NDJSON), yield an in-stream JSON-RPC error so the client sees it.
                    if call_type is not None and CallTypes(call_type) in A2A_CALL_TYPES:
                        request_id = _get_a2a_request_id(responses_so_far, request_data)
                        detail = (
                            e.detail
                            if isinstance(e.detail, dict)
                            else {"message": str(e.detail)}
                        )
                        error_chunk = (
                            json.dumps(
                                {
                                    "jsonrpc": "2.0",
                                    "id": request_id,
                                    "error": {
                                        "code": -32603,
                                        "message": detail.get(
                                            "error",
                                            detail.get("message", str(e.detail)),
                                        ),
                                        "data": {
                                            k: v
                                            for k, v in detail.items()
                                            if k not in ("error", "message")
                                        },
                                    },
                                }
                            )
                            + "\n"
                        )
                        yield error_chunk
                        return
                    raise
                yield original_item
            else:
                yield item

        # Stream has ended - do final processing with all collected chunks
        if (
            call_type is not None
            and CallTypes(call_type) in endpoint_guardrail_translation_mappings
        ):
            verbose_proxy_logger.debug(
                "Processing final streaming response with all %s chunks for guardrail %s",
                len(responses_so_far),
                guardrail_to_apply.guardrail_name,
            )

            endpoint_translation = endpoint_guardrail_translation_mappings[
                CallTypes(call_type)
            ]()

            try:
                await endpoint_translation.process_output_streaming_response(
                    responses_so_far=responses_so_far,
                    guardrail_to_apply=guardrail_to_apply,
                    litellm_logging_obj=request_data.get("litellm_logging_obj"),
                    user_api_key_dict=user_api_key_dict,
                )
            except HTTPException as e:
                if call_type is not None and CallTypes(call_type) in A2A_CALL_TYPES:
                    request_id = _get_a2a_request_id(responses_so_far, request_data)
                    detail = (
                        e.detail
                        if isinstance(e.detail, dict)
                        else {"message": str(e.detail)}
                    )
                    error_chunk = (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32603,
                                    "message": detail.get(
                                        "error", detail.get("message", str(e.detail))
                                    ),
                                    "data": {
                                        k: v
                                        for k, v in detail.items()
                                        if k not in ("error", "message")
                                    },
                                },
                            }
                        )
                        + "\n"
                    )
                    yield error_chunk
                else:
                    raise
