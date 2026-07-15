"""
Unified Guardrail, leveraging LiteLLM's /applyGuardrail endpoint

1. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_pre_call_hook
2. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_success_hook
3. Implements a way to call /applyGuardrail endpoint for `/chat/completions` + `/v1/messages` requests on async_post_call_streaming_iterator_hook
"""

import copy
import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, List, Optional, Union

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

if TYPE_CHECKING:
    # Imported lazily at runtime (inside the streaming hook) to avoid a
    # module-level cyclic import with litellm.integrations.custom_guardrail.
    from litellm.integrations.custom_guardrail import ModifyResponseException

# Call types that use NDJSON streaming (A2A); guardrail HTTPException is emitted as in-stream error
A2A_CALL_TYPES = (CallTypes.asend_message, CallTypes.send_message)

GUARDRAIL_NAME = "unified_llm_guardrails"


def _get_a2a_request_id(responses_so_far: List[Any], request_data: dict) -> Optional[str]:
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


def _ensure_litellm_metadata(data: dict, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Populate data['litellm_metadata'] from user_api_key_dict if absent."""
    if "litellm_metadata" not in data:
        from litellm.llms.base_llm.guardrail_translation.base_translation import (
            BaseTranslation,
        )

        user_metadata = BaseTranslation.transform_user_api_key_dict_to_metadata(user_api_key_dict)
        if user_metadata:
            data["litellm_metadata"] = user_metadata


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

        if guardrail_to_apply.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                "UnifiedLLMGuardrails: Pre-call scanning disabled for %s",
                guardrail_to_apply.guardrail_name,
            )
            return data

        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = load_guardrail_translation_mappings()

        try:
            if CallTypes(call_type) not in endpoint_guardrail_translation_mappings:
                return data
        except ValueError:
            return data  # handle unmapped call types

        endpoint_translation = endpoint_guardrail_translation_mappings[CallTypes(call_type)]()

        _ensure_litellm_metadata(data, user_api_key_dict)

        data = await endpoint_translation.process_input_messages(
            data=data,
            guardrail_to_apply=guardrail_to_apply,
            litellm_logging_obj=data.get("litellm_logging_obj"),
        )

        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=guardrail_to_apply.guardrail_name)
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

        if guardrail_to_apply.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                "UnifiedLLMGuardrails: Pre-call scanning disabled for %s",
                guardrail_to_apply.guardrail_name,
            )
            return data

        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = load_guardrail_translation_mappings()
        if call_type is not None and CallTypes(call_type) not in endpoint_guardrail_translation_mappings:
            return data

        endpoint_translation = endpoint_guardrail_translation_mappings[CallTypes(call_type)]()

        _ensure_litellm_metadata(data, user_api_key_dict)

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

        # Local import avoids a module-level cyclic import with
        # litellm.integrations.custom_guardrail.
        from litellm.integrations.custom_guardrail import ModifyResponseException

        guardrail_to_apply: CustomGuardrail = data.pop("guardrail_to_apply", None)

        if guardrail_to_apply is None:
            return

        if guardrail_to_apply.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return

        verbose_proxy_logger.debug("async_post_call_success_hook response: %s", response)

        call_type: Optional[CallTypesLiteral] = None
        if user_api_key_dict.request_route is not None:
            call_types = get_call_types_for_route(user_api_key_dict.request_route)
            if call_types is not None and len(call_types) > 0:  # type: ignore
                call_type = call_types[0]  # type: ignore
        if call_type is None:
            call_type = _infer_call_type(call_type=None, completion_response=response)  # type: ignore

        # Fallback: resolve call_type from logging_obj for pass-through endpoints
        if call_type is None:
            litellm_logging_obj = data.get("litellm_logging_obj")
            logging_call_type = (
                getattr(litellm_logging_obj, "call_type", None) if litellm_logging_obj is not None else None
            )
            if logging_call_type in (
                CallTypes.pass_through.value,
                CallTypes.allm_passthrough_route.value,
            ):
                call_type = logging_call_type

        if call_type is None:
            return response

        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = load_guardrail_translation_mappings()

        if CallTypes(call_type) not in endpoint_guardrail_translation_mappings:
            return response

        endpoint_translation = endpoint_guardrail_translation_mappings[CallTypes(call_type)]()

        try:
            response = await endpoint_translation.process_output_response(
                response=response,  # type: ignore
                guardrail_to_apply=guardrail_to_apply,
                litellm_logging_obj=data.get("litellm_logging_obj"),
                user_api_key_dict=user_api_key_dict,
                request_data=data,
            )
        except ModifyResponseException as e:
            # The guardrail blocked the response. Attach the original LLM
            # response so the endpoint handler can report its real token usage
            # instead of discarding it (the block replaces the content, but the
            # upstream call already consumed those tokens).
            if e.original_response is None:
                e.original_response = response
            raise
        # Add guardrail to applied guardrails header
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=guardrail_to_apply.guardrail_name)

        return response

    async def _handle_streaming_block(
        self,
        exc: "ModifyResponseException",
        endpoint_translation: Any,
        stream_started: bool,
        responses_so_far: list[Any],
    ) -> AsyncGenerator[Any, None]:
        """
        Terminate a streamed response cleanly when a guardrail blocks it.

        Format-agnostic routing: delegates to the provider translation handler's
        ``build_block_sse_chunks`` (see ``BaseTranslation.build_block_sse_chunks``
        for the ``stream_started`` / ``responses_so_far`` contract). When the
        format has no safe terminator the handler returns None and we re-raise
        ``exc`` so the proxy can surface a clean error.
        """
        block_chunks = endpoint_translation.build_block_sse_chunks(
            exc, stream_started=stream_started, responses_so_far=responses_so_far
        )
        if block_chunks is None:
            raise exc
        for chunk in block_chunks:
            yield chunk

    async def async_post_call_streaming_iterator_hook(
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

        # Local import avoids a module-level cyclic import with
        # litellm.integrations.custom_guardrail.
        from litellm.integrations.custom_guardrail import ModifyResponseException

        guardrail_to_apply: CustomGuardrail = request_data.pop("guardrail_to_apply", None)

        # Get streaming configuration. Resolution order (later wins): default
        # < guardrail attribute < guardrail_config dict < this callback's
        # optional_params.
        def _streaming_flag(name: str, default: Any) -> Any:
            value = default
            if guardrail_to_apply is not None:
                value = getattr(guardrail_to_apply, name, value)
                config = getattr(guardrail_to_apply, "guardrail_config", {})
                if isinstance(config, dict):
                    value = config.get(name, value)
            return self.optional_params.get(name, value)

        sampling_rate = _streaming_flag("streaming_sampling_rate", 5)
        # Only apply the guardrail at end of stream (not per chunk).
        end_of_stream_only = _streaming_flag("streaming_end_of_stream_only", False)
        # Withhold every chunk until end-of-stream moderation passes, then
        # release the original chunks (clean) or only the block message
        # (blocked) -- moderating the whole response *before* any content
        # reaches the client. Only safe for allow/block guardrails: on
        # release the original chunks are replayed as-is, so a
        # content-rewriting guardrail (e.g. PII masking) would leak
        # unredacted content. Guarded below via mask_response_content.
        buffer_until_moderated = _streaming_flag("streaming_buffer_until_moderated", False)

        if (
            buffer_until_moderated
            and guardrail_to_apply is not None
            and getattr(guardrail_to_apply, "mask_response_content", False)
        ):
            verbose_proxy_logger.warning(
                "UnifiedLLMGuardrails: streaming_buffer_until_moderated is disabled for %s "
                "because mask_response_content=True -- buffered replay would release "
                "unredacted original chunks instead of the moderated output.",
                guardrail_to_apply.guardrail_name,
            )
            buffer_until_moderated = False

        # Buffering can only moderate the assembled response, so it always
        # defers to end-of-stream.
        if buffer_until_moderated:
            end_of_stream_only = True

        if guardrail_to_apply is None:
            async for item in response:
                yield item
            return

        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if guardrail_to_apply.should_run_guardrail(data=request_data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                "UnifiedLLMGuardrails: Post-call streaming scanning disabled for %s",
                guardrail_to_apply.guardrail_name,
            )
            async for item in response:
                yield item
            return

        # Initialize translation mappings if needed
        if endpoint_guardrail_translation_mappings is None:
            endpoint_guardrail_translation_mappings = load_guardrail_translation_mappings()

        # Infer call type from first chunk
        call_type = None
        chunk_counter = 0
        responses_so_far: List[Any] = []
        responses_yielded: list[Any] = []
        pending_end_of_stream_items: list[Any] = []
        # Whether any real response chunk has been forwarded to the client.
        # Drives how a block terminates the stream: continue the in-progress
        # message (True) vs emit a standalone block message (False, buffered).
        chunks_yielded = False

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
            if call_type is None or CallTypes(call_type) not in endpoint_guardrail_translation_mappings:
                yield item
                async for remaining_item in response:
                    yield remaining_item
                return

            # If end_of_stream_only mode, yield chunks without processing.
            # When buffering, withhold them instead -- they are released (or
            # replaced by the block message) only after end-of-stream
            # moderation runs below.
            if end_of_stream_only:
                if not buffer_until_moderated:
                    endpoint_translation = endpoint_guardrail_translation_mappings[CallTypes(call_type)]()
                    stream_has_ended = hasattr(
                        endpoint_translation, "_check_streaming_has_ended"
                    ) and endpoint_translation._check_streaming_has_ended(responses_so_far)
                    if pending_end_of_stream_items or stream_has_ended:
                        pending_end_of_stream_items.append(item)
                    else:
                        chunks_yielded = True
                        responses_yielded.append(item)
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

                endpoint_translation = endpoint_guardrail_translation_mappings[CallTypes(call_type)]()

                try:
                    await endpoint_translation.process_output_streaming_response(
                        responses_so_far=responses_so_far,
                        guardrail_to_apply=guardrail_to_apply,
                        litellm_logging_obj=request_data.get("litellm_logging_obj"),
                        user_api_key_dict=user_api_key_dict,
                        request_data=request_data,
                    )
                except ModifyResponseException as e:
                    if e.original_response is None:
                        e.original_response = responses_so_far
                    # Guardrail blocked the response mid-stream. Emit a clean
                    # terminating SSE sequence delivering the block message
                    # instead of letting the exception propagate into a bare
                    # `data: {"error": ...}` blob (which truncates the stream).
                    # Chunks have already been forwarded here, so the block
                    # continues the in-progress message (stream_started=True).
                    # The current chunk was appended to responses_so_far but not
                    # yet yielded, so exclude it: the continuation must reflect
                    # only what the client has actually received.
                    async for block_chunk in self._handle_streaming_block(
                        e,
                        endpoint_translation,
                        stream_started=chunks_yielded,
                        responses_so_far=responses_yielded,
                    ):
                        yield block_chunk
                    return
                except HTTPException as e:
                    # Response already started (we already yielded chunks); cannot send 400.
                    # For A2A (NDJSON), yield an in-stream JSON-RPC error so the client sees it.
                    if call_type is not None and CallTypes(call_type) in A2A_CALL_TYPES:
                        request_id = _get_a2a_request_id(responses_so_far, request_data)
                        detail = e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}
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
                                        "data": {k: v for k, v in detail.items() if k not in ("error", "message")},
                                    },
                                }
                            )
                            + "\n"
                        )
                        yield error_chunk
                        return
                    raise
                chunks_yielded = True
                responses_yielded.append(original_item)
                yield original_item
            else:
                chunks_yielded = True
                responses_yielded.append(item)
                yield item

        # Stream has ended - do final processing with all collected chunks
        if call_type is not None and CallTypes(call_type) in endpoint_guardrail_translation_mappings:
            verbose_proxy_logger.debug(
                "Processing final streaming response with all %s chunks for guardrail %s",
                len(responses_so_far),
                guardrail_to_apply.guardrail_name,
            )

            endpoint_translation = endpoint_guardrail_translation_mappings[CallTypes(call_type)]()

            # When buffering, snapshot the original chunks before moderation.
            # A shallow copy suffices: end-of-stream
            # process_output_streaming_response builds a separate assembled
            # response (it does not mutate the individual chunks in place), and
            # the chunks themselves are replayed verbatim -- so we only need to
            # preserve the list, not clone every chunk (deepcopy would double
            # peak memory for large responses).
            buffered_items = list(responses_so_far) if buffer_until_moderated else None

            try:
                await endpoint_translation.process_output_streaming_response(
                    responses_so_far=responses_so_far,
                    guardrail_to_apply=guardrail_to_apply,
                    litellm_logging_obj=request_data.get("litellm_logging_obj"),
                    user_api_key_dict=user_api_key_dict,
                    request_data=request_data,
                )
                # Moderation passed: release the withheld original chunks.
                if buffered_items is not None:
                    for buffered_item in buffered_items:
                        yield buffered_item
                for pending_item in pending_end_of_stream_items:
                    responses_yielded.append(pending_item)
                    yield pending_item
            except ModifyResponseException as e:
                if e.original_response is None:
                    e.original_response = responses_so_far
                # Block detected during end-of-stream processing. Emit a clean
                # terminating SSE sequence with the block message rather than
                # propagating into a bare error blob that truncates the stream.
                # The withheld original chunks are never released.
                async for block_chunk in self._handle_streaming_block(
                    e,
                    endpoint_translation,
                    stream_started=bool(responses_yielded),
                    responses_so_far=responses_yielded,
                ):
                    yield block_chunk
                return
            except HTTPException as e:
                if call_type is not None and CallTypes(call_type) in A2A_CALL_TYPES:
                    request_id = _get_a2a_request_id(responses_so_far, request_data)
                    detail = e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}
                    error_chunk = (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "error": {
                                    "code": -32603,
                                    "message": detail.get("error", detail.get("message", str(e.detail))),
                                    "data": {k: v for k, v in detail.items() if k not in ("error", "message")},
                                },
                            }
                        )
                        + "\n"
                    )
                    yield error_chunk
                else:
                    raise
