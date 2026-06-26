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
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.proxy.guardrails.guardrail_hooks.generic_guardrail_api import (
    GENERIC_GUARDRAIL_ACTION_BLOCKED,
    GENERIC_GUARDRAIL_ACTION_GUARDRAIL_INTERVENED,
    GENERIC_GUARDRAIL_ACTION_NONE,
    GENERIC_GUARDRAIL_ACTION_WAIT,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    CallTypes,
    CallTypesLiteral,
    Delta,
    GenericGuardrailAPIInputs,
    ModelResponseStream,
    StreamingChoices,
)

# Call types that use NDJSON streaming (A2A); guardrail HTTPException is emitted as in-stream error
A2A_CALL_TYPES = (CallTypes.asend_message, CallTypes.send_message)

GUARDRAIL_NAME = "unified_llm_guardrails"


def _supports_action_protocol(guardrail_to_apply: Any) -> bool:
    """
    True iff the guardrail instance advertises support for the streaming
    action protocol.

    Auto-detected by looking for a callable `apply_guardrail_action` method
    on the guardrail instance. For Python-class-backed guardrails (subclass
    `CustomGuardrail` and override the method), presence on the class is the
    truth: the method *is* the implementation.

    For HTTP-service-backed wrappers like `GenericGuardrailAPI`, the wrapper
    class always *could* call apply_guardrail_action over the wire — but
    the protocol's actual semantics (handling `is_final`, returning WAIT)
    live on the third-party service. Such wrappers must hide the method on
    instances whose backing service doesn't support it (typically by
    setting `self.apply_guardrail_action = None` when an operator opt-in
    flag is off), so this auto-detection reflects the service's
    capabilities rather than the wrapper's.

    Guardrails for which this returns False receive the historical
    moderation-mode iterator hook (observe-only); guardrails for which it
    returns True drive the wait/pass/block/modify state machine.
    """
    return callable(getattr(guardrail_to_apply, "apply_guardrail_action", None))


def _combine_streaming_text(chunks: List[Any]) -> str:
    """Concatenate delta.content across all chunks for choice index 0."""
    parts: List[str] = []
    for chunk in chunks:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if content:
            parts.append(content)
    return "".join(parts)


def _build_delta_chunk(
    template: Any,
    content: str,
) -> ModelResponseStream:
    """Build a content-only delta chunk modelled on `template`'s id/model/created."""
    return ModelResponseStream(
        id=getattr(template, "id", None),
        created=getattr(template, "created", None),
        model=getattr(template, "model", None),
        object=getattr(template, "object", "chat.completion.chunk"),
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=content, role="assistant"),
                finish_reason=None,
            )
        ],
    )


def _build_terminal_chunk(
    template: Any,
    finish_reason: str = "stop",
) -> ModelResponseStream:
    """Build an empty-content chunk carrying the stream's finish_reason."""
    return ModelResponseStream(
        id=getattr(template, "id", None),
        created=getattr(template, "created", None),
        model=getattr(template, "model", None),
        object=getattr(template, "object", "chat.completion.chunk"),
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(),
                finish_reason=finish_reason,
            )
        ],
    )


def _last_finish_reason(chunks: List[Any], default: str = "stop") -> str:
    """Pull the finish_reason off the most recent chunk that has one."""
    for chunk in reversed(chunks):
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        fr = getattr(choices[0], "finish_reason", None)
        if fr:
            return fr
    return default


def _chunk_tool_call_deltas(chunk: Any) -> List[Any]:
    """Return tool_call deltas (if any) for choice 0 of `chunk`, else empty list."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return []
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return []
    tcs = getattr(delta, "tool_calls", None)
    if not tcs:
        return []
    return list(tcs)


def _get_attr_or_key(obj: Any, key: str) -> Any:
    """Read `key` off `obj` whether it's a dict (subscript) or a model (attribute)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _accumulate_tool_calls(chunks: List[Any]) -> List[dict]:
    """
    Reduce all tool_call deltas across chunks into per-index accumulated state.

    Returns a list of {id, type, function: {name, arguments}} dicts (one per
    distinct tool_call index seen), in index order. `arguments` is the
    concatenation of all argument deltas seen for that index.

    Used to surface the in-progress tool calls to a guardrail in streaming
    action mode so the guardrail can inspect/block based on the args. Note
    that mid-stream the args JSON may not yet parse; guardrails that need
    a complete payload should return WAIT until they can parse it.
    """
    by_index: dict[int, dict] = {}
    _g = _get_attr_or_key
    for chunk in chunks:
        for tc in _chunk_tool_call_deltas(chunk):
            idx = getattr(tc, "index", None)
            if idx is None and isinstance(tc, dict):
                idx = tc.get("index")
            if idx is None:
                continue
            slot = by_index.setdefault(
                idx,
                {"id": None, "type": None, "function": {"name": None, "arguments": ""}},
            )

            if slot["id"] is None:
                slot["id"] = _g(tc, "id")
            if slot["type"] is None:
                slot["type"] = _g(tc, "type")
            fn = _g(tc, "function")
            if fn is not None:
                if slot["function"]["name"] is None:
                    name = _g(fn, "name")
                    if name:
                        slot["function"]["name"] = name
                args = _g(fn, "arguments")
                if args:
                    slot["function"]["arguments"] += args
    return [by_index[i] for i in sorted(by_index.keys())]


def _replay_tool_call_chunks(
    chunks: List[Any],
    start_idx: int,
    end_idx: int,
) -> List[ModelResponseStream]:
    """
    Build emit-ready chunks for tool_call deltas in chunks[start_idx:end_idx].

    Each output chunk carries only the tool_call portion of the original (no
    content, no finish_reason). Chunks with no tool_call deltas in the slice
    are skipped. Used in action mode so tool_call deltas reach the client
    after the guardrail clears them, even though text content is rebuilt
    from the cursor.
    """
    out: List[ModelResponseStream] = []
    for chunk in chunks[start_idx:end_idx]:
        tcs = _chunk_tool_call_deltas(chunk)
        if not tcs:
            continue
        out.append(
            ModelResponseStream(
                id=getattr(chunk, "id", None),
                created=getattr(chunk, "created", None),
                model=getattr(chunk, "model", None),
                object=getattr(chunk, "object", "chat.completion.chunk"),
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(tool_calls=tcs),
                        finish_reason=None,
                    )
                ],
            )
        )
    return out


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


def _ensure_litellm_metadata(data: dict, user_api_key_dict: UserAPIKeyAuth) -> None:
    """Populate data['litellm_metadata'] from user_api_key_dict if absent."""
    if "litellm_metadata" not in data:
        from litellm.llms.base_llm.guardrail_translation.base_translation import (
            BaseTranslation,
        )

        user_metadata = BaseTranslation.transform_user_api_key_dict_to_metadata(
            user_api_key_dict
        )
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

        _ensure_litellm_metadata(data, user_api_key_dict)

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

        # Fallback: resolve call_type from logging_obj for pass-through endpoints
        if call_type is None:
            litellm_logging_obj = data.get("litellm_logging_obj")
            logging_call_type = (
                getattr(litellm_logging_obj, "call_type", None)
                if litellm_logging_obj is not None
                else None
            )
            if logging_call_type in (
                CallTypes.pass_through.value,
                CallTypes.allm_passthrough_route.value,
            ):
                call_type = logging_call_type

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
            request_data=data,
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
    ) -> AsyncGenerator[Any, None]:
        """
        Passes the entire stream to the guardrail

        This is useful for guardrails that need to see the entire response, such as PII masking.

        See Aim guardrail implementation for an example - https://github.com/BerriAI/litellm/blob/d0e022cfacb8e9ebc5409bb652059b6fd97b45c0/litellm/proxy/guardrails/guardrail_hooks/aim.py#L168

        Triggered by mode: 'post_call'

        Supports sampling_rate parameter to control how often chunks are processed.
        sampling_rate=1 means every chunk, sampling_rate=5 means every 5th chunk, etc.

        Two operating modes, auto-detected per-guardrail:
        - moderation (default): observe-only — chunks pass through to client
          unmodified; guardrail can block via raised exception. Selected when
          the guardrail does not implement apply_guardrail_action.
        - action: streaming action protocol (wait/pass/block/modify). Selected
          when the guardrail implements apply_guardrail_action.
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

        # Auto-detect dispatch: guardrails implementing apply_guardrail_action
        # opt into the streaming action protocol; everything else stays on
        # the historical moderation-mode iterator hook below.
        if _supports_action_protocol(guardrail_to_apply):
            async for chunk in self._action_mode_stream(
                response=response,
                guardrail_to_apply=guardrail_to_apply,
                request_data=request_data,
                user_api_key_dict=user_api_key_dict,
                sampling_rate=max(1, int(sampling_rate)),
                end_of_stream_only=bool(end_of_stream_only),
            ):
                yield chunk
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
                        request_data=request_data,
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
                    request_data=request_data,
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

    async def _call_action_guardrail(
        self,
        *,
        guardrail_to_apply: CustomGuardrail,
        accumulated_text: str,
        chunks: List[Any],
        is_final: bool,
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Optional[Any]:
        """
        Call apply_guardrail_action on the guardrail with the accumulated text.

        Returns the raw GenericGuardrailAPIResponse, or None when the guardrail
        returns no decision (transport fail-open, or this guardrail integration
        does not implement the action protocol). Callers treat None as
        "no modification, no block" (NONE-equivalent).
        """
        apply_action = getattr(guardrail_to_apply, "apply_guardrail_action", None)
        if apply_action is None:
            verbose_proxy_logger.warning(
                "UnifiedLLMGuardrails: iterator_hook_mode=action requires "
                "guardrail.apply_guardrail_action; guardrail=%s does not "
                "implement it. Falling back to passthrough.",
                getattr(guardrail_to_apply, "guardrail_name", "<unknown>"),
            )
            return None

        inputs: GenericGuardrailAPIInputs = {
            "texts": [accumulated_text],
            "is_final": is_final,
        }
        first_chunk = chunks[0] if chunks else None
        model = getattr(first_chunk, "model", None) if first_chunk else None
        if model:
            inputs["model"] = model

        # Surface accumulated tool_calls for inspect-and-block. Modify is not
        # supported in action mode — tool_call chunks pass through unchanged
        # on emit, regardless of guardrail response.
        accumulated_tool_calls = _accumulate_tool_calls(chunks)
        if accumulated_tool_calls:
            inputs["tool_calls"] = accumulated_tool_calls  # type: ignore[typeddict-item]

        return await apply_action(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
            logging_obj=request_data.get("litellm_logging_obj"),
        )

    async def _action_mode_stream(  # noqa: PLR0915
        self,
        *,
        response: Any,
        guardrail_to_apply: CustomGuardrail,
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        sampling_rate: int,
        end_of_stream_only: bool = False,
    ) -> AsyncGenerator[Any, None]:
        """
        Streaming action protocol state machine.

        Buffers upstream chunks, calls the guardrail at sample points (and
        on every chunk while in WAIT state), emits delta chunks past the
        cursor on NONE/GUARDRAIL_INTERVENED, terminates on BLOCKED, and
        always makes a final is_final=True call after upstream EOS.

        When `end_of_stream_only=True`, the per-sample mid-stream calls are
        skipped entirely: chunks accumulate without invoking the guardrail
        and without emitting anything to the client; a single is_final=True
        call decides the whole response. This preserves the operator's
        `streaming_end_of_stream_only` config flag for action-mode
        guardrails — without it, an action-mode guardrail would still be
        sampled mid-stream and could emit content past the cursor before
        the final inspection had a chance to BLOCK.
        """
        cursor = 0
        in_wait_state = False
        chunk_counter = 0
        all_chunks: List[Any] = []
        # Index into all_chunks of the next chunk whose tool_call deltas (if
        # any) have not yet been replayed to the client. Advances on every
        # NONE/GUARDRAIL_INTERVENED emit so tool_call deltas reach the client
        # in lockstep with text emission, never mid-WAIT.
        tool_calls_replayed_through = 0
        template_chunk: Optional[Any] = None
        guardrail_name = getattr(guardrail_to_apply, "guardrail_name", GUARDRAIL_NAME)

        async for item in response:
            chunk_counter += 1
            all_chunks.append(item)
            if template_chunk is None:
                template_chunk = item

            # end_of_stream_only forces buffer-all behavior: skip per-sample
            # guardrail calls and per-sample emissions; a single is_final=True
            # call below decides the whole response.
            if end_of_stream_only:
                continue

            sample_due = (chunk_counter % sampling_rate == 0) or in_wait_state
            if not sample_due:
                continue

            accumulated_text = _combine_streaming_text(all_chunks)
            decision = await self._call_action_guardrail(
                guardrail_to_apply=guardrail_to_apply,
                accumulated_text=accumulated_text,
                chunks=all_chunks,
                is_final=False,
                request_data=request_data,
                user_api_key_dict=user_api_key_dict,
            )

            new_text, action = self._resolve_action_decision(
                decision=decision,
                accumulated_text=accumulated_text,
            )

            if action == GENERIC_GUARDRAIL_ACTION_BLOCKED:
                error_message = (
                    decision.blocked_reason if decision is not None else None
                ) or "Content violates policy"
                verbose_proxy_logger.warning(
                    "UnifiedLLMGuardrails action mode: BLOCKED mid-stream by %s: %s",
                    guardrail_name,
                    error_message,
                )
                raise GuardrailRaisedException(
                    guardrail_name=guardrail_name,
                    message=error_message,
                    should_wrap_with_default_message=False,
                )

            if action == GENERIC_GUARDRAIL_ACTION_WAIT:
                in_wait_state = True
                continue

            # NONE or GUARDRAIL_INTERVENED — emit delta past cursor and
            # replay any buffered tool_call deltas accumulated since last emit.
            #
            # Edge case: NONE with `len(new_text) < cursor` happens when an
            # earlier call returned an *expanded* GUARDRAIL_INTERVENED text
            # and a subsequent call returns NONE on raw accumulated text
            # that's now shorter than what's already been emitted. The bytes
            # through `cursor` are committed; emitting raw shorter text past
            # cursor isn't possible (already covered) and raising would
            # terminate an otherwise healthy stream. Defer to the next call
            # instead — cursor and tool-call replay marker stay put, so a
            # later GUARDRAIL_INTERVENED can refine cleanly.
            if action == GENERIC_GUARDRAIL_ACTION_NONE and len(new_text) < cursor:
                verbose_proxy_logger.debug(
                    "UnifiedLLMGuardrails action mode: %s returned NONE "
                    "with len=%d < cursor=%d; deferring emission to next "
                    "call",
                    guardrail_name,
                    len(new_text),
                    cursor,
                )
                in_wait_state = False
                continue

            # GUARDRAIL_INTERVENED with shrink IS a real protocol violation:
            # the guardrail tried to retract its own modifications.
            self._validate_cursor_monotonic(
                new_text=new_text,
                cursor=cursor,
                guardrail_name=guardrail_name,
                is_final=False,
            )
            delta = new_text[cursor:]
            cursor = len(new_text)
            in_wait_state = False
            if delta and template_chunk is not None:
                yield _build_delta_chunk(template_chunk, delta)
            for tc_chunk in _replay_tool_call_chunks(
                all_chunks, tool_calls_replayed_through, len(all_chunks)
            ):
                yield tc_chunk
            tool_calls_replayed_through = len(all_chunks)

        # Upstream exhausted — final call with is_final=True.
        accumulated_text = _combine_streaming_text(all_chunks)
        decision = await self._call_action_guardrail(
            guardrail_to_apply=guardrail_to_apply,
            accumulated_text=accumulated_text,
            chunks=all_chunks,
            is_final=True,
            request_data=request_data,
            user_api_key_dict=user_api_key_dict,
        )

        new_text, action = self._resolve_action_decision(
            decision=decision,
            accumulated_text=accumulated_text,
        )

        if action == GENERIC_GUARDRAIL_ACTION_BLOCKED:
            error_message = (
                decision.blocked_reason if decision is not None else None
            ) or "Content violates policy"
            verbose_proxy_logger.warning(
                "UnifiedLLMGuardrails action mode: BLOCKED at EOS by %s: %s",
                guardrail_name,
                error_message,
            )
            raise GuardrailRaisedException(
                guardrail_name=guardrail_name,
                message=error_message,
                should_wrap_with_default_message=False,
            )

        if action == GENERIC_GUARDRAIL_ACTION_WAIT:
            verbose_proxy_logger.error(
                "UnifiedLLMGuardrails action mode: %s returned WAIT at "
                "is_final=True (action protocol violation)",
                guardrail_name,
            )
            fallback = getattr(
                guardrail_to_apply, "unreachable_fallback", "fail_closed"
            )
            if fallback == "fail_open":
                new_text = accumulated_text
            else:
                raise GuardrailRaisedException(
                    guardrail_name=guardrail_name,
                    message=(
                        "guardrail returned WAIT at end of stream "
                        "(action protocol violation)"
                    ),
                    should_wrap_with_default_message=False,
                )

        # At EOS we can't retract bytes already emitted. If the guardrail
        # returns text shorter than what's been emitted, emit nothing more
        # — bytes through `cursor` were sent (with whatever substitutions
        # earlier calls applied) and falling back to `accumulated_text`
        # here would leak the raw, unmodified tail (e.g. unredacted PII)
        # for guardrails that rewrote mid-stream but returned short at EOS.
        # The terminal chunk below still carries the finish_reason.
        if len(new_text) < cursor:
            verbose_proxy_logger.error(
                "UnifiedLLMGuardrails action mode (EOS): %s returned text "
                "shorter than already-emitted (cursor=%d, new=%d) — "
                "stopping emission to avoid leaking unmodified tail",
                guardrail_name,
                cursor,
                len(new_text),
            )
        else:
            delta = new_text[cursor:]
            cursor = len(new_text)
            if delta and template_chunk is not None:
                yield _build_delta_chunk(template_chunk, delta)
        # Replay any tool_call deltas that arrived between the last sample-point
        # emit and EOS so the client sees the complete tool-call stream.
        for tc_chunk in _replay_tool_call_chunks(
            all_chunks, tool_calls_replayed_through, len(all_chunks)
        ):
            yield tc_chunk
        tool_calls_replayed_through = len(all_chunks)

        if template_chunk is not None:
            yield _build_terminal_chunk(
                template_chunk,
                finish_reason=_last_finish_reason(all_chunks),
            )

    @staticmethod
    def _resolve_action_decision(
        decision: Any,
        accumulated_text: str,
    ) -> tuple[str, str]:
        """Pull (new_text, action) from a GenericGuardrailAPIResponse, with NONE fallback."""
        if decision is None:
            return accumulated_text, GENERIC_GUARDRAIL_ACTION_NONE
        action = decision.action or GENERIC_GUARDRAIL_ACTION_NONE
        if action == GENERIC_GUARDRAIL_ACTION_GUARDRAIL_INTERVENED and decision.texts:
            return decision.texts[0], action
        if action not in (
            GENERIC_GUARDRAIL_ACTION_NONE,
            GENERIC_GUARDRAIL_ACTION_BLOCKED,
            GENERIC_GUARDRAIL_ACTION_GUARDRAIL_INTERVENED,
            GENERIC_GUARDRAIL_ACTION_WAIT,
        ):
            verbose_proxy_logger.warning(
                "UnifiedLLMGuardrails action mode: unknown action=%r; "
                "treating as NONE",
                action,
            )
            return accumulated_text, GENERIC_GUARDRAIL_ACTION_NONE
        return accumulated_text, action

    @staticmethod
    def _validate_cursor_monotonic(
        new_text: str,
        cursor: int,
        guardrail_name: str,
        is_final: bool,
    ) -> None:
        if len(new_text) >= cursor:
            return
        msg = (
            f"guardrail attempted to retract already-emitted content "
            f"(cursor={cursor}, returned len={len(new_text)})"
        )
        verbose_proxy_logger.error(
            "UnifiedLLMGuardrails action mode: %s — %s (is_final=%s)",
            guardrail_name,
            msg,
            is_final,
        )
        raise GuardrailRaisedException(
            guardrail_name=guardrail_name,
            message=f"{msg} — action protocol violation",
            should_wrap_with_default_message=False,
        )
