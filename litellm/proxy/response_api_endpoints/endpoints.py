import asyncio
import contextlib
import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    cast,
    get_args,
    runtime_checkable,
)

import fastapi
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from openai.types.responses import ResponseItemList
from openai.types.responses.response_create_params import ResponseInputParam
from pydantic import ConfigDict, TypeAdapter, ValidationError, with_config
from starlette.websockets import WebSocket, WebSocketDisconnect
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.anthropic_interface.exceptions.exception_mapping_utils import AnthropicExceptionMapping
from litellm.caching.dual_cache import DualCache
from litellm.constants import EMPTY_MAPPING, MANAGED_BACKGROUND_RESPONSE_METADATA_KEY, TERMINAL_RESPONSE_STATUSES
from litellm.integrations.custom_guardrail import ModifyResponseException
from litellm.litellm_core_utils.core_helpers import as_str_mapping, get_metadata_variable_name_from_kwargs
from litellm.litellm_core_utils.internal_call_metadata import is_background_response, requests_background
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.litellm_core_utils.tool_fields import TOOL_FIELDS, first_validation_error
from litellm.llms.openai.responses.guardrail_translation.handler import (
    OpenAIResponsesHandler,
    build_blocked_response,
)
from litellm.proxy._types import *
from litellm.proxy.auth.auth_utils import carries_client_side_overrides
from litellm.proxy.auth.user_api_key_auth import (
    UserAPIKeyAuth,
    user_api_key_auth,
    user_api_key_auth_websocket,
)
from litellm.proxy.common_request_processing import (
    ProxyBaseLLMRequestProcessing,
    cost_headers,
    create_response,
    resolve_counted_tool_fields,
)
from litellm.proxy.common_utils.error_body_call_id import error_body_call_id, with_call_id
from litellm.proxy.common_utils.http_parsing_utils import (  # noqa: F401  # legacy module exports
    _read_request_body,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _safe_set_request_parsed_body,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    read_request_body,
    safe_set_request_parsed_body,
)
from litellm.proxy.common_utils.openai_error_payload import (
    LITELLM_CALL_ID_HEADER,
    error_status_code,
    openai_error_type_for_status,
)
from litellm.proxy.common_utils.token_count_settings import count_source_fields
from litellm.proxy.hooks.proxy_track_cost_callback import ProxyDBLogger
from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity, decrypted_response_id
from litellm.proxy.litellm_pre_call_utils import (
    _CLIENT_PRICING_METADATA_FIELDS,  # pyright: ignore[reportPrivateUsage]  # the HTTP path's own strip list
    _UNTRUSTED_METADATA_CONTROL_FIELDS,  # pyright: ignore[reportPrivateUsage]  # the HTTP path's own strip list
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.batch_attribution import (
    request_tags_from_request_data,
)
from litellm.proxy.route_llm_request import raise_if_required_body_param_missing
from litellm.proxy.utils import count_request_input_tokens, handle_exception_on_proxy
from litellm.responses.litellm_completion_transformation.compaction_items import messages_since_newest_compaction
from litellm.responses.litellm_completion_transformation.count_request import MODEL_FILE_ID_MAPPING_FIELD
from litellm.responses.streaming_iterator import GuardedResponsesWsFrame, ResponsesWsFrameGuard, spawn_logging_task
from litellm.router import Router
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.openai import (
    REASONING_EFFORT,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.token_count import TokenCountSourceFields
from litellm.types.utils import ADDRESSED_RESPONSE_ID_FIELD, TokenCountResponse

if TYPE_CHECKING:
    from litellm.proxy.utils import ProxyLogging

router: Final = APIRouter()
_RESPONSES_WS_CONFIG_VALUE_ADAPTER: Final[TypeAdapter[object | None]] = TypeAdapter(object | None)
_DECODED_JSON: Final = TypeAdapter(object)
_PROXY_SERVER_REQUEST_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])

_ResponseDocSchemas: TypeAlias = dict[int | str, dict[str, object]]  # fastapi's responses kwarg

RESPONSES_API_RESPONSE_SCHEMAS: Final[_ResponseDocSchemas] = {200: {"model": ResponsesAPIResponse}}
RESPONSES_API_CREATE_RESPONSE_SCHEMAS: Final[_ResponseDocSchemas] = {
    200: {
        "model": ResponsesAPIResponse,
        "content": {
            "text/event-stream": {"schema": {"type": "string", "description": "Server sent events when stream=true"}}
        },
    }
}
DELETE_RESPONSE_SCHEMAS: Final[_ResponseDocSchemas] = {200: {"model": DeleteResponseResult}}
RESPONSE_ITEM_LIST_SCHEMAS: Final[_ResponseDocSchemas] = {200: {"model": ResponseItemList}}

_user_api_key_auth_dep: Final = Depends(user_api_key_auth)
_RESPONSES_TAGS: Final[list[str | Enum]] = ["responses"]  # mutable-ok: fastapi's route signature requires list tags

_TOOL_PAYLOAD_KEYS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "custom": ("name", "description", "format"),
        "function": ("name", "description", "parameters", "strict"),
    }
)
_EMPTY_TOOL_PAYLOAD: Final[Mapping[str, object]] = MappingProxyType({})


def _convert_tool_payload_value(key: str, value: object, *, to_chat: bool) -> object:
    if key != "format" or not isinstance(value, dict):
        return value
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        convert_custom_tool_format_to_chat_shape,
        convert_custom_tool_format_to_responses_shape,
    )

    convert = convert_custom_tool_format_to_chat_shape if to_chat else convert_custom_tool_format_to_responses_shape
    return convert(value)


def _convert_tool_envelope(obj: object, *, to_chat: bool) -> object:
    if not isinstance(obj, dict):
        return obj
    tool_type: Final = obj.get("type")
    payload_keys: Final = _TOOL_PAYLOAD_KEYS.get(tool_type)
    if payload_keys is None:
        return obj
    nested: Final = obj.get(tool_type)
    nested_source: Final = nested if isinstance(nested, dict) else _EMPTY_TOOL_PAYLOAD
    payload: Final = {
        key: _convert_tool_payload_value(key, nested_source[key] if key in nested_source else obj[key], to_chat=to_chat)
        for key in payload_keys
        if key in nested_source or key in obj
    }
    if "name" not in payload:
        return obj
    return {"type": tool_type, tool_type: payload} if to_chat else {"type": tool_type, **payload}


def _normalize_tool_dialect(
    data: dict, *, to_chat: bool
) -> dict:  # mutable-ok: the parsed request body contract is a plain dict
    tools: Final = data.get("tools")
    tool_choice: Final = data.get("tool_choice")
    normalized_tools: Final = (
        [_convert_tool_envelope(tool, to_chat=to_chat) for tool in tools] if isinstance(tools, list) else tools
    )
    normalized_choice: Final = _convert_tool_envelope(tool_choice, to_chat=to_chat)
    if normalized_tools == tools and normalized_choice == tool_choice:
        return data
    replaceable: Final = (("tools", normalized_tools), ("tool_choice", normalized_choice))
    return {**data, **{key: value for key, value in replaceable if key in data}}


def _is_chat_completions_body(data: Mapping[str, object]) -> bool:
    messages: Final = data.get("messages")
    if isinstance(messages, list) and messages:
        return True
    return "messages" in data and "input" not in data


_CURSOR_THINKING_SEPARATOR: Final = "-thinking-"
_CURSOR_FAST_SUFFIX: Final = "-fast"
_CURSOR_THINKING_LEVELS: Final[frozenset[str]] = frozenset(get_args(REASONING_EFFORT))


class _CursorModelVariant(NamedTuple):
    base_model: str
    reasoning_effort: str | None


def _parse_cursor_model_variant(model: str) -> _CursorModelVariant:
    stripped: Final = model.removesuffix(_CURSOR_FAST_SUFFIX)
    base, separator, level = stripped.rpartition(_CURSOR_THINKING_SEPARATOR)
    if separator and base and level in _CURSOR_THINKING_LEVELS:
        return _CursorModelVariant(base, level)
    return _CursorModelVariant(stripped, None)


def _router_can_serve(model: str, llm_router: "Router | None") -> bool:
    if llm_router is None:
        return False
    if llm_router.is_recognized_model(model):
        return True
    if model in llm_router.team_public_model_names:
        return True
    return bool(llm_router.pattern_router.get_pattern(model))


def _resolve_cursor_model_variant(
    data: dict, llm_router: "Router | None"
) -> dict:  # mutable-ok: the parsed request body contract is a plain dict
    model: Final = data.get("model")
    if not isinstance(model, str) or _router_can_serve(model, llm_router):
        return data
    variant: Final = _parse_cursor_model_variant(model)
    if variant.base_model == model or not _router_can_serve(variant.base_model, llm_router):
        return data
    resolved: Final = {**data, "model": variant.base_model}
    if variant.reasoning_effort is None:
        return resolved
    if _is_chat_completions_body(data):
        if "reasoning_effort" in data:
            return resolved
        return {**resolved, "reasoning_effort": variant.reasoning_effort}
    reasoning: Final = data.get("reasoning")
    if isinstance(reasoning, dict):
        if reasoning.get("effort"):
            return resolved
        return {**resolved, "reasoning": {**reasoning, "effort": variant.reasoning_effort}}
    return {**resolved, "reasoning": {"effort": variant.reasoning_effort}}


async def _resolve_cursor_model_variant_before_auth(request: Request) -> None:
    from litellm.proxy.proxy_server import llm_router

    try:
        raw_body: Final = await read_request_body(request=request)
    except (json.JSONDecodeError, ProxyException):
        return
    resolved: Final = _resolve_cursor_model_variant(raw_body, llm_router)
    if resolved is not raw_body:
        safe_set_request_parsed_body(request=request, parsed_body=resolved)


@router.post(
    "/v1/responses",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=RESPONSES_API_CREATE_RESPONSE_SCHEMAS,
)
@router.post(
    "/responses",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=RESPONSES_API_CREATE_RESPONSE_SCHEMAS,
)
@router.post(
    "/openai/v1/responses",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=RESPONSES_API_CREATE_RESPONSE_SCHEMAS,
)
async def responses_api(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses

    Supports background mode with polling_via_cache for partial response retrieval.
    When background=true and polling_via_cache is enabled, returns a polling_id immediately
    and streams the response in the background, updating Redis cache.

    ```bash
    # Normal request
    curl -X POST http://localhost:4000/v1/responses \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -d '{
        "model": "gpt-4o",
        "input": "Tell me about AI"
    }'

    # Background request with polling
    curl -X POST http://localhost:4000/v1/responses \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -d '{
        "model": "gpt-4o",
        "input": "Tell me about AI",
        "background": true
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        native_background_mode,
        polling_cache_ttl,
        polling_via_cache_enabled,
        proxy_config,
        proxy_logging_obj,
        read_request_body,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    native_data_generator: Final = partial(select_data_generator, responses_stream_errors=True)
    data = await read_request_body(request=request)  # rebind-ok: pre-existing rebinding on a rename-only line

    # Check if polling via cache should be used for this request
    from litellm.proxy.response_polling.polling_handler import (
        should_use_polling_for_request,
    )

    should_use_polling: Final = should_use_polling_for_request(
        background_mode=data.get("background", False),
        polling_via_cache_enabled=polling_via_cache_enabled,
        redis_cache=redis_usage_cache,
        model=data.get("model", ""),
        llm_router=llm_router,
        native_background_mode=native_background_mode,
    )

    # If polling is enabled, use polling mode
    if should_use_polling:
        from litellm.proxy.response_polling.background_streaming import (
            background_streaming_task,
        )
        from litellm.proxy.response_polling.polling_handler import (
            ResponsePollingHandler,
        )

        verbose_proxy_logger.info("Starting background response with polling for model=%s", data.get("model"))

        # Run pre-call checks (rate limits, guardrails, budget) BEFORE creating
        # polling ID. This ensures rate-limited requests get a synchronous 429
        # instead of a polling ID that immediately fails in the background task.
        processor = ProxyBaseLLMRequestProcessing(data=data)
        try:
            data, _logging_obj = await processor.common_processing_pre_call_logic(
                request=request,
                general_settings=general_settings,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
                version=version,
                proxy_config=proxy_config,
                user_model=user_model,
                user_temperature=user_temperature,
                user_request_timeout=user_request_timeout,
                user_max_tokens=user_max_tokens,
                user_api_base=user_api_base,
                model=None,
                route_type="aresponses",
                llm_router=llm_router,
            )
            raise_if_required_body_param_missing(route_type="aresponses", data=data, llm_router=llm_router)
        except Exception as e:
            raise await processor.handle_llm_api_exception(
                e=e,
                user_api_key_dict=user_api_key_dict,
                proxy_logging_obj=proxy_logging_obj,
                version=version,
            )

        # Initialize polling handler with configured TTL (from global config)
        polling_handler: Final = ResponsePollingHandler(
            redis_cache=redis_usage_cache,
            ttl=polling_cache_ttl,  # Global var set at startup
        )

        # Generate polling ID
        polling_id: Final = ResponsePollingHandler.generate_polling_id()

        # Create initial state in Redis
        initial_state: Final = await polling_handler.create_initial_state(
            polling_id=polling_id,
            request_data=data,
        )

        # Start background task to stream and update cache.
        # Pass pre-processed data so the background task skips pre-call logic
        # (rate limits, guardrails already checked above).
        asyncio.create_task(
            background_streaming_task(
                polling_id=polling_id,
                data=dict(_without_managed_background_marker(data)),
                polling_handler=polling_handler,
                request=request,
                fastapi_response=fastapi_response,
                user_api_key_dict=user_api_key_dict,
                general_settings=general_settings,
                llm_router=llm_router,
                proxy_config=proxy_config,
                proxy_logging_obj=proxy_logging_obj,
                select_data_generator=native_data_generator,
                user_model=user_model,
                user_temperature=user_temperature,
                user_request_timeout=user_request_timeout,
                user_max_tokens=user_max_tokens,
                user_api_base=user_api_base,
                version=version,
            )
        )

        # Return OpenAI Response object format (initial state)
        # https://platform.openai.com/docs/api-reference/responses/object
        return initial_state

    # Normal response flow
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        response: Final = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aresponses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=(
                partial(_streamed_background_data_generator, native_data_generator, proxy_logging_obj)
                if requests_background(_PROXY_SERVER_REQUEST_ADAPTER.validate_python(data))
                else native_data_generator
            ),
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        if isinstance(response, ResponsesAPIResponse) and (data.get("background") or is_background_response(response)):
            managed_files_obj: Final = _managed_files_hook(proxy_logging_obj)

            if managed_files_obj and llm_router:
                try:
                    hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
                    model_id: Final = hidden_params.get("model_id", None)

                    if not model_id:
                        verbose_proxy_logger.warning(
                            "No model_id found in response hidden params for response %s, skipping managed object storage",
                            response.id,
                        )
                        raise Exception("No model_id found in response hidden params")
                    await managed_files_obj.store_unified_object_id(
                        unified_object_id=response.id,
                        file_object=response,
                        litellm_parent_otel_span=None,
                        model_object_id=decrypted_response_id(response.id),
                        file_purpose="response",
                        user_api_key_dict=user_api_key_dict,
                        request_tags=_background_response_spend_tags(processor.data),
                        persist_attribution=True,
                    )

                    verbose_proxy_logger.info(
                        "Stored background response %s in managed objects table with unified_id=%s",
                        response.id,
                        response.id,
                    )
                except Exception as e:
                    verbose_proxy_logger.error("Failed to store background response in managed objects table: %s", e)

        return response
    except ModifyResponseException as e:
        # Guardrail passthrough: return violation message in Responses API format (200)
        _data: Final = e.request_data
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=_data,
        )

        if data.get("stream") is True:
            block_chunks: Final = OpenAIResponsesHandler().build_block_sse_chunks(e)

            async def _blocked_stream() -> AsyncGenerator[str, None]:
                for chunk in block_chunks:
                    yield chunk.decode()
                yield "data: [DONE]\n\n"

            return await create_response(generator=_blocked_stream(), media_type="text/event-stream", headers={})
        return build_blocked_response(e)
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/cursor/models",
    dependencies=(_user_api_key_auth_dep,),
    tags=_RESPONSES_TAGS,
)
@router.get(
    "/cursor/v1/models",
    dependencies=(_user_api_key_auth_dep,),
    tags=_RESPONSES_TAGS,
)
async def cursor_model_list(
    user_api_key_dict: UserAPIKeyAuth = _user_api_key_auth_dep,
):
    """
    OpenAI-compatible model listing for the Cursor BYOK base URL.

    Clients pointed at `<proxy>/cursor` as an OpenAI-compatible base URL resolve and
    verify models via `GET {base}/models` (the OpenAI SDK contract). Without this
    route those requests fall through to the Cursor Cloud Agents passthrough, which
    demands a Cursor API key and 401s, so key verification silently fails before any
    chat request is ever sent. Delegates to the standard `/v1/models` handler.
    """
    from litellm.proxy.proxy_server import model_list

    return await model_list(user_api_key_dict=user_api_key_dict)


def _dispatched_prompt_token_counter(logging_obj: LiteLLMLoggingObj) -> Callable[[], int]:
    return lambda: count_request_input_tokens(
        model=str(logging_obj.model_call_details.get("model") or ""),
        request_input=messages_since_newest_compaction(logging_obj.model_call_details.get("messages")),
        system_input=None,
    )


@router.post(
    "/cursor/chat/completions",
    dependencies=[
        Depends(_resolve_cursor_model_variant_before_auth),
        Depends(user_api_key_auth),
    ],
    tags=["responses"],
)
async def cursor_chat_completions(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cursor BYOK endpoint. Accepts both request shapes Cursor sends to its OpenAI-compatible
    base URL and always answers in chat completions format.

    Cursor agent mode sends Responses API format bodies (`input`, flat tool defs, `reasoning`,
    custom tools) to the chat/completions path while expecting chat completions responses;
    those are routed through the Responses API pipeline and converted back. Genuine chat
    completions bodies (`messages` present) are routed through the standard chat completions
    pipeline, after normalizing each level of the `tools` array and `tool_choice` to the chat
    completions shapes OpenAI requires. Cursor mixes Responses API shapes into chat bodies
    per level, independently: a flat tool def (`{"type": "custom", "name": "ApplyPatch", ...}`)
    gets nested under `custom`, and a flat grammar format
    (`{"type": "grammar", "definition", "syntax"}`) gets wrapped as
    `{"type": "grammar", "grammar": {...}}` wherever it appears, including inside tool defs
    Cursor already sent pre-nested.

    ```bash
    curl -X POST http://localhost:4000/cursor/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -d '{
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Hello"}]
    }'
    Responds back in chat completions format.
    ```
    """
    from litellm.completion_extras.litellm_responses_transformation.handler import (
        responses_api_bridge,
    )
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.proxy.proxy_server import (
        async_data_generator,
        chat_completion,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.types.llms.openai import ResponsesAPIResponse
    from litellm.types.utils import ModelResponse

    raw_body: Final = await read_request_body(request=request)

    if _is_chat_completions_body(raw_body):
        # Genuine chat completions body (Cursor sends these for models whose BYOK it
        # already fixed); delegate so behavior matches /chat/completions exactly.
        # Keyed on messages CONTENT, not key presence: Cursor can send a null or
        # empty messages stub alongside a real agent-mode input array
        normalized: Final = _normalize_tool_dialect(raw_body, to_chat=True)
        if normalized is not raw_body:
            safe_set_request_parsed_body(request=request, parsed_body=normalized)
        return await chat_completion(
            request=request,
            fastapi_response=fastapi_response,
            model=None,
            user_api_key_dict=user_api_key_dict,
        )

    # OpenAI's Responses API rejects chat-completions-only stream_options
    # (Cursor sends include_usage); usage arrives via response.completed anyway.
    # Rebuild rather than pop: _read_request_body can return the request-scope
    # cached parsed-body dict itself, and removing keys from it corrupts the
    # cache's key snapshot so later readers get an empty body
    body_without_stream_options: Final = {key: value for key, value in raw_body.items() if key != "stream_options"}

    data: Final = _normalize_tool_dialect(body_without_stream_options, to_chat=False)

    processor: Final = ProxyBaseLLMRequestProcessing(data=data)

    def cursor_data_generator(response, user_api_key_dict, request_data, request=None):
        """
        Custom generator that transforms Responses API streaming chunks to chat completion chunks.

        This generator is used for the cursor endpoint to convert Responses API format responses
        to chat completion format that Cursor IDE expects.

        Args:
            response: The streaming Responses API event iterator (router-wrapped or not)
            user_api_key_dict: User API key authentication dict
            request_data: Request data containing model, logging_obj, etc.
            request: The originating FastAPI request, forwarded for disconnect handling

        Returns:
            Async generator that yields SSE-formatted chat completion chunks
        """
        # Any async-iterable here is a Responses API event stream needing conversion.
        # Class-identity checks miss router-wrapped streams (e.g.
        # HiddenParamsAsyncIteratorWrapper around LiteLLMCompletionStreamingIterator),
        # which previously leaked raw Responses events to the client.
        if hasattr(response, "__anext__"):
            # Transform Responses API iterator to chat completion iterator
            # Cast to AsyncIterator[str] since the stream implements __aiter__/__anext__
            completion_stream: Final = responses_api_bridge.transformation_handler.get_model_response_iterator(
                streaming_response=cast(AsyncIterator[str], response),
                sync_stream=False,
                json_mode=False,
            )
            # Wrap in CustomStreamWrapper to get the async generator
            logging_obj: Final = request_data.get("litellm_logging_obj")
            streamwrapper: Final = CustomStreamWrapper(
                completion_stream=completion_stream,
                model=request_data.get("model", ""),
                custom_llm_provider=None,
                logging_obj=logging_obj,
                count_prompt_tokens=(
                    _dispatched_prompt_token_counter(logging_obj)
                    if isinstance(logging_obj, LiteLLMLoggingObj)
                    else None
                ),
            )
            # Use async_data_generator to format as SSE
            return async_data_generator(
                response=streamwrapper,
                user_api_key_dict=user_api_key_dict,
                request_data=request_data,
                request=request,
                dispatched_call_type=processor.route_type,
            )
        # Otherwise, use the default generator
        return async_data_generator(
            response=response,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
            request=request,
            dispatched_call_type=processor.route_type,
        )

    try:
        response: Final = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aresponses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=cursor_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        # Transform non-streaming Responses API response to chat completions format
        if isinstance(response, ResponsesAPIResponse):
            logging_obj: Final = processor.data.get("litellm_logging_obj")
            transformed_response: Final = responses_api_bridge.transformation_handler.transform_response(
                model=processor.data.get("model", ""),
                raw_response=response,
                model_response=ModelResponse(),
                logging_obj=cast(Any, logging_obj),
                request_data=processor.data,
                messages=processor.data.get("input", []),
                optional_params={},
                litellm_params={},
                encoding=None,
                api_key=None,
                json_mode=None,
            )
            return transformed_response

        # Streaming responses are already transformed by cursor_select_data_generator
        return response
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=RESPONSES_API_RESPONSE_SCHEMAS,
)
@router.get(
    "/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=RESPONSES_API_RESPONSE_SCHEMAS,
)
@router.get(
    "/openai/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=RESPONSES_API_RESPONSE_SCHEMAS,
)
async def get_response(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Get a response by ID.
    
    Supports both:
    - Polling IDs (litellm_poll_*): Returns cumulative cached content from background responses
    - Provider response IDs: Passes through to provider API
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/get
    
    ```bash
    # Get polling response
    curl -X GET http://localhost:4000/v1/responses/litellm_poll_abc123 \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY"
    
    # Get provider response
    curl -X GET http://localhost:4000/v1/responses/resp_abc123 \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY"
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        read_request_body,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    # Check if this is a polling ID
    if ResponsePollingHandler.is_polling_id(response_id):
        # Handle polling response
        if not redis_usage_cache:
            raise HTTPException(
                status_code=500,
                detail="Redis cache not configured. Polling requires Redis.",
            )

        polling_handler: Final = ResponsePollingHandler(redis_cache=redis_usage_cache)

        # Get current state from cache
        state: Final = await polling_handler.get_state(response_id)

        if not state:
            raise HTTPException(
                status_code=404,
                detail=f"Polling response {response_id} not found or expired",
            )

        # Return the whole state directly (OpenAI Response object format)
        # https://platform.openai.com/docs/api-reference/responses/object
        return state

    # Normal provider response flow
    data: Final = await read_request_body(request=request)
    read_settles_billing: Final = not carries_client_side_overrides(data)
    data["response_id"] = response_id
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        response: Final[object] = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aget_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
    await _bill_finished_background_response(
        response=response,
        response_id=response_id,
        user_api_key_dict=user_api_key_dict,
        fastapi_response=fastapi_response,
        read_settles_billing=read_settles_billing,
    )
    return response


def _background_response_spend_tags(data: Mapping[str, object]) -> tuple[str, ...] | None:
    proxy_server_request: Final = data.get("proxy_server_request")
    client_tags: Final = (
        StandardLoggingPayloadSetup.get_request_tags(
            {}, _PROXY_SERVER_REQUEST_ADAPTER.validate_python(proxy_server_request)
        )
        if isinstance(proxy_server_request, dict)
        else []
    )
    tags: Final = (*(request_tags_from_request_data(data) or ()), *client_tags)
    return tags or None


@runtime_checkable
class _BackgroundResponseStore(Protocol):
    async def store_unified_object_id(
        self,
        unified_object_id: str,
        file_object: ResponsesAPIResponse,
        litellm_parent_otel_span: None,
        model_object_id: str,
        file_purpose: Literal["response"],
        user_api_key_dict: UserAPIKeyAuth,
        request_tags: Sequence[str] | None = None,
        persist_attribution: bool = False,
    ) -> None: ...

    async def track_untracked_background_response(
        self,
        response_id: str,
        response: ResponsesAPIResponse,
        user_api_key_dict: UserAPIKeyAuth,
        request_tags: Sequence[str] | None = None,
    ) -> bool: ...


def _managed_files_hook(proxy_logging_obj: "ProxyLogging") -> _BackgroundResponseStore | None:
    hook: Final = proxy_logging_obj.get_proxy_hook("managed_files")
    return hook if isinstance(hook, _BackgroundResponseStore) else None


async def _bill_finished_background_response(
    response: object,
    response_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    fastapi_response: Response,
    read_settles_billing: bool,
) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    if not isinstance(response, ResponsesAPIResponse) or not is_background_response(response):
        return
    if response.status not in TERMINAL_RESPONSE_STATUSES:
        return
    managed_files_hook: Final = _managed_files_hook(proxy_logging_obj)
    track_untracked: Final = (
        partial(
            managed_files_hook.track_untracked_background_response,
            response_id,
            user_api_key_dict=user_api_key_dict,
        )
        if managed_files_hook is not None
        else None
    )
    charge: Final = await _settle_background_billing(
        (lambda cost_check: cost_check.bill_if_finished(response_id, response, track_untracked))
        if read_settles_billing
        else (lambda cost_check: cost_check.bill_from_deployment(response_id, track_untracked))
    )
    _report_charge(charge, fastapi_response)


class _BilledCharge(Protocol):
    @property
    def cost(self) -> float | None: ...

    @property
    def logging_obj(self) -> "LiteLLMLoggingObj": ...


_UntrackedResponseTracker: TypeAlias = Callable[[ResponsesAPIResponse], Awaitable[bool]]


class _BackgroundCostCheck(Protocol):
    async def bill_from_deployment(
        self, response_id: str, track_untracked: _UntrackedResponseTracker | None = None
    ) -> _BilledCharge | None: ...

    async def bill_if_finished(
        self,
        response_id: str,
        response: ResponsesAPIResponse,
        track_untracked: _UntrackedResponseTracker | None = None,
    ) -> _BilledCharge | None: ...


def _background_cost_check() -> _BackgroundCostCheck | None:
    from litellm.proxy.proxy_server import llm_router, prisma_client, proxy_logging_obj

    if prisma_client is None or llm_router is None:
        return None
    try:
        from litellm_enterprise.proxy.common_utils.check_responses_cost import CheckResponsesCost
    except ImportError:
        return None

    return CheckResponsesCost(  # pyright: ignore[reportUnknownVariableType]  # litellm_enterprise is absent from the type-check venv
        proxy_logging_obj=proxy_logging_obj, prisma_client=prisma_client, llm_router=llm_router
    )


_BillingSettlement: TypeAlias = Callable[[_BackgroundCostCheck], Awaitable[_BilledCharge | None]]
_BILLING_RETRY_DELAYS_SECONDS: Final = (5.0, 30.0, 120.0)
_billing_retries: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: the event loop keeps only weak task refs


@dataclass(frozen=True, slots=True)
class _BillingFailed:
    error: Exception


async def _attempt_billing(
    settle: _BillingSettlement, cost_check: _BackgroundCostCheck
) -> _BilledCharge | None | _BillingFailed:
    try:
        return await settle(cost_check)
    except Exception as e:
        return _BillingFailed(e)


async def _retry_background_billing(
    settle: _BillingSettlement, cost_check: _BackgroundCostCheck, retry_delays: Sequence[float]
) -> None:
    for delay in retry_delays:
        await asyncio.sleep(delay)
        outcome = await _attempt_billing(settle, cost_check)
        if not isinstance(outcome, _BillingFailed):
            return
        verbose_proxy_logger.warning("Retrying the bill for a finished background response failed: %s", outcome.error)
    verbose_proxy_logger.error("Gave up billing a finished background response; the cost poll bills it when enabled")


def _schedule_billing_retry(
    settle: _BillingSettlement, cost_check: _BackgroundCostCheck, retry_delays: Sequence[float]
) -> None:
    retry: Final = asyncio.create_task(_retry_background_billing(settle, cost_check, retry_delays))
    _billing_retries.add(retry)
    retry.add_done_callback(_billing_retries.discard)


async def _settle_background_billing(
    settle: _BillingSettlement, retry_delays: Sequence[float] = _BILLING_RETRY_DELAYS_SECONDS
) -> _BilledCharge | None:
    cost_check: Final = _background_cost_check()
    if cost_check is None:
        return None
    outcome: Final = await _attempt_billing(settle, cost_check)
    if not isinstance(outcome, _BillingFailed):
        return outcome
    verbose_proxy_logger.warning(
        "Billing a finished background response failed, retrying in background: %s", outcome.error
    )
    _schedule_billing_retry(settle, cost_check, retry_delays)
    return None


def _report_charge(charge: _BilledCharge | None, fastapi_response: Response) -> None:
    if charge is None or charge.cost is None:
        return
    billed_headers: Final = cost_headers(charge.cost, charge.logging_obj)
    fastapi_response.headers.update({name: value for name, value in billed_headers.items() if value is not None})


def _caller_may_address_response(
    response_id: str, user_api_key_dict: UserAPIKeyAuth, proxy_logging_obj: "ProxyLogging"
) -> bool:
    security_hook: Final = proxy_logging_obj.get_proxy_hook("responses_id_security")
    if not isinstance(security_hook, ResponsesIDSecurity):
        return True
    try:
        security_hook.authorize_response_id(response_id, user_api_key_dict)
    except HTTPException:
        return False
    return True


def _without_managed_background_marker(request_data: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = request_data.get("litellm_metadata")
    if not isinstance(metadata, Mapping):
        return dict(request_data)
    return {
        **request_data,
        "litellm_metadata": {
            key: value
            for key, value in _PROXY_SERVER_REQUEST_ADAPTER.validate_python(metadata).items()
            if key != MANAGED_BACKGROUND_RESPONSE_METADATA_KEY
        },
    }


def _streamed_background_data_generator(
    select_data_generator: Callable[..., AsyncIterator[str | bytes]],
    proxy_logging_obj: "ProxyLogging",
    response: object,
    user_api_key_dict: UserAPIKeyAuth,
    request_data: dict[str, object],
    request: Request | None = None,
) -> AsyncIterator[str | bytes]:
    chunks: Final = select_data_generator(
        response=response, user_api_key_dict=user_api_key_dict, request_data=request_data, request=request
    )
    managed_files_hook: Final = _managed_files_hook(proxy_logging_obj)
    if managed_files_hook is None:
        return chunks
    track_untracked: Final = partial(
        managed_files_hook.track_untracked_background_response,
        user_api_key_dict=user_api_key_dict,
        request_tags=_background_response_spend_tags(request_data),
    )
    return _bill_streamed_background_response(response, chunks, track_untracked)


async def _bill_streamed_background_response(
    stream: object,
    chunks: AsyncIterator[str | bytes],
    track_untracked: Callable[[str, ResponsesAPIResponse], Awaitable[bool]],
) -> AsyncIterator[str | bytes]:
    async for chunk in chunks:
        completed: object = ProxyBaseLLMRequestProcessing.extract_completed_responses_response(stream)
        if not isinstance(completed, ResponsesAPIResponse):
            yield chunk
            continue
        await _bill_finished_stream(completed, track_untracked)
        yield chunk
        async for remaining_chunk in chunks:
            yield remaining_chunk
        return
    late_completed: Final = ProxyBaseLLMRequestProcessing.extract_completed_responses_response(stream)
    if isinstance(late_completed, ResponsesAPIResponse):
        await _bill_finished_stream(late_completed, track_untracked)


async def _bill_finished_stream(
    response: ResponsesAPIResponse, track_untracked: Callable[[str, ResponsesAPIResponse], Awaitable[bool]]
) -> None:
    if not is_background_response(response):
        return
    await _settle_background_billing(
        lambda cost_check: cost_check.bill_if_finished(response.id, response, partial(track_untracked, response.id))
    )


@router.delete(
    "/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=DELETE_RESPONSE_SCHEMAS,
)
@router.delete(
    "/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=DELETE_RESPONSE_SCHEMAS,
)
@router.delete(
    "/openai/v1/responses/{response_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=DELETE_RESPONSE_SCHEMAS,
)
async def delete_response(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a response by ID.
    
    Supports both:
    - Polling IDs (litellm_poll_*): Deletes from Redis cache
    - Provider response IDs: Passes through to provider API
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/delete
    
    ```bash
    curl -X DELETE http://localhost:4000/v1/responses/resp_abc123 \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY"
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        read_request_body,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    # Check if this is a polling ID
    if ResponsePollingHandler.is_polling_id(response_id):
        # Handle polling response deletion
        if not redis_usage_cache:
            raise HTTPException(status_code=500, detail="Redis cache not configured.")

        polling_handler: Final = ResponsePollingHandler(redis_cache=redis_usage_cache)

        # Get state to verify access
        state: Final = await polling_handler.get_state(response_id)

        if not state:
            raise HTTPException(status_code=404, detail=f"Polling response {response_id} not found")

        # Delete from cache
        success: Final = await polling_handler.delete_polling(response_id)

        if success:
            return DeleteResponseResult(id=response_id, object="response", deleted=True)
        else:
            raise HTTPException(status_code=500, detail="Failed to delete polling response")

    # Normal provider response flow
    data: Final = await read_request_body(request=request)
    data["response_id"] = response_id
    charge: Final = (
        await _settle_background_billing(lambda cost_check: cost_check.bill_from_deployment(response_id))
        if _caller_may_address_response(response_id, user_api_key_dict, proxy_logging_obj)
        else None
    )
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        deleted: Final = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="adelete_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
    _report_charge(charge, fastapi_response)
    return deleted


@router.get(
    "/v1/responses/{response_id}/input_items",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=RESPONSE_ITEM_LIST_SCHEMAS,
)
@router.get(
    "/responses/{response_id}/input_items",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=RESPONSE_ITEM_LIST_SCHEMAS,
)
@router.get(
    "/openai/v1/responses/{response_id}/input_items",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
    responses=RESPONSE_ITEM_LIST_SCHEMAS,
)
async def get_response_input_items(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List input items for a response."""
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        read_request_body,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Final = await read_request_body(request=request)
    data["response_id"] = response_id
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="alist_input_items",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/responses/compact",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/responses/compact",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/openai/v1/responses/compact",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def compact_response(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Compact a response by running a compaction pass over a conversation.
    
    Returns encrypted, opaque items that can be used to reduce context size.
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/compact
    
    ```bash
    curl -X POST http://localhost:4000/v1/responses/compact \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -d '{
        "model": "gpt-4o",
        "input": [{"role": "user", "content": "Hello"}]
    }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        read_request_body,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: Final = await read_request_body(request=request)
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acompact_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


class _ResponsesApiErrorDetail(TypedDict):
    message: ReadOnly[str]
    type: ReadOnly[str]
    param: ReadOnly[str | None]
    code: ReadOnly[str | None]


class _ResponsesApiErrorBody(TypedDict):
    error: ReadOnly[_ResponsesApiErrorDetail]


class _ResponsesInputTokensResult(TokenCountSourceFields):
    object: ReadOnly[str]
    input_tokens: ReadOnly[int]


class _TokenCountPayload(TypedDict):
    model: ReadOnly[str]
    messages: ReadOnly[tuple[Mapping[str, object], ...]]
    tools: ReadOnly[object]


class _TokenCounter(Protocol):
    def __call__(
        self, *, request: TokenCountRequest, call_endpoint: bool, responses_request: Mapping[str, object] | None
    ) -> Awaitable[TokenCountResponse]: ...


def _proxy_token_counter() -> _TokenCounter:
    from litellm.proxy.proxy_server import count_request_tokens

    return count_request_tokens


_token_counter_dep: Final = Depends(_proxy_token_counter)


def _proxy_llm_router() -> Router | None:
    from litellm.proxy.proxy_server import llm_router

    return llm_router


_llm_router_dep: Final = Depends(_proxy_llm_router)


def _proxy_response_id_security() -> ResponsesIDSecurity | None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    security_hook: Final = proxy_logging_obj.get_proxy_hook("responses_id_security")
    return security_hook if isinstance(security_hook, ResponsesIDSecurity) else None


_response_id_security_dep: Final = Depends(_proxy_response_id_security)


class _ManagedFileAccess(Protocol):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: Literal["aresponses"],
    ) -> object: ...


def _proxy_managed_files_hook() -> _ManagedFileAccess | None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    return proxy_logging_obj.get_proxy_hook("managed_files")


_managed_files_hook_dep: Final = Depends(_proxy_managed_files_hook)


async def _with_managed_file_id_mapping(
    data: Mapping[str, object], managed_files: _ManagedFileAccess | None, user_api_key_dict: UserAPIKeyAuth
) -> Mapping[str, object]:
    caller_fields: Final = {name: value for name, value in data.items() if name != MODEL_FILE_ID_MAPPING_FIELD}
    if managed_files is None:
        return caller_fields
    from litellm.proxy.proxy_server import user_api_key_cache

    hooked: Final = dict(caller_fields)
    try:
        await managed_files.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict, cache=user_api_key_cache, data=hooked, call_type="aresponses"
        )
    except HTTPException as e:
        raise ProxyException(message=str(e.detail), type="permission_error", param=None, code=e.status_code) from e
    mapping: Final = hooked.get(MODEL_FILE_ID_MAPPING_FIELD)
    return caller_fields if mapping is None else {**caller_fields, MODEL_FILE_ID_MAPPING_FIELD: mapping}


def _with_authorized_previous_response(
    data: Mapping[str, object], security: ResponsesIDSecurity | None, user_api_key_dict: UserAPIKeyAuth
) -> Mapping[str, object]:
    previous_response_id: Final = data.get("previous_response_id")
    if security is None or not isinstance(previous_response_id, str) or not previous_response_id:
        return data
    try:
        authorized_id: Final = security.authorize_response_id(previous_response_id, user_api_key_dict)
    except HTTPException as e:
        raise ProxyException(message=str(e.detail), type="permission_error", param=None, code=e.status_code) from e
    return {**data, "previous_response_id": authorized_id, ADDRESSED_RESPONSE_ID_FIELD: previous_response_id}


def _responses_invalid_request_response(message: str, param: str | None, code: str | None) -> JSONResponse:
    body: Final[_ResponsesApiErrorBody] = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code,
        }
    }
    return JSONResponse(status_code=400, content=body)


class _ProviderErrorMessage(TypedDict):
    message: ReadOnly[str]


class _NestedProviderError(TypedDict):
    error: ReadOnly[_ProviderErrorMessage]


@with_config(ConfigDict(extra="ignore"))
class _ProviderOpenAIErrorDetail(TypedDict):
    message: ReadOnly[str]
    type: ReadOnly[str]
    param: NotRequired[ReadOnly[str | None]]
    code: NotRequired[ReadOnly[str | None]]


@with_config(ConfigDict(extra="forbid"))
class _ProviderOpenAIError(TypedDict):
    error: ReadOnly[_ProviderOpenAIErrorDetail]


_PROVIDER_COUNT_ERROR_TYPE: Final = "token_counting_error"
_PROVIDER_OPENAI_ERROR: Final = TypeAdapter(_ProviderOpenAIError)
_NESTED_PROVIDER_ERROR: Final = TypeAdapter(_NestedProviderError)


def _provider_error_json(raw_message: str) -> str:
    start: Final = raw_message.find("{")
    return raw_message if start == -1 else raw_message[start:]


def _provider_error_message(raw_message: str) -> str:
    error_json: Final = _provider_error_json(raw_message)
    try:
        return _NESTED_PROVIDER_ERROR.validate_json(error_json)["error"]["message"]
    except ValidationError:
        flat_message: Final = AnthropicExceptionMapping.extract_error_message(error_json)
        return raw_message if flat_message == error_json else flat_message


def _provider_count_error_body(raw_message: str, status_code: int) -> _ResponsesApiErrorBody:
    try:
        provider_error: Final = _PROVIDER_OPENAI_ERROR.validate_json(_provider_error_json(raw_message))["error"]
    except ValidationError:
        return {
            "error": {
                "message": _provider_error_message(raw_message),
                "type": openai_error_type_for_status(status_code),
                "param": None,
                "code": None,
            }
        }
    return {
        "error": {
            "message": provider_error["message"],
            "type": provider_error["type"],
            "param": provider_error.get("param"),
            "code": provider_error.get("code"),
        }
    }


def _provider_count_error_response(exc: ProxyException, request: Request) -> JSONResponse:
    from litellm.proxy.proxy_server import (
        _close_dangling_otel_server_span,  # pyright: ignore[reportPrivateUsage]  # proxy_server keeps the span-close helper private; a route that answers an error itself must stamp the OTel server span like the global ProxyException handler does
        general_settings_view,
    )

    status_code: Final = error_status_code(exc, 500)
    _close_dangling_otel_server_span(request, status_code, exc=exc)
    body: Final = _provider_count_error_body(exc.message, status_code)
    call_id: Final = error_body_call_id(general_settings_view(), exc.headers.get(LITELLM_CALL_ID_HEADER))
    return JSONResponse(
        status_code=status_code,
        content={"error": with_call_id({**body["error"]}, call_id)},
        headers=exc.headers,
    )


def _missing_responses_param_response(param: str) -> JSONResponse:
    return _responses_invalid_request_response(
        message=f"Missing required parameter: '{param}'.",
        param=param,
        code="missing_required_parameter",
    )


def _responses_input_as_token_count_messages(
    input_value: str | ResponseInputParam,
    instructions: str | None,
) -> tuple[Mapping[str, object], ...]:
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    request_params: Final[ResponsesAPIOptionalRequestParams] = {"instructions": instructions}
    transformed: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_value,
        responses_api_request=request_params,
        only_since_compaction=True,
    )
    return tuple(
        message if isinstance(message, dict) else message.model_dump(exclude_none=True) for message in transformed
    )


@router.post(
    "/v1/responses/input_tokens",
    dependencies=(_user_api_key_auth_dep,),
    tags=_RESPONSES_TAGS,
)
@router.post(
    "/responses/input_tokens",
    dependencies=(_user_api_key_auth_dep,),
    tags=_RESPONSES_TAGS,
)
@router.post(
    "/openai/v1/responses/input_tokens",
    dependencies=(_user_api_key_auth_dep,),
    tags=_RESPONSES_TAGS,
)
async def responses_input_tokens(
    request: Request,
    token_counter: _TokenCounter = _token_counter_dep,
    llm_router: Router | None = _llm_router_dep,
    user_api_key_dict: UserAPIKeyAuth = _user_api_key_auth_dep,
    response_id_security: ResponsesIDSecurity | None = _response_id_security_dep,
    managed_files: _ManagedFileAccess | None = _managed_files_hook_dep,
):
    """
    Count the input tokens of a Responses API request without calling the model.

    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/input-tokens

    ```bash
    curl -X POST http://localhost:4000/v1/responses/input_tokens \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    -d '{
        "model": "gpt-4o",
        "input": "Hello, how are you?"
    }'
    ```

    Returns: `{"object": "response.input_tokens", "input_tokens": <count>}`
    """
    data: Final = await read_request_body(request=request)
    model_name: Final = data.get("model")
    input_value: Final = data.get("input")
    if not isinstance(model_name, str) or not model_name:
        return _missing_responses_param_response("model")
    if input_value is None:
        return _missing_responses_param_response("input")
    if isinstance(input_value, (str, list)) and not input_value:
        return _responses_invalid_request_response(
            message="""One of "input" or "previous_response_id" or 'prompt' or 'conversation' must be provided.""",
            param=None,
            code="missing_required_parameter",
        )

    counted_fields: Final = {
        name: data[name] for name in ("model", "drop_params", "additional_drop_params", *TOOL_FIELDS) if name in data
    }
    token_fields: Final = resolve_counted_tool_fields(counted_fields, llm_router, user_api_key_dict.team_id)
    if isinstance(token_fields, ProxyException):
        return _responses_invalid_request_response(message=token_fields.message, param=token_fields.param, code=None)

    try:
        payload: Final[_TokenCountPayload] = {
            "model": model_name,
            "messages": _responses_input_as_token_count_messages(
                input_value=input_value,
                instructions=data.get("instructions"),
            ),
            "tools": token_fields.get("tools"),
        }
        token_request: Final = TokenCountRequest.model_validate(payload)
    except ValidationError as e:
        return _responses_invalid_request_response(
            message=first_validation_error(e), param=str(e.errors(include_url=False)[0]["loc"][0]), code=None
        )
    except Exception as e:
        return _responses_invalid_request_response(
            message=f"Invalid request for token counting: {e}", param=None, code=None
        )

    request_fields: Final = _PROXY_SERVER_REQUEST_ADAPTER.validate_python(data)
    responses_request: Final = await _with_managed_file_id_mapping(
        _with_authorized_previous_response(
            {**{name: value for name, value in request_fields.items() if name not in counted_fields}, **token_fields},
            response_id_security,
            user_api_key_dict,
        ),
        managed_files,
        user_api_key_dict,
    )
    try:
        token_response: Final = await token_counter(
            request=token_request, call_endpoint=True, responses_request=responses_request
        )
    except ProxyException as e:
        if e.type != _PROVIDER_COUNT_ERROR_TYPE:
            raise
        return _provider_count_error_response(e, request)
    result: Final[_ResponsesInputTokensResult] = {
        "object": "response.input_tokens",
        "input_tokens": token_response.total_tokens,
        **count_source_fields(token_response),
    }
    return result


@router.post(
    "/v1/responses/{response_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/responses/{response_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
@router.post(
    "/openai/v1/responses/{response_id}/cancel",
    dependencies=[Depends(user_api_key_auth)],
    tags=["responses"],
)
async def cancel_response(
    response_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Cancel a response by ID.
    
    Supports both:
    - Polling IDs (litellm_poll_*): Cancels background response and updates status in Redis
    - Provider response IDs: Passes through to provider API
    
    Follows the OpenAI Responses API spec: https://platform.openai.com/docs/api-reference/responses/cancel
    
    ```bash
    # Cancel polling response
    curl -X POST http://localhost:4000/v1/responses/litellm_poll_abc123/cancel \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY"
    
    # Cancel provider response
    curl -X POST http://localhost:4000/v1/responses/resp_abc123/cancel \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY"
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        read_request_body,
        redis_usage_cache,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler

    # Check if this is a polling ID
    if ResponsePollingHandler.is_polling_id(response_id):
        # Handle polling response cancellation
        if not redis_usage_cache:
            raise HTTPException(status_code=500, detail="Redis cache not configured.")

        polling_handler: Final = ResponsePollingHandler(redis_cache=redis_usage_cache)

        # Get current state to verify it exists
        state: Final = await polling_handler.get_state(response_id)

        if not state:
            raise HTTPException(status_code=404, detail=f"Polling response {response_id} not found")

        # Cancel the polling response (sets status to "cancelled")
        success: Final = await polling_handler.cancel_polling(response_id)

        if success:
            # Fetch the updated state with cancelled status
            updated_state: Final = await polling_handler.get_state(response_id)

            # Return the whole state directly (now with status="cancelled")
            return updated_state
        else:
            raise HTTPException(status_code=500, detail="Failed to cancel polling response")

    # Normal provider response flow
    data: Final = await read_request_body(request=request)
    read_settles_billing: Final = not carries_client_side_overrides(data)
    data["response_id"] = response_id
    processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        response: Final[object] = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="acancel_responses",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor.handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
    await _bill_finished_background_response(
        response=response,
        response_id=response_id,
        user_api_key_dict=user_api_key_dict,
        fastapi_response=fastapi_response,
        read_settles_billing=read_settles_billing,
    )
    return response


def _resolve_responses_ws_session_limit_seconds() -> float:
    from litellm.proxy.proxy_server import general_settings

    field: Final = "responses_websocket_session_limit_seconds"
    raw: Final = _RESPONSES_WS_CONFIG_VALUE_ADAPTER.validate_python(general_settings.get(field))
    try:
        return ConfigGeneralSettings.model_validate(
            {} if raw is None else {field: raw}
        ).responses_websocket_session_limit_seconds
    except ValidationError as e:
        default: Final = DEFAULT_RESPONSES_WEBSOCKET_SESSION_LIMIT_SECONDS
        verbose_proxy_logger.warning("invalid general_settings.%s=%r (%s); using default %ss", field, raw, e, default)
        return default


async def _read_ws_model_from_first_frame(
    websocket: WebSocket,
    query_model: str | None = None,
) -> tuple[str, str] | None:
    """Read the first WS frame and return (model, raw_message), or None on error.

    Sends an appropriate error frame and closes the socket before returning None.
    The session-duration deadline is enforced by the caller, not here.
    """
    try:
        first_message: Final = await websocket.receive_text()
    except WebSocketDisconnect:
        return None
    except Exception:
        verbose_proxy_logger.exception("Responses WebSocket error reading first message")
        await websocket.close(code=1011, reason="Internal server error")
        return None

    try:
        first_event: Final = as_str_mapping(_DECODED_JSON.validate_python(json.loads(first_message)))
    except json.JSONDecodeError:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "First message is not valid JSON.",
                    },
                }
            )
        )
        await websocket.close(code=1008, reason="Invalid JSON in first message")
        return None

    if first_event is None or first_event.get("type") != "response.create":
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "First message must be a response.create JSON object.",
                    },
                }
            )
        )
        await websocket.close(code=1008, reason="Invalid first message")
        return None

    model: Final = query_model or _extract_model_from_first_ws_event(first_event)
    if not model:
        await websocket.send_text(
            json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "No model provided. Supply ?model=<model> in the URL or include 'model' in the first response.create event.",
                    },
                }
            )
        )
        await websocket.close(code=1008, reason="No model provided")
        return None

    return model, first_message


def _extract_model_from_first_ws_event(first_event: object) -> str | None:
    """Extract model from a response.create WS event, handling flat and nested formats.

    Flat:   {"type": "response.create", "model": "gpt-4o", ...}
    Nested: {"type": "response.create", "response": {"model": "gpt-4o", ...}}
    """
    if not isinstance(first_event, dict):
        return None
    nested: Final = first_event.get("response")
    return (nested.get("model") if isinstance(nested, dict) else None) or first_event.get("model")


class _ResponseCreateRoutingHints(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    input: str | Sequence[object] | None = None
    previous_response_id: str | None = None
    response: "_ResponseCreateRoutingHints | None" = None


def _routing_hints_from_first_ws_frame(first_message: str) -> Mapping[str, object]:
    try:
        frame: Final = _ResponseCreateRoutingHints.model_validate_json(first_message)
    except ValidationError:
        return EMPTY_MAPPING
    nested: Final = frame.response or frame
    hints: Final = {
        "input": frame.input if nested.input is None else nested.input,
        "previous_response_id": (
            frame.previous_response_id if nested.previous_response_id is None else nested.previous_response_id
        ),
    }
    return MappingProxyType({key: value for key, value in hints.items() if value is not None})


def _responses_ws_failure_frame(failure: Exception) -> str:
    proxy_exception: Final = handle_exception_on_proxy(failure)
    status: Final = int(proxy_exception.code) if str(proxy_exception.code).isdigit() else 500
    error_type: Final = (
        "rate_limit_exceeded" if status == 429 else "invalid_request_error" if 400 <= status < 500 else "server_error"
    )
    return json.dumps(
        {"type": "error", "status": status, "error": {"type": error_type, "message": proxy_exception.message}}
    )


async def _enforce_responses_ws_first_frame_model_auth(
    request: Request,
    model: str,
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: "Router | None",
) -> None:
    from litellm.proxy.auth.user_api_key_auth import (
        enforce_key_and_fallback_model_access,
        run_centralized_common_checks,
    )
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_model_list,
        master_key,
        user_custom_auth,
    )

    request_data: Final = {"model": model}
    route: Final = request.scope.get("path") or "/v1/responses"
    if master_key is None and not (
        general_settings.get("enable_jwt_auth", False)
        or general_settings.get("enable_oauth2_auth", False)
        or general_settings.get("enable_oauth2_proxy_auth", False)
    ):
        return
    if user_custom_auth is not None and not general_settings.get("custom_auth_run_common_checks", False):
        return
    await enforce_key_and_fallback_model_access(
        valid_token=user_api_key_dict,
        request_data=request_data,
        route=route,
        request=request,
        llm_model_list=llm_model_list,
        llm_router=llm_router,
    )
    await run_centralized_common_checks(
        user_api_key_auth_obj=user_api_key_dict,
        request=request,
        request_data=request_data,
        route=route,
    )


_WS_FRAME: Final = TypeAdapter(dict[str, object])
_GUARDRAIL_INFORMATION: Final = TypeAdapter(list[dict[str, object]])
_PII_TOKENS: Final = TypeAdapter(dict[str, str])
_GUARDRAIL_INFORMATION_KEY: Final = "standard_logging_guardrail_information"
_GUARDED_FRAME_FIELDS: Final = ("input", "instructions", "tools")
_CLIENT_METADATA_KEYS: Final = ("metadata", "litellm_metadata")
# the guardrail list is merged from key, team and the admission frame; a later frame cannot shrink it
_CLIENT_GUARDRAIL_SELECTORS: Final = ("guardrails", "guardrail_config", "policies")


class _PreCallGuardrailWalk(Protocol):
    async def pre_call_hook(
        self,
        *,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict[str, object],  # mutable-ok: ProxyLogging.pre_call_hook hands the guardrails a dict they edit
        call_type: Literal["aresponses"],
        guardrails_only: bool,
    ) -> dict[str, object]: ...  # mutable-ok: ProxyLogging.pre_call_hook returns the guardrails' own dict


def _responses_ws_frame_guard(
    admitted: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    proxy_logging_obj: _PreCallGuardrailWalk,
    record_refused_turn: Callable[[tuple[Mapping[str, object], ...], Exception], None],
) -> ResponsesWsFrameGuard:
    """Socket admission skips guardrails (the first frame's ``input`` is not part of the admitted request), so every
    ``response.create`` frame gets the pre_call guardrail walk the HTTP ``/v1/responses`` path runs on its body."""
    metadata_key: Final = get_metadata_variable_name_from_kwargs(admitted)
    admitted_bucket: Final = admitted.get(metadata_key)
    bucket: Final = {
        key: value
        for key, value in (
            _WS_FRAME.validate_python(admitted_bucket) if isinstance(admitted_bucket, dict) else {}
        ).items()
        if key != _GUARDRAIL_INFORMATION_KEY
    }
    model: Final = admitted.get("model")

    # a blocked frame is billed as its own failed request; only the first turn settles the admission reservation
    rejected_frame_auth: Final = user_api_key_dict.model_copy(update={"budget_reservation": None})

    async def guard(frame: str) -> GuardedResponsesWsFrame:
        frame_obj: Final = _WS_FRAME.validate_json(frame)
        nested: Final = frame_obj.get("response")
        if isinstance(nested, dict) and any(key in frame_obj for key in _GUARDED_FRAME_FIELDS):
            raise ValueError(
                "response.create carries input, instructions or tools both at the top level and under response"
            )
        body: Final = _WS_FRAME.validate_python(nested) if isinstance(nested, dict) else frame_obj
        frame_data: Final = {
            **{
                key: _without_untrusted_metadata(key, value)
                for key, value in body.items()
                if key != "type" and key not in _CLIENT_GUARDRAIL_SELECTORS
            },
            "model": model,
            "litellm_call_id": str(uuid.uuid4()),
            # shallow: the bucket carries proxy-owned objects (the OTEL parent span) that cannot be deep-copied
            metadata_key: {**bucket},
        }
        try:
            guarded_data: Final = await proxy_logging_obj.pre_call_hook(
                user_api_key_dict=user_api_key_dict, data=frame_data, call_type="aresponses", guardrails_only=True
            )
        except Exception as e:
            await _bill_rejected_frame(rejected_frame_auth, e, frame_data)
            raise
        if guarded_data.get("model") != model:
            rerouted: Final = ValueError(
                f"a guardrail rerouted this turn to {guarded_data.get('model')!r}; an open Responses WebSocket"
                f" is bound to {model!r}, send this turn over HTTP /v1/responses"
            )
            await _bill_rejected_frame(rejected_frame_auth, rerouted, frame_data)
            raise rerouted
        # a guardrail may delete a field (the Responses translation pops ``instructions``), so start without them
        guarded_fields: Final = {key: guarded_data[key] for key in _GUARDED_FRAME_FIELDS if key in guarded_data}
        kept: Final = {key: value for key, value in body.items() if key not in _GUARDED_FRAME_FIELDS}
        rewritten: Final = (
            {**frame_obj, "response": {**kept, **guarded_fields}}
            if isinstance(nested, dict)
            else {**kept, **guarded_fields}
        )
        guardrail_information: Final = _recorded_guardrail_information(guarded_data.get(metadata_key))
        return GuardedResponsesWsFrame(
            frame=json.dumps(rewritten),
            guardrail_information=guardrail_information,
            pii_tokens=_recorded_pii_tokens(guarded_data.get("metadata")),
            bill_refusal=partial(record_refused_turn, guardrail_information),
        )

    return guard


def _without_untrusted_metadata(key: str, value: object) -> object:
    """Like the HTTP path, drop the proxy-owned ``user_api_key_*`` and guardrail control keys a client may forge."""
    if key not in _CLIENT_METADATA_KEYS or not isinstance(value, dict):
        return value
    return {
        name: item
        for name, item in _WS_FRAME.validate_python(value).items()
        if not name.startswith("user_api_key_")
        and name not in _UNTRUSTED_METADATA_CONTROL_FIELDS
        and name not in _CLIENT_PRICING_METADATA_FIELDS
        and name not in ("headers", "pii_tokens", _GUARDRAIL_INFORMATION_KEY, *_CLIENT_GUARDRAIL_SELECTORS)
    }


async def _bill_rejected_frame(
    user_api_key_dict: UserAPIKeyAuth,
    exception: Exception,
    frame_data: dict[str, object],  # mutable-ok: the spend logger stamps the failure row on the request dict
) -> None:
    """Only the spend logger sees a rejected frame; the other failure hooks (the rate limiter's slot release,
    alerting) belong to the live connection, which stays open. Billing never replaces the guardrail's own error."""
    for callback in litellm.callbacks:
        if isinstance(callback, ProxyDBLogger):
            try:
                await callback.async_post_call_failure_hook(  # pyright: ignore[reportUnknownMemberType]  # bare dict
                    request_data=frame_data, original_exception=exception, user_api_key_dict=user_api_key_dict
                )
            except Exception:  # noqa: BLE001  # billing never replaces the guardrail's own error
                verbose_proxy_logger.exception("[Non-Blocking] Responses WebSocket: billing a rejected frame failed")


def _recorded_pii_tokens(metadata: object) -> Mapping[str, str]:
    """Presidio stores the tokens it masked a frame with in ``metadata.pii_tokens``; the socket unmasks with them."""
    recorded: Final = _WS_FRAME.validate_python(metadata).get("pii_tokens") if isinstance(metadata, dict) else None
    return MappingProxyType(_PII_TOKENS.validate_python(recorded)) if isinstance(recorded, dict) else _NO_PII_TOKENS


_NO_PII_TOKENS: Final[Mapping[str, str]] = MappingProxyType({})


@dataclass(slots=True)
class _RefusedTurns:
    """Turns the deployment refused after their guardrails passed. The socket's close-time failure row bills their
    guardrail records, so its failure hooks, model group and call type cover those turns too, and the first refusal
    is that row's exception when the session limit cancels the relay before it returns one."""

    first: Exception | None = None
    records: tuple[Mapping[str, object], ...] = ()

    def add(self, records: tuple[Mapping[str, object], ...], refusal: Exception) -> None:
        self.records = (*self.records, *records)
        if self.first is None:
            self.first = refusal

    def billed_on(self, request_data: Mapping[str, object]) -> Mapping[str, object]:
        if not self.records:
            return request_data
        metadata_key: Final = get_metadata_variable_name_from_kwargs(request_data)
        bucket: Final = request_data.get(metadata_key)
        fields: Final = _WS_FRAME.validate_python(bucket) if isinstance(bucket, dict) else {}
        recorded: Final = (*_recorded_guardrail_information(fields), *self.records)
        return {
            **request_data,
            metadata_key: {**fields, _GUARDRAIL_INFORMATION_KEY: [dict(record) for record in recorded]},
        }


def _recorded_guardrail_information(bucket: object) -> tuple[Mapping[str, object], ...]:
    recorded: Final = (
        _WS_FRAME.validate_python(bucket).get(_GUARDRAIL_INFORMATION_KEY) if isinstance(bucket, dict) else None
    )
    return tuple(_GUARDRAIL_INFORMATION.validate_python(recorded)) if recorded else ()


async def _responses_websocket_session(
    websocket: WebSocket,
    model: str | None,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.route_llm_request import route_request

    result: Final = await _read_ws_model_from_first_frame(websocket, query_model=model)
    if result is None:
        return
    resolved_model, first_message = result

    data: dict[str, object] = {
        "model": resolved_model,
        "websocket": websocket,
        "first_message": first_message,
    }

    # Construct a synthetic Request for pre-call processing
    headers_list: Final = list(websocket.scope.get("headers") or [])
    scope: Final[dict[str, object]] = {
        "type": "http",
        "method": "POST",
        "path": "/v1/responses",
        "headers": headers_list,
    }
    request: Final = Request(scope=scope)
    request._url = websocket.url  # pyright: ignore[reportPrivateUsage]  # Starlette WebSocket URL storage

    _body_bytes: Final = json.dumps({"model": resolved_model}).encode()

    async def return_body():
        return _body_bytes

    request.body = return_body

    # Phase 1: pre-call processing (auth, guardrails, rate limits)
    base_llm_response_processor: Final = ProxyBaseLLMRequestProcessing(data=data)
    try:
        if not model:
            await _enforce_responses_ws_first_frame_model_auth(
                request=request,
                model=resolved_model,
                user_api_key_dict=user_api_key_dict,
                llm_router=llm_router,
            )
        (
            data,
            litellm_logging_obj,
        ) = await base_llm_response_processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            model=resolved_model,
            route_type="_aresponses_websocket",
            skip_guardrails=True,
        )
    except Exception as e:
        verbose_proxy_logger.exception("Responses WebSocket pre-call error")
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": str(e),
                        },
                    }
                )
            )
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        await websocket.close(code=1008, reason="Pre-call error")
        return

    refused_turns: Final = _RefusedTurns()
    routed_data: Final = dict(
        data,
        user_api_key_dict=user_api_key_dict,
        responses_ws_frame_guard=_responses_ws_frame_guard(
            data, user_api_key_dict, proxy_logging_obj, refused_turns.add
        ),
        **_routing_hints_from_first_ws_frame(first_message),
    )

    async def write_failure_row(exception: Exception) -> None:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=exception,
            request_data=dict(refused_turns.billed_on(routed_data)),
        )

    def bill_socket_failure(exception: Exception) -> asyncio.Task[object]:
        return spawn_logging_task(
            asyncio.get_running_loop(), write_failure_row(exception), task_name="Responses WebSocket failure row"
        )

    async def relay_and_bill_failure() -> asyncio.Task[object] | None:
        try:
            llm_call: Final = await route_request(
                data=routed_data,
                route_type="_aresponses_websocket",
                llm_router=llm_router,
                user_model=user_model,
            )
            outcome: Final = await llm_call
        except asyncio.CancelledError:
            if refused_turns.first is not None:
                _ = bill_socket_failure(refused_turns.first)
            raise
        except Exception as e:
            crash_row: Final = bill_socket_failure(e)
            with contextlib.suppress(Exception):
                await websocket.send_text(_responses_ws_failure_frame(e))
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason="Internal server error")
            return crash_row
        return bill_socket_failure(outcome) if isinstance(outcome, Exception) else None

    # Phase 2: route to upstream provider
    try:
        failure_row: Final = await relay_and_bill_failure()
    finally:
        await proxy_logging_obj.arelease_responses_websocket_slot(user_api_key_dict)
    if failure_row is not None:
        with contextlib.suppress(Exception):
            await asyncio.shield(failure_row)


@router.websocket("/v1/responses")
@router.websocket("/responses")
async def responses_websocket_endpoint(
    websocket: WebSocket,
    model: str | None = fastapi.Query(None, description="The model to use for the responses WebSocket session."),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth_websocket),
):
    """
    Responses API WebSocket mode endpoint.

    Keeps a persistent WebSocket connection for response.create events,
    enabling lower-latency agentic workflows with many tool-call round trips.

    Follows the OpenAI split: the bearer token is validated at connection time
    (before accept); the model is resolved either from the ?model= query param
    or from the first response.create frame, whichever is present.

    The session is bounded by a lifetime measured from accept, configured via
    general_settings.responses_websocket_session_limit_seconds (60-7200,
    default 3600). There is no separate first-frame deadline, so
    pre-established connections may sit idle until their first response.create.

    See: https://developers.openai.com/api/docs/guides/websocket-mode/
    """
    # Accept the WebSocket handshake. Key was already validated by the Depends
    # above; we can safely accept regardless of whether ?model= was supplied.
    requested_protocols: Final = [
        p.strip() for p in (websocket.headers.get("sec-websocket-protocol") or "").split(",") if p.strip()
    ]
    accept_kwargs: Final[dict] = {}
    if requested_protocols:
        accept_kwargs["subprotocol"] = requested_protocols[0]
    await websocket.accept(**accept_kwargs)

    limit_seconds: Final = _resolve_responses_ws_session_limit_seconds()
    session_task: Final = asyncio.ensure_future(
        _responses_websocket_session(websocket=websocket, model=model, user_api_key_dict=user_api_key_dict)
    )
    try:
        await asyncio.wait_for(asyncio.shield(session_task), timeout=limit_seconds)
    except asyncio.TimeoutError:
        verbose_proxy_logger.info("Responses WebSocket closed: session duration limit reached")
        session_task.cancel()
        with contextlib.suppress(Exception):
            await websocket.close(code=1000, reason="Session duration limit reached")
    finally:
        if not session_task.done():
            session_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await session_task
