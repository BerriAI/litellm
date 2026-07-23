import asyncio
import json
import logging
import math
import time
import traceback
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Literal,
    Optional,
    Tuple,
    Union,
)

import anyio
import httpx
import orjson
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

import litellm
from litellm._logging import _redact_string, verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import (
    DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE,
    DEFAULT_MAX_RECURSE_DEPTH,
    LITELLM_DETAILED_TIMING,
    LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED,
    MAX_PAYLOAD_SIZE_FOR_DEBUG_LOG,
    RETURN_RAW_MODEL_NAME_METADATA_KEY,
    STREAM_SSE_DATA_PREFIX,
)
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.dd_tracing import NullTracer, tracer
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_response_utils.get_headers import (
    get_response_headers,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import check_response_size_is_safe
from litellm.proxy.common_utils.callback_utils import (
    get_logging_caching_headers,
    get_remaining_tokens_and_requests_from_request_data,
)
from litellm.proxy.dd_span_tagger import DDSpanTagger
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.router_utils.add_retry_fallback_headers import get_hidden_params_dict
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.router import RouterRateLimitError
from litellm.types.utils import ServerToolUse

# Type alias for streaming chunk serializer (chunk after hooks + cost injection -> wire format)
StreamChunkSerializer = Callable[[Any], str]
# Type alias for streaming error serializer (ProxyException -> wire format)
StreamErrorSerializer = Callable[[ProxyException], str]

if TYPE_CHECKING:
    from litellm.proxy.proxy_server import ProxyConfig as _ProxyConfig

    ProxyConfig = _ProxyConfig
else:
    ProxyConfig = Any
from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
from litellm.types.utils import (
    ModelResponse,
    ModelResponseStream,
    StandardLoggingPayloadErrorInformation,
    Usage,
)

# Datadog streaming spans are a no-op when ddtrace is not enabled, but the
# ``with tracer.trace(...)`` context manager still allocates a NullSpan and
# runs __enter__/__exit__ for every streamed chunk. Resolve once at import so
# the per-chunk hot path can skip the context manager entirely when tracing
# is off (the default).
_DD_STREAMING_TRACE_ENABLED = not isinstance(tracer, NullTracer)


_CLIENT_DISCONNECTED_ERROR_INFORMATION: StandardLoggingPayloadErrorInformation = {
    "error_code": str(LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED),
    "error_message": "Client disconnected the request",
    "error_class": "ClientDisconnected",
}


def _should_return_raw_model_name(request_data: dict[str, object]) -> bool:
    return any(
        isinstance(metadata, dict) and metadata.get(RETURN_RAW_MODEL_NAME_METADATA_KEY) is True
        for metadata in (request_data.get("metadata"), request_data.get("litellm_metadata"))
    )


def _apply_client_disconnect_metadata(target_metadata: Optional[dict[str, object]]) -> None:
    if target_metadata is None:
        return
    target_metadata["client_disconnected"] = True
    target_metadata["error_information"] = dict(_CLIENT_DISCONNECTED_ERROR_INFORMATION)


async def _record_streaming_client_disconnect_if_needed(
    request: Request | None,
    request_data: dict,
    client_disconnected: bool = False,
) -> bool:
    if not client_disconnected:
        if request is None:
            return False
        try:
            disconnected = await request.is_disconnected()
        except Exception:  # noqa: BLE001
            return False
        if not disconnected:
            return False

    logging_obj = request_data.get("litellm_logging_obj")
    if logging_obj is not None:
        litellm_params = logging_obj.model_call_details.setdefault("litellm_params", {})
        _lp_metadata = litellm_params.get("metadata")
        if _lp_metadata is None:
            _lp_metadata = {}
            litellm_params["metadata"] = _lp_metadata
        _apply_client_disconnect_metadata(_lp_metadata)

        _mcd_metadata = logging_obj.model_call_details.get("metadata")
        if _mcd_metadata is None:
            _mcd_metadata = {}
            logging_obj.model_call_details["metadata"] = _mcd_metadata
        _apply_client_disconnect_metadata(_mcd_metadata)

    _rd_metadata = request_data.get("metadata")
    if _rd_metadata is None:
        _rd_metadata = {}
        request_data["metadata"] = _rd_metadata
    _apply_client_disconnect_metadata(_rd_metadata)

    _rd_litellm_params = request_data.get("litellm_params")
    if _rd_litellm_params is None:
        _rd_litellm_params = {}
        request_data["litellm_params"] = _rd_litellm_params
    _rd_lp_metadata = _rd_litellm_params.get("metadata")
    if _rd_lp_metadata is None:
        _rd_lp_metadata = {}
        _rd_litellm_params["metadata"] = _rd_lp_metadata
    _apply_client_disconnect_metadata(_rd_lp_metadata)

    verbose_proxy_logger.debug(
        "Recorded streaming client disconnect with error_code=499 for litellm_call_id=%s",
        request_data.get("litellm_call_id"),
    )
    return True


def _deferred_stream_logging_is_armed(request_data: dict) -> bool:
    logging_obj = request_data.get("litellm_logging_obj")
    if logging_obj is None:
        return False
    return (
        getattr(logging_obj, "_on_deferred_stream_complete", None) is not None
        and getattr(logging_obj, "_deferred_stream_complete_args", None) is not None
    )


async def _bill_partial_streamed_spend_on_disconnect(request_data: dict, response: object) -> bool:
    """
    A client disconnect throws GeneratorExit/CancelledError into the streaming
    generator, so neither the success nor the failure logging callback fires
    and the chunks already streamed (plus any sub-call cost folded into the
    logging object) would never reach spend tracking. Assemble the partial
    response from the wrapper's collected chunks and dispatch success logging
    for it; dispatch_success_handlers dedups against a natural end-of-stream
    dispatch via has_dispatched_final_stream_success.

    Awaited directly by the shielded cleanup rather than scheduled with
    create_task: the client is already gone so the extra latency is harmless,
    and an unrooted task could be garbage-collected before it bills.

    Returns True when a disconnect-time success event owns the request's
    max_parallel_requests slot release (one was dispatched here, or one had
    already been dispatched for this stream), so the caller can skip the
    explicit slot release and avoid a double release. Returns False when no
    success event fired (logging disabled, nothing streamed, or assembly
    failed) and the caller must release the slot itself.
    """
    if litellm.disable_streaming_logging is True:
        return False
    logging_obj = request_data.get("litellm_logging_obj")
    if not isinstance(logging_obj, LiteLLMLoggingObj):
        return False
    if logging_obj.model_call_details.get("has_dispatched_final_stream_success"):
        # A natural end-of-stream success event already fired and released the
        # slot; do not bill again, and let the caller skip the slot release.
        return True
    chunks: object = getattr(response, "chunks", None)
    if not isinstance(chunks, list) or not chunks:
        return False
    verbose_proxy_logger.debug(
        "Billing partial streamed spend for %s chunks after client disconnect, litellm_call_id=%s",
        len(chunks),
        request_data.get("litellm_call_id"),
    )
    messages: object = getattr(response, "messages", None)
    try:
        partial_response = litellm.stream_chunk_builder(
            chunks=chunks,
            messages=messages if isinstance(messages, list) else None,
            logging_obj=logging_obj,
        )
    except Exception as e:  # noqa: BLE001  # partial billing is best-effort; never break stream teardown
        verbose_proxy_logger.debug("Failed to assemble partial streamed response for disconnect billing: %s", e)
        return False
    if partial_response is None:
        return False
    try:
        await logging_obj.dispatch_success_handlers(
            partial_response,
            cache_hit=False,
            start_time=None,
            end_time=None,
            prefer_async_handlers=True,
        )
    except Exception as e:  # noqa: BLE001  # partial billing is best-effort; never break stream teardown
        verbose_proxy_logger.debug("Failed to dispatch disconnect billing event: %s", e)
        return False
    return True


async def _cancel_pending_gather_tasks(tasks: list["asyncio.Task[Any]"]) -> None:
    pending_tasks = [task for task in tasks if not task.done()]
    for task in pending_tasks:
        task.cancel()
    for task in pending_tasks:
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


def _serialize_http_exception_detail(
    detail: Any,
) -> Tuple[str, Optional[dict]]:
    """
    Convert an HTTPException.detail value into (message, structured_fields)
    for ProxyException / SSE error frames.

    Dict-detail HTTPExceptions raised by guardrails were previously str()-mangled
    into a Python repr blob, producing unparseable error responses on both the
    streaming and non-streaming proxy surfaces. This helper extracts a clean
    human-readable message while preserving the full payload as structured
    fields, so the dominant guardrail shapes (`{"error": "..."}` flat and
    `{"error": {"message": "..."}}` nested) both round-trip cleanly.
    """
    if isinstance(detail, str):
        return detail, None
    if isinstance(detail, dict):
        err = detail.get("error")
        if isinstance(err, str):
            return err, detail
        if isinstance(err, dict):
            nested_msg = err.get("message")
            if isinstance(nested_msg, str):
                return nested_msg, detail
        msg = detail.get("message")
        if isinstance(msg, str):
            return msg, detail
        return json.dumps(detail), detail
    return str(detail), None


def _collect_response_file_search_vector_store_ids(data: Dict[str, Any]) -> set[str]:
    vector_store_ids: set[str] = set()
    tools = data.get("tools")
    if not isinstance(tools, list):
        return vector_store_ids

    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "file_search":
            continue
        ids = tool.get("vector_store_ids") or []
        if not isinstance(ids, list):
            raise HTTPException(
                status_code=400,
                detail={"error": "file_search.vector_store_ids must be a list of strings"},
            )
        for vector_store_id in ids:
            if not isinstance(vector_store_id, str) or not vector_store_id:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "file_search.vector_store_ids must be a list of strings"},
                )
            vector_store_ids.add(vector_store_id)

    return vector_store_ids


async def _authorize_response_file_search_vector_stores(
    data: Dict[str, Any],
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    vector_store_ids = _collect_response_file_search_vector_store_ids(data)
    if not vector_store_ids:
        return

    from litellm.proxy.vector_store_endpoints.utils import (
        assert_user_can_access_vector_store_id,
    )

    for vector_store_id in sorted(vector_store_ids):
        await assert_user_can_access_vector_store_id(
            vector_store_id=vector_store_id,
            user_api_key_dict=user_api_key_dict,
        )


async def _parse_event_data_for_error(event_line: Union[str, bytes]) -> Optional[int]:
    """Parses an event line and returns an error code if present, else None."""
    event_line = event_line.decode("utf-8") if isinstance(event_line, bytes) else event_line
    if event_line.startswith("data: "):
        json_str = event_line[len("data: ") :].strip()
        if not json_str or json_str == "[DONE]":  # handle empty data or [DONE] message
            return None
        try:
            data = orjson.loads(json_str)
            if isinstance(data, dict) and "error" in data and isinstance(data["error"], dict):
                error_code_raw = data["error"].get("code")
                error_code: Optional[int] = None

                if isinstance(error_code_raw, int):
                    error_code = error_code_raw
                elif isinstance(error_code_raw, str):
                    try:
                        error_code = int(error_code_raw)
                    except ValueError:
                        verbose_proxy_logger.warning(
                            f"Error code is a string but not a valid integer: {error_code_raw}"
                        )
                        # Not a valid integer string, treat as if no valid code was found for this check
                        pass

                # Ensure error_code is a valid HTTP status code
                if error_code is not None and 100 <= error_code <= 599:
                    return error_code
                elif error_code_raw is not None:  # Log if original code was present but not valid
                    verbose_proxy_logger.warning(f"Error has invalid or non-convertible code: {error_code_raw}")
        except (orjson.JSONDecodeError, json.JSONDecodeError):
            # not a known error chunk
            pass
    return None


def _extract_error_from_sse_chunk(event_line: Union[str, bytes]) -> dict:
    """
    Extract error dictionary from SSE format chunk.

    Args:
        event_line: SSE format event line, e.g. "data: {"error": {...}}\n\n"

    Returns:
        Error dictionary in OpenAI API format
    """
    event_line = event_line.decode("utf-8") if isinstance(event_line, bytes) else event_line

    # Default error format
    default_error = {
        "message": "Unknown error",
        "type": "internal_server_error",
        "param": None,
        "code": "500",
    }

    if event_line.startswith("data: "):
        json_str = event_line[len("data: ") :].strip()
        if not json_str or json_str == "[DONE]":
            return default_error

        try:
            data = orjson.loads(json_str)
            if isinstance(data, dict) and "error" in data:
                error_obj = data["error"]
                if isinstance(error_obj, dict):
                    return error_obj
        except (orjson.JSONDecodeError, json.JSONDecodeError):
            pass

    return default_error


class _UpstreamClosingStreamingResponse(StreamingResponse):
    """StreamingResponse that always closes its body iterator and the wrapped
    upstream generator.

    When the client disconnects mid-stream, Starlette abandons the body
    iterator without calling aclose(), leaving the upstream LLM connection
    open until garbage collection; the backend (e.g. vLLM) keeps generating
    into a dead pipe. The upstream generator is closed directly (not via the
    body iterator) because aclose() on a never-started generator skips its
    body, so a cascade through it would be a no-op if the client disconnects
    before the first chunk is sent.
    """

    def __init__(
        self,
        content: AsyncGenerator[str, None],
        *,
        media_type: Optional[str] = None,
        headers: Optional[dict] = None,
        status_code: int = status.HTTP_200_OK,
        upstream_generator: Optional[AsyncGenerator[str, None]] = None,
    ) -> None:
        super().__init__(content, status_code=status_code, headers=headers, media_type=media_type)
        self._upstream_generator = upstream_generator

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                for target in (self.body_iterator, self._upstream_generator):
                    aclose = getattr(target, "aclose", None)
                    if aclose is None:
                        continue
                    try:
                        await aclose()
                    except BaseException as e:
                        verbose_proxy_logger.debug("error closing streaming generator: %s", e)


class _ClientDisconnectedBeforeFirstChunk(Exception):
    """Client went away during create_response's first-chunk buffering window.

    The upstream LLM stream has already been closed by the time this is raised.
    """


async def _wait_for_http_disconnect(request: Request) -> None:
    try:
        while True:
            message = await request.receive()
            if message.get("type") == "http.disconnect":
                return
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        verbose_proxy_logger.warning(
            "create_response: request.receive() raised %s; first-chunk disconnect monitoring disabled for this request",
            exc,
        )
        # A receive() failure must not masquerade as a disconnect.
        await asyncio.Event().wait()


async def _buffer_first_chunk_honoring_disconnect(
    generator: AsyncGenerator[str, None],
    request: Optional[Request],
) -> str:
    """Fetch the first streamed chunk, cancelling the upstream LLM call if the
    client disconnects before it arrives.

    create_response buffers the first chunk to detect error-only streams before
    handing the StreamingResponse to Starlette, which only begins listening for
    client disconnects once it is serving that response. A disconnect during a
    long time-to-first-token would otherwise leave the upstream call running
    until the request timeout (LIT-3568). Cancelling the fetch propagates into
    async_streaming_data_generator, whose finally block records the 499 and
    closes the upstream stream.
    """
    if request is None:
        return await generator.__anext__()

    chunk_task: asyncio.Task[str] = asyncio.ensure_future(generator.__anext__())
    disconnect_task: asyncio.Task[None] = asyncio.ensure_future(_wait_for_http_disconnect(request))
    try:
        await asyncio.wait({chunk_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
        # A completed disconnect_task has already consumed the http.disconnect
        # message, so Starlette's later listen_for_disconnect would never see it.
        # Take the cancellation path whenever a disconnect was observed, even if
        # the first chunk landed in the same scheduler turn.
        disconnect_observed = disconnect_task.done()
    finally:
        disconnect_task.cancel()
        try:
            await disconnect_task
        except BaseException:  # noqa: BLE001
            pass

    if not disconnect_observed and chunk_task.done() and not chunk_task.cancelled():
        return chunk_task.result()

    chunk_task.cancel()
    with anyio.CancelScope(shield=True):
        try:
            await chunk_task
        except BaseException:  # noqa: BLE001
            pass
        try:
            await generator.aclose()
        except BaseException as exc:  # noqa: BLE001
            verbose_proxy_logger.debug("create_response: error closing generator on disconnect: %s", exc)
    verbose_proxy_logger.info("create_response: client disconnected before first chunk, upstream LLM request cancelled")
    raise _ClientDisconnectedBeforeFirstChunk()


async def create_response(
    generator: AsyncGenerator[str, None],
    media_type: str,
    headers: dict,
    default_status_code: int = status.HTTP_200_OK,
    request: Optional[Request] = None,
) -> Union[StreamingResponse, JSONResponse]:
    """
    Create streaming response, checking if the first chunk is an error.
    If the first chunk is an error, return a standard JSON error response.
    Otherwise, return StreamingResponse and stream all content.
    """
    # Tell buffering reverse proxies (nginx, ingress-nginx, Envoy) to flush SSE
    # immediately instead of releasing the whole stream in one batch (issue #28384).
    streaming_headers = {
        **headers,
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    first_chunk_value: Optional[str] = None
    final_status_code = default_status_code

    try:
        # Handle coroutine that returns a generator
        if asyncio.iscoroutine(generator):
            generator = await generator

        # Now get the first chunk from the actual generator
        first_chunk_value = await _buffer_first_chunk_honoring_disconnect(generator, request)

        if first_chunk_value is not None:
            try:
                error_code_from_chunk = await _parse_event_data_for_error(first_chunk_value)
                if error_code_from_chunk is not None:
                    # First chunk is an error, stream hasn't really started yet
                    # Should return standard JSON error response instead of SSE format
                    final_status_code = error_code_from_chunk
                    verbose_proxy_logger.debug(
                        f"Error detected in first stream chunk. Returning JSON error response with status code: {final_status_code}"
                    )

                    # Parse error content
                    error_dict = _extract_error_from_sse_chunk(first_chunk_value)

                    # Consume and close generator (avoid resource leak)
                    try:
                        await generator.aclose()
                    except Exception:
                        pass

                    # Return JSON format error response
                    return JSONResponse(
                        status_code=final_status_code,
                        content={"error": error_dict},
                        headers=headers,
                    )
            except Exception as e:
                verbose_proxy_logger.debug(f"Error parsing first chunk value: {e}")

    except _ClientDisconnectedBeforeFirstChunk:
        # Client vanished during the time-to-first-token wait; the upstream
        # stream is already closed. Return a 499 the (now-gone) client never reads.
        return JSONResponse(
            status_code=LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED,
            content={
                "error": {
                    "message": _CLIENT_DISCONNECT_DETAIL,
                    "type": "client_disconnect",
                    "param": "None",
                    "code": str(LITELLM_HTTP_STATUS_CLIENT_DISCONNECTED),
                }
            },
            headers=headers,
        )
    except StopAsyncIteration:
        # Generator was empty. Default status
        async def empty_gen() -> AsyncGenerator[str, None]:
            if False:
                yield  # type: ignore

        return StreamingResponse(
            empty_gen(),
            media_type=media_type,
            headers=streaming_headers,
            status_code=default_status_code,
        )
    except Exception as e:
        # Unexpected error consuming first chunk.
        verbose_proxy_logger.exception(f"Error consuming first chunk from generator: {e}")

        # Preserve status code from HTTPException (e.g., guardrail blocks)
        error_status = getattr(e, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        raw_detail = getattr(e, "detail", "Error processing stream start")
        message, structured_fields = _serialize_http_exception_detail(raw_detail)

        existing_fields = getattr(e, "provider_specific_fields", None) or {}
        if structured_fields:
            merged_fields: Optional[dict] = {**existing_fields, **structured_fields}
        else:
            merged_fields = existing_fields or None

        # Match ProxyException.to_dict() shape so streaming and non-streaming
        # error frames are byte-identical.
        error_obj: Dict[str, Any] = {
            "message": message,
            "type": getattr(e, "type", "None"),
            "param": getattr(e, "param", "None"),
            "code": str(error_status),
        }
        if merged_fields:
            error_obj["provider_specific_fields"] = merged_fields

        async def error_gen_message() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'error': error_obj})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            error_gen_message(),
            media_type=media_type,
            headers=streaming_headers,
            status_code=error_status,
        )

    async def combined_generator() -> AsyncGenerator[str, None]:
        if not _DD_STREAMING_TRACE_ENABLED:
            # Fast path: no per-chunk span object / context-manager overhead.
            if first_chunk_value is not None:
                yield first_chunk_value
            async for chunk in generator:
                yield chunk
            return
        if first_chunk_value is not None:
            with tracer.trace(DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE):
                yield first_chunk_value
        async for chunk in generator:
            with tracer.trace(DD_TRACER_STREAMING_CHUNK_YIELD_RESOURCE):
                yield chunk

    return _UpstreamClosingStreamingResponse(
        combined_generator(),
        media_type=media_type,
        headers=streaming_headers,
        status_code=final_status_code,
        upstream_generator=generator,
    )


def _is_azure_model_router_request(model: str) -> bool:
    """
    Check if the requested model is an Azure Model Router.

    Azure Model Router models follow the pattern:
    - azure_ai/model_router/<deployment-name>
    - azure_ai/model-router
    - model_router/<deployment-name>
    - model-router

    Args:
        model: The requested model name

    Returns:
        bool: True if this is an Azure Model Router request
    """
    model_lower = model.lower()
    return "model-router" in model_lower or "model_router" in model_lower


def _override_openai_response_model(
    *,
    response_obj: Any,
    requested_model: str,
    log_context: str,
    return_raw_model_name: bool = False,
) -> None:
    """
    Force the OpenAI-compatible `model` field in the response to match what the client requested.

    LiteLLM internally prefixes some provider/deployment model identifiers (e.g. `hosted_vllm/...`).
    That internal identifier should not be returned to clients in the OpenAI `model` field.

    Note: This is intentionally verbose at debug level. A model mismatch is a useful signal that an
    internal model identifier is being stamped/preserved somewhere in the request/response pipeline.
    We log mismatches as debug (and then restamp to the client-requested value) so these paths stay
    observable for maintainers without breaking client compatibility or alarming operators.

    Responses that omit an OpenAI-style `model` field are left unchanged (silent return),
    including dict responses with no `model` key.

    Exceptions:
    1. If a fallback occurred (indicated by x-litellm-attempted-fallbacks header),
       we preserve the actual model that was used (the fallback model).
    2. If the request was to an Azure Model Router, we preserve the actual model
       that was used (e.g., gpt-5-nano-2025-08-07) instead of the router model.
    3. If this was a fastest_response batch completion, use the winning model's
       model group name instead of the comma-separated list the client sent.
    """
    if return_raw_model_name or not requested_model:
        return

    hidden_params = get_hidden_params_dict(response_obj)
    if isinstance(hidden_params, dict):
        # Check if a fallback occurred - if so, preserve the actual model used
        fallback_headers = hidden_params.get("additional_headers", {}) or {}
        attempted_fallbacks = fallback_headers.get("x-litellm-attempted-fallbacks", None)
        if attempted_fallbacks is not None and attempted_fallbacks > 0:
            verbose_proxy_logger.debug(
                "%s: fallback detected (attempted_fallbacks=%d), preserving actual model used instead of overriding to requested model.",
                log_context,
                attempted_fallbacks,
            )
            return

        # For fastest_response batch completions, use the winning model's group
        # name rather than the comma-separated list the client sent.
        if hidden_params.get("fastest_response_batch_completion"):
            winning_model = fallback_headers.get("x-litellm-model-group")
            if winning_model:
                verbose_proxy_logger.debug(
                    "%s: fastest_response detected, using winning model group=%r instead of requested=%r.",
                    log_context,
                    winning_model,
                    requested_model,
                )
                requested_model = winning_model
            else:
                verbose_proxy_logger.debug(
                    "%s: fastest_response detected but no model group header found, preserving actual model from response.",
                    log_context,
                )
                return

    # Check if this is an Azure Model Router request - if so, preserve the actual model used
    if _is_azure_model_router_request(requested_model):
        verbose_proxy_logger.debug(
            "%s: Azure Model Router detected - preserving actual model used from response instead of overriding to router model.",
            log_context,
        )
        return

    if isinstance(response_obj, dict):
        if "model" not in response_obj:
            return
        downstream_model = response_obj.get("model")
        if downstream_model != requested_model:
            verbose_proxy_logger.debug(
                "%s: response model mismatch - requested=%r downstream=%r. Overriding response['model'] to requested model.",
                log_context,
                requested_model,
                downstream_model,
            )
        response_obj["model"] = requested_model
        return

    if not hasattr(response_obj, "model"):
        return

    downstream_model = getattr(response_obj, "model", None)
    if downstream_model != requested_model:
        verbose_proxy_logger.debug(
            "%s: response model mismatch - requested=%r downstream=%r. Overriding response.model to requested model.",
            log_context,
            requested_model,
            downstream_model,
        )

    try:
        setattr(response_obj, "model", requested_model)
    except Exception as e:
        verbose_proxy_logger.debug(
            "%s: failed to override response.model=%r on response_type=%s. error=%s",
            log_context,
            requested_model,
            type(response_obj),
            str(e),
            exc_info=True,
        )


def _get_cost_breakdown_from_logging_obj(
    litellm_logging_obj: Optional[LiteLLMLoggingObj],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract discount and margin information from logging object's cost breakdown.

    Returns:
        Tuple of (original_cost, discount_amount, margin_total_amount, margin_percent)
    """
    if not litellm_logging_obj or not hasattr(litellm_logging_obj, "cost_breakdown"):
        return None, None, None, None

    cost_breakdown = litellm_logging_obj.cost_breakdown
    if not cost_breakdown:
        return None, None, None, None

    original_cost = cost_breakdown.get("original_cost")
    discount_amount = cost_breakdown.get("discount_amount")
    margin_total_amount = cost_breakdown.get("margin_total_amount")
    margin_percent = cost_breakdown.get("margin_percent")

    return original_cost, discount_amount, margin_total_amount, margin_percent


def _has_attribute_error_in_chain(exc: Exception) -> bool:
    """Walk the exception chain to find an AttributeError at any depth.

    Checks __cause__, __context__, and the litellm-specific original_exception
    attribute iteratively. Depth is capped at DEFAULT_MAX_RECURSE_DEPTH to
    avoid infinite loops from circular exception references.
    """
    stack: list[BaseException] = [exc]
    seen: set[int] = set()
    depth = 0
    while stack and depth < DEFAULT_MAX_RECURSE_DEPTH:
        current = stack.pop()
        exc_id = id(current)
        if exc_id in seen:
            continue
        seen.add(exc_id)
        if isinstance(current, AttributeError):
            return True
        for attr in ("__cause__", "__context__", "original_exception"):
            inner = getattr(current, attr, None)
            if inner is not None and isinstance(inner, BaseException):
                stack.append(inner)
        depth += 1
    return False


_CLIENT_DISCONNECT_DETAIL = "Client disconnected the request"


def _log_llm_api_exception(e: Exception) -> None:
    if getattr(e, "status_code", None) == 499 and getattr(e, "detail", None) == _CLIENT_DISCONNECT_DETAIL:
        verbose_proxy_logger.info(
            "litellm.proxy.proxy_server._handle_llm_api_exception(): client disconnected, upstream LLM request cancelled"
        )
        return
    verbose_proxy_logger.exception(
        f"litellm.proxy.proxy_server._handle_llm_api_exception(): Exception occured - {str(e)}"
    )


async def _cancel_llm_call_on_client_disconnect(
    request: Request,
    llm_api_call: "asyncio.Future[Any]",
    disconnect_event: asyncio.Event,
) -> None:
    try:
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                disconnect_event.set()
                llm_api_call.cancel()
                return
    except Exception as exc:
        verbose_proxy_logger.warning(
            "cancel_on_disconnect: request.receive() raised %s; upstream LLM call will not be cancelled on disconnect",
            exc,
        )


async def _await_llm_call_cancelling_on_disconnect(
    request: Request,
    llm_api_call: "asyncio.Future[Any]",
) -> Any:
    disconnect_event = asyncio.Event()
    monitor = asyncio.create_task(_cancel_llm_call_on_client_disconnect(request, llm_api_call, disconnect_event))
    try:
        return await llm_api_call
    except asyncio.CancelledError:
        if disconnect_event.is_set():
            raise HTTPException(
                status_code=499,
                detail=_CLIENT_DISCONNECT_DETAIL,
            )
        raise
    finally:
        monitor.cancel()


class ProxyBaseLLMRequestProcessing:
    def __init__(self, data: dict):
        self.data = data

    @staticmethod
    def get_custom_headers(
        *,
        user_api_key_dict: UserAPIKeyAuth,
        call_id: Optional[str] = None,
        model_id: Optional[str] = None,
        cache_key: Optional[str] = None,
        api_base: Optional[str] = None,
        version: Optional[str] = None,
        model_region: Optional[str] = None,
        response_cost: Optional[Union[float, str]] = None,
        hidden_params: Optional[dict] = None,
        fastest_response_batch_completion: Optional[bool] = None,
        request_data: Optional[dict] = {},
        timeout: Optional[Union[float, int, httpx.Timeout]] = None,
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = None,
        **kwargs,
    ) -> dict:
        exclude_values = {"", None, "None"}
        hidden_params = hidden_params or {}

        # Extract discount and margin info from cost_breakdown if available
        (
            original_cost,
            discount_amount,
            margin_total_amount,
            margin_percent,
        ) = _get_cost_breakdown_from_logging_obj(litellm_logging_obj=litellm_logging_obj)

        # Calculate updated spend for header (include current response_cost)
        current_spend = user_api_key_dict.spend or 0.0
        updated_spend = current_spend
        if response_cost is not None:
            try:
                # Convert response_cost to float if it's a string
                cost_value = float(response_cost) if isinstance(response_cost, str) else response_cost
                if cost_value > 0:
                    updated_spend = current_spend + cost_value
            except (ValueError, TypeError):
                # If conversion fails, use original spend
                pass

        model_name = ProxyBaseLLMRequestProcessing._get_deployment_model_name(litellm_logging_obj)

        headers = {
            "x-litellm-call-id": call_id,
            "x-litellm-model-id": model_id,
            "x-litellm-model-name": model_name,
            "x-litellm-cache-key": cache_key,
            "x-litellm-model-api-base": (
                api_base.split("?")[0] if api_base else None
            ),  # don't include query params, risk of leaking sensitive info
            "x-litellm-version": version,
            "x-litellm-model-region": model_region,
            "x-litellm-response-cost": str(response_cost),
            "x-litellm-response-cost-original": (str(original_cost) if original_cost is not None else None),
            "x-litellm-response-cost-discount-amount": (str(discount_amount) if discount_amount is not None else None),
            "x-litellm-response-cost-margin-amount": (
                str(margin_total_amount) if margin_total_amount is not None else None
            ),
            "x-litellm-response-cost-margin-percent": (str(margin_percent) if margin_percent is not None else None),
            "x-litellm-key-tpm-limit": str(user_api_key_dict.tpm_limit),
            "x-litellm-key-rpm-limit": str(user_api_key_dict.rpm_limit),
            "x-litellm-key-max-budget": str(user_api_key_dict.max_budget),
            "x-litellm-key-spend": str(updated_spend),
            "x-litellm-response-duration-ms": str(hidden_params.get("_response_ms", None)),
            "x-litellm-overhead-duration-ms": str(hidden_params.get("litellm_overhead_time_ms", None)),
            "x-litellm-callback-duration-ms": str(hidden_params.get("callback_duration_ms", None)),
            **(
                {
                    "x-litellm-timing-pre-processing-ms": str(hidden_params.get("timing_pre_processing_ms", None)),
                    "x-litellm-timing-llm-api-ms": str(hidden_params.get("timing_llm_api_ms", None)),
                    "x-litellm-timing-post-processing-ms": str(hidden_params.get("timing_post_processing_ms", None)),
                    "x-litellm-timing-message-copy-ms": str(hidden_params.get("timing_message_copy_ms", None)),
                }
                if LITELLM_DETAILED_TIMING
                else {}
            ),
            "x-litellm-fastest_response_batch_completion": (
                str(fastest_response_batch_completion) if fastest_response_batch_completion is not None else None
            ),
            "x-litellm-timeout": str(timeout) if timeout is not None else None,
            **{k: str(v) for k, v in kwargs.items()},
        }
        if request_data:
            remaining_tokens_header = get_remaining_tokens_and_requests_from_request_data(request_data)
            headers.update(remaining_tokens_header)

            logging_caching_headers = get_logging_caching_headers(request_data)
            if logging_caching_headers:
                headers.update(logging_caching_headers)

        try:
            return {key: str(value) for key, value in headers.items() if value not in exclude_values}
        except Exception as e:
            verbose_proxy_logger.error(f"Error setting custom headers: {e}")
            return {}

    @staticmethod
    async def build_litellm_proxy_success_headers_from_llm_response(
        *,
        response: Any,
        request_data: dict,
        request: Request,
        user_api_key_dict: UserAPIKeyAuth,
        logging_obj: LiteLLMLoggingObj,
        version: Optional[str],
        proxy_logging_obj: ProxyLogging,
    ) -> Dict[str, str]:
        """
        Build LiteLLM proxy response headers for routes that call the LLM directly
        (e.g. Google native :generateContent) instead of base_process_llm_request.
        """
        if isinstance(response, dict):
            hidden_params = get_hidden_params_dict(response)
        else:
            hidden_params = getattr(response, "_hidden_params", None) or {}
        if not isinstance(hidden_params, dict):
            hidden_params = {}

        model_id = ProxyBaseLLMRequestProcessing._get_model_id_from_response(hidden_params, request_data)

        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        fastest_response_batch_completion = hidden_params.get("fastest_response_batch_completion", None)
        additional_headers = hidden_params.get("additional_headers", {}) or {}

        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=logging_obj.litellm_call_id,
            model_id=model_id,
            cache_key=cache_key,
            api_base=api_base,
            version=version,
            response_cost=response_cost,
            model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            fastest_response_batch_completion=fastest_response_batch_completion,
            request_data=request_data,
            hidden_params=hidden_params,
            litellm_logging_obj=logging_obj,
            **additional_headers,
        )

        callback_headers = await proxy_logging_obj.post_call_response_headers_hook(
            data=request_data,
            user_api_key_dict=user_api_key_dict,
            response=response,
            request_headers=dict(request.headers),
        )
        if callback_headers:
            custom_headers.update(callback_headers)

        return custom_headers

    async def common_processing_pre_call_logic(
        self,
        request: Request,
        general_settings: dict,
        user_api_key_dict: UserAPIKeyAuth,
        proxy_logging_obj: ProxyLogging,
        proxy_config: ProxyConfig,
        route_type: Literal[
            "acompletion",
            "aembedding",
            "aresponses",
            "_arealtime",
            "_aresponses_websocket",
            "acreate_realtime_client_secret",
            "arealtime_calls",
            "aget_responses",
            "adelete_responses",
            "acancel_responses",
            "acompact_responses",
            "acreate_batch",
            "aretrieve_batch",
            "alist_batches",
            "acancel_batch",
            "afile_content",
            "afile_retrieve",
            "afile_delete",
            "atext_completion",
            "acreate_fine_tuning_job",
            "acancel_fine_tuning_job",
            "alist_fine_tuning_jobs",
            "aretrieve_fine_tuning_job",
            "alist_input_items",
            "aimage_edit",
            "agenerate_content",
            "agenerate_content_stream",
            "allm_passthrough_route",
            "avector_store_search",
            "avector_store_create",
            "avector_store_retrieve",
            "avector_store_list",
            "avector_store_update",
            "avector_store_delete",
            "avector_store_file_create",
            "avector_store_file_list",
            "avector_store_file_retrieve",
            "avector_store_file_content",
            "avector_store_file_update",
            "avector_store_file_delete",
            "aocr",
            "asearch",
            "avideo_generation",
            "avideo_list",
            "avideo_status",
            "avideo_content",
            "avideo_remix",
            "avideo_create_character",
            "avideo_get_character",
            "avideo_edit",
            "avideo_extension",
            "acreate_container",
            "alist_containers",
            "aingest",
            "aretrieve_container",
            "adelete_container",
            "aupload_container_file",
            "alist_container_files",
            "aretrieve_container_file",
            "adelete_container_file",
            "aretrieve_container_file_content",
            "acreate_skill",
            "alist_skills",
            "aget_skill",
            "adelete_skill",
            "anthropic_messages",
            "acreate_interaction",
            "aget_interaction",
            "adelete_interaction",
            "acancel_interaction",
            "acreate_agent",
            "alist_agents",
            "aget_agent",
            "adelete_agent",
            "alist_agent_versions",
            "asend_message",
            "call_mcp_tool",
            "acreate_eval",
            "alist_evals",
            "aget_eval",
            "aupdate_eval",
            "adelete_eval",
            "acancel_eval",
            "acreate_run",
            "alist_runs",
            "aget_run",
            "acancel_run",
            "adelete_run",
            "apply_guardrail",
        ],
        version: Optional[str] = None,
        user_model: Optional[str] = None,
        user_temperature: Optional[float] = None,
        user_request_timeout: Optional[float] = None,
        user_max_tokens: Optional[int] = None,
        user_api_base: Optional[str] = None,
        model: Optional[str] = None,
        llm_router: Optional[Router] = None,
    ) -> Tuple[dict, LiteLLMLoggingObj]:
        start_time = datetime.now()  # start before calling guardrail hooks

        self.data = await add_litellm_data_to_request(
            data=self.data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )
        if not general_settings.get("expose_fallback_errors_to_caller"):
            self.data.pop("include_fallback_errors", None)
        if route_type in {"aresponses", "_aresponses_websocket"}:
            await _authorize_response_file_search_vector_stores(
                data=self.data,
                user_api_key_dict=user_api_key_dict,
            )

        # Calculate request queue time after add_litellm_data_to_request
        # which sets arrival_time in proxy_server_request
        proxy_server_request = self.data.get("proxy_server_request", {})
        arrival_time = proxy_server_request.get("arrival_time")
        queue_time_seconds = None
        if arrival_time is not None:
            processing_start_time = time.time()
            queue_time_seconds = processing_start_time - arrival_time

        # Store queue time in metadata after add_litellm_data_to_request to ensure it's preserved
        if queue_time_seconds is not None:
            from litellm.proxy.litellm_pre_call_utils import _get_metadata_variable_name

            _metadata_variable_name = _get_metadata_variable_name(request)
            if _metadata_variable_name not in self.data:
                self.data[_metadata_variable_name] = {}
            if not isinstance(self.data[_metadata_variable_name], dict):
                self.data[_metadata_variable_name] = {}
            self.data[_metadata_variable_name]["queue_time_seconds"] = queue_time_seconds

        self.data["model"] = (
            general_settings.get("completion_model", None)  # server default
            or user_model  # model name passed via cli args
            or model  # for azure deployments
            or self.data.get("model", None)  # default passed in http request
        )

        # override with user settings, these are params passed via cli
        if user_temperature:
            self.data["temperature"] = user_temperature
        if user_request_timeout:
            self.data["request_timeout"] = user_request_timeout
        if user_max_tokens:
            self.data["max_tokens"] = user_max_tokens
        if user_api_base:
            self.data["api_base"] = user_api_base

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if isinstance(self.data["model"], str) and self.data["model"] in litellm.model_alias_map:
            self.data["model"] = litellm.model_alias_map[self.data["model"]]

        # Check key-specific aliases
        if (
            isinstance(self.data["model"], str)
            and user_api_key_dict.aliases
            and isinstance(user_api_key_dict.aliases, dict)
            and self.data["model"] in user_api_key_dict.aliases
        ):
            self.data["model"] = user_api_key_dict.aliases[self.data["model"]]

        self.data["litellm_call_id"] = request.headers.get("x-litellm-call-id", str(uuid.uuid4()))
        DDSpanTagger.tag_call_id(self.data.get("litellm_call_id"))
        DDSpanTagger.tag_request(
            user_api_key_dict=user_api_key_dict,
            requested_model=self.data.get("model"),
        )

        ### AUTO STREAM USAGE TRACKING ###
        # If always_include_stream_usage is enabled and this is a streaming request
        # automatically add stream_options={'include_usage': True} if not already set
        if (
            general_settings.get("always_include_stream_usage", False) is True
            and self.data.get("stream", False) is True
        ):
            # Only set if stream_options is not already provided by the client
            if "stream_options" not in self.data:
                self.data["stream_options"] = {"include_usage": True}
            elif isinstance(self.data["stream_options"], dict) and "include_usage" not in self.data["stream_options"]:
                self.data["stream_options"]["include_usage"] = True
        ### CALL HOOKS ### - modify/reject incoming data before calling the model

        ## LOGGING OBJECT ## - initialize logging object for logging success/failure events for call
        ## IMPORTANT Note: - initialize this before running pre-call checks. Ensures we log rejected requests to langfuse.
        logging_obj, self.data = litellm.utils.function_setup(
            original_function=route_type,
            rules_obj=litellm.utils.Rules(),
            start_time=start_time,
            **self.data,
        )

        self.data["litellm_logging_obj"] = logging_obj

        self.data = await proxy_logging_obj.pre_call_hook(  # type: ignore
            user_api_key_dict=user_api_key_dict,
            data=self.data,
            call_type=route_type,  # type: ignore
        )

        # Apply hierarchical router_settings (Key > Team)
        # Global router_settings are already on the Router object itself.
        if llm_router is not None and proxy_config is not None:
            from litellm.proxy.proxy_server import prisma_client

            router_settings = await proxy_config._get_hierarchical_router_settings(
                user_api_key_dict=user_api_key_dict,
                prisma_client=prisma_client,
                proxy_logging_obj=proxy_logging_obj,
            )

            # If router_settings found (from key or team), apply them
            # Pass settings as per-request overrides instead of creating a new Router
            # This avoids expensive Router instantiation on each request
            if router_settings is not None:
                self.data["router_settings_override"] = router_settings

        if "messages" in self.data and self.data["messages"]:
            logging_obj.update_messages(self.data["messages"])

        return self.data, logging_obj

    async def _pre_call_with_fallbacks(
        self,
        request: Request,
        general_settings: dict,
        proxy_logging_obj: ProxyLogging,
        user_api_key_dict: UserAPIKeyAuth,
        version: Optional[str],
        proxy_config: ProxyConfig,
        user_model: Optional[str],
        user_temperature: Optional[float],
        user_request_timeout: Optional[float],
        user_max_tokens: Optional[int],
        user_api_base: Optional[str],
        model: Optional[str],
        route_type: str,
        llm_router: Optional[Router],
    ) -> tuple[dict, LiteLLMLoggingObj]:
        from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError

        try:
            return await self.common_processing_pre_call_logic(
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
                model=model,
                route_type=route_type,
                llm_router=llm_router,
            )
        except ProxyRateLimitError as original_exc:
            original_model = self.data.get("model")
            if not original_model or not llm_router or self.data.get("disable_fallbacks"):
                raise

            fallback_models = self._resolve_fallback_models(
                model=original_model,
                llm_router=llm_router,
                user_api_key_dict=user_api_key_dict,
            )
            if not fallback_models:
                raise

            verbose_proxy_logger.info(
                "Local rate limit hit for model=%s, attempting fallbacks: %s",
                original_model,
                fallback_models,
            )

            try:
                for fallback_model in fallback_models:
                    if fallback_model == original_model:
                        continue
                    self.data["model"] = fallback_model
                    try:
                        return await self.common_processing_pre_call_logic(
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
                            model=fallback_model,
                            route_type=route_type,
                            llm_router=llm_router,
                        )
                    except ProxyRateLimitError:
                        continue
            except BaseException:
                self.data["model"] = original_model
                raise

            self.data["model"] = original_model
            raise original_exc

    def _resolve_fallback_models(
        self,
        model: str,
        llm_router: Router,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Optional[list]:
        from litellm.router_utils.fallback_event_handlers import get_fallback_model_group

        fallbacks = None

        key_router_settings = user_api_key_dict.router_settings
        if isinstance(key_router_settings, dict) and "fallbacks" in key_router_settings:
            fallbacks = key_router_settings["fallbacks"]

        if fallbacks is None:
            fallbacks = llm_router.fallbacks

        if not fallbacks:
            return None

        fallback_model_group, generic_fallback_idx = get_fallback_model_group(
            fallbacks=fallbacks,
            model_group=model,
        )
        if fallback_model_group is None and generic_fallback_idx is not None:
            fallback_model_group = fallbacks[generic_fallback_idx]["*"]
        return fallback_model_group

    @staticmethod
    def _get_model_id_from_response(hidden_params: dict, data: dict) -> str:
        """Extract model_id from hidden_params with fallback to litellm_metadata."""
        model_id = hidden_params.get("model_id", None) or ""
        if not model_id:
            litellm_metadata = data.get("litellm_metadata", {}) or {}
            model_info = litellm_metadata.get("model_info", {}) or {}
            model_id = model_info.get("id", "") or ""
        return model_id

    @staticmethod
    def _get_deployment_model_name(
        litellm_logging_obj: LiteLLMLoggingObj | None,
    ) -> str | None:
        """Extract the underlying deployment model string (e.g. ``azure/gpt-4o``).

        The router rewrites the response ``model`` field to the model-group alias
        the client requested, so neither the response body nor the existing
        headers expose the concrete deployment model. The router records it under
        ``litellm_params`` metadata as ``deployment``, so read it back from there.
        """
        litellm_params = getattr(litellm_logging_obj, "litellm_params", None)
        if not isinstance(litellm_params, dict):
            return None
        for key in ("litellm_metadata", "metadata"):
            metadata = litellm_params.get(key, {}) or {}
            deployment = metadata.get("deployment")
            if deployment:
                return deployment
        return None

    @staticmethod
    def _response_cost_from_logging_obj(
        *,
        response: Any,
        logging_obj: LiteLLMLoggingObj,
    ) -> float | str:
        """
        Recover the response cost when the response never recorded one in its
        ``_hidden_params``: Anthropic /v1/messages returns a TypedDict that cannot
        hold the attribute at all, and Google :generateContent carries
        ``_hidden_params`` but no synchronously-populated ``response_cost``. In both
        cases the cost is read back from the logging object instead, recomputing from
        the same calculator only when it has not been stored yet.
        """
        stored_cost = logging_obj.model_call_details.get("response_cost")
        if isinstance(stored_cost, (int, float)):
            return float(stored_cost)
        recomputed_cost = logging_obj._response_cost_calculator(result=response)
        return recomputed_cost if isinstance(recomputed_cost, (int, float)) else ""

    def _debug_log_request_payload(self) -> None:
        """Log request payload at DEBUG level, truncating if too large."""
        if not verbose_proxy_logger.isEnabledFor(logging.DEBUG):
            return
        _payload_str = json.dumps(self.data, default=str)
        if len(_payload_str) > MAX_PAYLOAD_SIZE_FOR_DEBUG_LOG:
            verbose_proxy_logger.debug(
                "Request received by LiteLLM: payload too large to log (%d bytes, limit %d). Keys: %s",
                len(_payload_str),
                MAX_PAYLOAD_SIZE_FOR_DEBUG_LOG,
                (list(self.data.keys()) if isinstance(self.data, dict) else type(self.data).__name__),
            )
        else:
            verbose_proxy_logger.debug(
                "Request received by LiteLLM:\n%s",
                _payload_str,
            )

    async def base_process_llm_request(
        self,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        route_type: Literal[
            "acompletion",
            "aembedding",
            "aresponses",
            "_arealtime",
            "_aresponses_websocket",
            "acreate_realtime_client_secret",
            "arealtime_calls",
            "aget_responses",
            "adelete_responses",
            "acancel_responses",
            "acompact_responses",
            "acreate_batch",
            "aretrieve_batch",
            "alist_batches",
            "acancel_batch",
            "afile_content",
            "afile_retrieve",
            "afile_delete",
            "atext_completion",
            "acreate_fine_tuning_job",
            "acancel_fine_tuning_job",
            "alist_fine_tuning_jobs",
            "aretrieve_fine_tuning_job",
            "alist_input_items",
            "aimage_edit",
            "agenerate_content",
            "agenerate_content_stream",
            "allm_passthrough_route",
            "avector_store_search",
            "avector_store_create",
            "avector_store_retrieve",
            "avector_store_list",
            "avector_store_update",
            "avector_store_delete",
            "avector_store_file_create",
            "avector_store_file_list",
            "avector_store_file_retrieve",
            "avector_store_file_content",
            "avector_store_file_update",
            "avector_store_file_delete",
            "aocr",
            "asearch",
            "avideo_generation",
            "avideo_list",
            "avideo_status",
            "avideo_content",
            "avideo_remix",
            "avideo_create_character",
            "avideo_get_character",
            "avideo_edit",
            "avideo_extension",
            "acreate_container",
            "alist_containers",
            "aingest",
            "aretrieve_container",
            "adelete_container",
            "aupload_container_file",
            "alist_container_files",
            "aretrieve_container_file",
            "adelete_container_file",
            "aretrieve_container_file_content",
            "acreate_skill",
            "alist_skills",
            "aget_skill",
            "adelete_skill",
            "anthropic_messages",
            "acreate_interaction",
            "aget_interaction",
            "adelete_interaction",
            "acancel_interaction",
            "acreate_agent",
            "alist_agents",
            "aget_agent",
            "adelete_agent",
            "alist_agent_versions",
            "asend_message",
            "call_mcp_tool",
            "acreate_eval",
            "alist_evals",
            "aget_eval",
            "aupdate_eval",
            "adelete_eval",
            "acancel_eval",
            "acreate_run",
            "alist_runs",
            "aget_run",
            "acancel_run",
            "adelete_run",
        ],
        proxy_logging_obj: ProxyLogging,
        general_settings: dict,
        proxy_config: ProxyConfig,
        select_data_generator: Optional[Callable] = None,
        llm_router: Optional[Router] = None,
        model: Optional[str] = None,
        user_model: Optional[str] = None,
        user_temperature: Optional[float] = None,
        user_request_timeout: Optional[float] = None,
        user_max_tokens: Optional[int] = None,
        user_api_base: Optional[str] = None,
        version: Optional[str] = None,
        is_streaming_request: Optional[bool] = False,
        contents: Optional[list] = None,  # Add contents parameter
        skip_pre_call_logic: bool = False,
    ) -> Any:
        """
        Common request processing logic for both chat completions and responses API endpoints
        """
        requested_model_from_client: Optional[str] = (
            self.data.get("model") if isinstance(self.data.get("model"), str) else None
        )
        self._debug_log_request_payload()

        if skip_pre_call_logic:
            logging_obj = self.data.get("litellm_logging_obj")
            if logging_obj is None:
                raise ValueError(
                    "skip_pre_call_logic=True requires litellm_logging_obj to be set in data. "
                    "Ensure common_processing_pre_call_logic was called before using this parameter."
                )
        else:
            self.data, logging_obj = await self._pre_call_with_fallbacks(
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
                model=model,
                route_type=route_type,
                llm_router=llm_router,
            )

        # Defer async logging when post-call guardrails are configured so the
        # StandardLoggingPayload is built after guardrails write to metadata.
        # Cache the result to avoid scanning litellm.callbacks twice.
        _post_call_guardrails_active = self._has_post_call_guardrails()

        # Non-streaming: defer the create_task in wrapper_async so the
        # SLP is built after guardrails write to metadata.  Streaming
        # uses a separate closure mechanism (see below).
        #
        # Edge case: if _is_streaming_request is False but the response
        # turns out to be a CustomStreamWrapper (rare provider behavior),
        # wrapper_async exits early before the _defer_async_logging block
        # so _enqueue_deferred_logging is never stored — the finally
        # block is a no-op.  The CSW path handles this correctly via
        # _on_deferred_stream_complete, which fires its own logging.
        if _post_call_guardrails_active and not self._is_streaming_request(
            data=self.data, is_streaming_request=is_streaming_request
        ):
            logging_obj._defer_async_logging = True  # type: ignore

        tasks = []
        # Start the moderation check (during_call_hook) as early as possible
        # This gives it a head start to mask/validate input while the proxy handles routing
        tasks.append(
            asyncio.create_task(
                proxy_logging_obj.during_call_hook(
                    data=self.data,
                    user_api_key_dict=user_api_key_dict,
                    call_type=route_type,  # type: ignore
                )
            )
        )

        # Pass contents if provided
        if contents:
            self.data["contents"] = contents

        ### ROUTE THE REQUEST ###
        # Do not change this - it should be a constant time fetch - ALWAYS
        llm_call = await route_request(
            data=self.data,
            route_type=route_type,
            llm_router=llm_router,
            user_model=user_model,
            user_api_key_dict=user_api_key_dict,
        )
        llm_call_task = asyncio.create_task(llm_call)
        tasks.append(llm_call_task)

        llm_responses = asyncio.gather(*tasks)  # run the moderation check in parallel to the actual llm api call

        try:
            if general_settings.get("cancel_on_disconnect", False):
                responses = await _await_llm_call_cancelling_on_disconnect(request, llm_responses)
            else:
                responses = await llm_responses
        finally:
            await _cancel_pending_gather_tasks(tasks)

        response = responses[1]

        # GH#30566: set overhead duration for non-chat-completions
        # routes (/v1/messages, /v1/responses). Chat completions
        # already have litellm_overhead_time_ms from the SDK.
        # Only set the overhead field directly (don't call
        # update_response_metadata which also touches cost).
        _overhead_hidden_params = getattr(response, "_hidden_params", {}) or {}
        if not _overhead_hidden_params.get("litellm_overhead_time_ms") and route_type not in (
            "acompletion",
            "completion",
        ):
            end_time = datetime.now()
            _logging_obj = self.data.get("litellm_logging_obj")
            if _logging_obj is not None and _logging_obj.start_time is not None:
                overhead_ms = (end_time - _logging_obj.start_time).total_seconds() * 1000 - (
                    _logging_obj.model_call_details.get("llm_api_duration_ms", 0)
                )
                if not isinstance(_overhead_hidden_params, dict):
                    _overhead_hidden_params = {}
                _overhead_hidden_params["litellm_overhead_time_ms"] = overhead_ms
                if hasattr(response, "_hidden_params"):
                    response._hidden_params = _overhead_hidden_params
                elif isinstance(response, dict):
                    response["_hidden_params"] = _overhead_hidden_params

        _exception_raised = False
        try:
            hidden_params = get_hidden_params_dict(response)
            model_id = self._get_model_id_from_response(hidden_params, self.data)

            cache_key, api_base, response_cost = (
                hidden_params.get("cache_key", None) or "",
                hidden_params.get("api_base", None) or "",
                hidden_params.get("response_cost", None) or "",
            )
            fastest_response_batch_completion, additional_headers = (
                hidden_params.get("fastest_response_batch_completion", None),
                hidden_params.get("additional_headers", {}) or {},
            )

            # Post Call Processing
            if llm_router is not None:
                self.data["deployment"] = llm_router.get_deployment(model_id=model_id)
            asyncio.create_task(
                proxy_logging_obj.update_request_status(
                    litellm_call_id=self.data.get("litellm_call_id", ""),
                    status="success",
                )
            )
            if self._is_streaming_request(
                data=self.data, is_streaming_request=is_streaming_request
            ) or self._is_streaming_response(response):  # use generate_responses to stream responses
                custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
                    user_api_key_dict=user_api_key_dict,
                    call_id=logging_obj.litellm_call_id,
                    model_id=model_id,
                    cache_key=cache_key,
                    api_base=api_base,
                    version=version,
                    response_cost=response_cost,
                    model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                    fastest_response_batch_completion=fastest_response_batch_completion,
                    request_data=self.data,
                    hidden_params=hidden_params,
                    litellm_logging_obj=logging_obj,
                    **additional_headers,
                )

                # Call response headers hook for streaming success
                callback_headers = await proxy_logging_obj.post_call_response_headers_hook(
                    data=self.data,
                    user_api_key_dict=user_api_key_dict,
                    response=response,
                    request_headers=dict(request.headers),
                )
                if callback_headers:
                    custom_headers.update(callback_headers)

                # Preserve the original client-requested model (pre-alias mapping) for downstream
                # streaming generators. Pre-call processing can rewrite `self.data["model"]` for
                # aliasing/routing, but the OpenAI-compatible response `model` field should reflect
                # what the client sent.
                if requested_model_from_client:
                    self.data["_litellm_client_requested_model"] = requested_model_from_client

                # Streaming: attach a closure that fires after all guardrail
                # end-of-stream blocks complete.  CSW.__anext__ stores the
                # assembled response on logging_obj; the outer consumer
                # (ProxyLogging._fire_deferred_stream_logging) fires the
                # closure after the full streaming pipeline finishes.
                # The closure runs non-apply_guardrail hooks on the
                # assembled response, then fires success logging.
                # Only for CustomStreamWrapper — raw async generators from
                # passthrough routes bypass CSW and would orphan the closure.
                from litellm.litellm_core_utils.streaming_handler import (
                    CustomStreamWrapper,
                )

                if _post_call_guardrails_active and isinstance(response, CustomStreamWrapper):
                    # Intentionally a live reference (not a copy) — mirrors
                    # ProxyLogging.post_call_success_hook which also mutates
                    # data["guardrail_to_apply"] during iteration.
                    _captured_data = self.data
                    _captured_user_api_key_dict = user_api_key_dict
                    _captured_logging_obj = logging_obj

                    async def _on_deferred_stream_complete(assembled_response, cache_hit):
                        await ProxyBaseLLMRequestProcessing._run_deferred_stream_guardrails(
                            captured_data=_captured_data,
                            captured_user_api_key_dict=_captured_user_api_key_dict,
                            captured_logging_obj=_captured_logging_obj,
                            assembled_response=assembled_response,
                            cache_hit=cache_hit,
                        )

                    logging_obj._on_deferred_stream_complete = _on_deferred_stream_complete  # type: ignore[union-attr]

                if route_type == "allm_passthrough_route":
                    # Check if response is an async generator
                    if self._is_streaming_response(response):
                        if asyncio.iscoroutine(response):
                            generator = await response
                        else:
                            generator = response

                        if (
                            self._has_post_call_guardrails_for_passthrough()
                            and self._passthrough_endpoint_has_stream_guardrail_handler()
                        ):
                            body_bytes = b"".join(
                                [chunk async for chunk in generator]  # type: ignore[union-attr]
                            )
                            modified_bytes = await self._handle_event_stream_allm_passthrough_route(
                                body_bytes=body_bytes,
                                proxy_logging_obj=proxy_logging_obj,
                                user_api_key_dict=user_api_key_dict,
                            )
                            response_headers = {
                                k: v for k, v in custom_headers.items() if k.lower() != "content-length"
                            }
                            return Response(
                                content=modified_bytes,
                                status_code=status.HTTP_200_OK,
                                media_type=self._passthrough_event_stream_media_type(),
                                headers=response_headers,
                            )

                        # For passthrough routes, stream directly without error parsing
                        # since we're dealing with raw binary data (e.g., AWS event streams)
                        return StreamingResponse(
                            content=generator,  # type: ignore[arg-type]
                            status_code=status.HTTP_200_OK,
                            headers=custom_headers,
                        )
                    else:
                        _early = await self._handle_non_streaming_allm_passthrough_route(
                            response=response,
                            proxy_logging_obj=proxy_logging_obj,
                            user_api_key_dict=user_api_key_dict,
                            custom_headers=custom_headers,
                            request_headers=dict(request.headers),
                        )
                        if _early is not None:
                            return _early
                        return StreamingResponse(
                            content=response.aiter_bytes(),  # type: ignore[union-attr]
                            status_code=response.status_code,  # type: ignore[union-attr]
                            headers=custom_headers,
                        )
                elif route_type == "anthropic_messages":
                    # Check if response is actually a streaming response (async generator)
                    # Non-streaming responses (dict) should be returned directly
                    # This handles cases like websearch_interception agentic loop
                    # which returns a non-streaming dict even for streaming requests
                    if self._is_streaming_response(response):
                        selected_data_generator = ProxyBaseLLMRequestProcessing.async_sse_data_generator(
                            response=response,
                            user_api_key_dict=user_api_key_dict,
                            request_data=self.data,
                            proxy_logging_obj=proxy_logging_obj,
                            request=request,
                        )
                        return await create_response(
                            generator=selected_data_generator,
                            media_type="text/event-stream",
                            headers=custom_headers,
                            request=request,
                        )
                    # Non-streaming response - fall through to normal response handling
                elif select_data_generator:
                    selected_data_generator = select_data_generator(
                        response=response,
                        user_api_key_dict=user_api_key_dict,
                        request_data=self.data,
                        request=request,
                    )
                    if route_type == "aresponses":
                        # Streaming /v1/responses returns here without
                        # reaching the non-streaming ownership tail below.
                        # Wrap the SSE generator so container ownership is
                        # written once the upstream iterator finishes
                        # assembling ``completed_response`` — otherwise
                        # code-interpreter containers created during the
                        # stream stay unregistered and follow-up file API
                        # calls 403. Covers the background-polling path
                        # too, which loops ``body_iterator`` end-to-end.
                        selected_data_generator = (
                            ProxyBaseLLMRequestProcessing._wrap_responses_stream_for_container_ownership(
                                original_stream_response=response,
                                wrapped_generator=selected_data_generator,
                                user_api_key_dict=user_api_key_dict,
                            )
                        )
                    return await create_response(
                        generator=selected_data_generator,
                        media_type="text/event-stream",
                        headers=custom_headers,
                        request=request,
                    )

            ### CALL HOOKS ### - modify outgoing data
            # If we reach here with a streaming closure still set, it means
            # no early-return route consumed the CSW (hypothetical fallthrough).
            # Clear the closure so guardrails run inline as before — this
            # preserves blocking behavior and avoids double invocation.
            if getattr(logging_obj, "_on_deferred_stream_complete", None):
                logging_obj._on_deferred_stream_complete = None  # type: ignore[union-attr]

            if route_type == "allm_passthrough_route":
                _non_streaming_custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
                    user_api_key_dict=user_api_key_dict,
                    call_id=logging_obj.litellm_call_id,
                    model_id=model_id,
                    cache_key=cache_key,
                    api_base=api_base,
                    version=version,
                    response_cost=response_cost,
                    model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                    fastest_response_batch_completion=fastest_response_batch_completion,
                    request_data=self.data,
                    hidden_params=hidden_params,
                    litellm_logging_obj=logging_obj,
                    **additional_headers,
                )
                _early = await self._handle_non_streaming_allm_passthrough_route(
                    response=response,
                    proxy_logging_obj=proxy_logging_obj,
                    user_api_key_dict=user_api_key_dict,
                    custom_headers=_non_streaming_custom_headers,
                    request_headers=dict(request.headers),
                )
                if _early is not None:
                    return _early

            response = await proxy_logging_obj.post_call_success_hook(
                data=self.data,
                user_api_key_dict=user_api_key_dict,
                response=response,  # type: ignore[arg-type]
            )
        except Exception:
            _exception_raised = True
            raise
        finally:
            ProxyBaseLLMRequestProcessing._flush_deferred_async_logging(
                logging_obj=logging_obj,
                exception_raised=_exception_raised,
            )

            # Streaming cleanup: if an exception occurred AND the deferred
            # streaming closure is still set, no streaming route will
            # consume the CSW — the closure is orphaned.  Clear it and
            # fire logging directly to avoid silent loss.
            #
            # On normal streaming returns the closure must stay: CSW calls
            # it at stream end.  _exception_raised is function-scoped and
            # immune to outer exception context, avoiding false positives.
            if _exception_raised:
                _deferred_fn = getattr(logging_obj, "_on_deferred_stream_complete", None)
                if _deferred_fn is not None:
                    logging_obj._on_deferred_stream_complete = None  # type: ignore[union-attr]
                    try:
                        asyncio.create_task(
                            logging_obj.dispatch_success_handlers(
                                response,
                                cache_hit=None,
                                start_time=None,
                                end_time=None,
                                prefer_async_handlers=True,
                            )
                        )
                    except Exception as e:
                        verbose_proxy_logger.exception("Error in orphaned streaming async logging: %s", e)

        # Always return the client-requested model name (not provider-prefixed internal identifiers)
        # for OpenAI-compatible responses.
        if requested_model_from_client:
            _override_openai_response_model(
                response_obj=response,
                requested_model=requested_model_from_client,
                log_context=f"litellm_call_id={logging_obj.litellm_call_id}",
                return_raw_model_name=_should_return_raw_model_name(self.data),
            )

        hidden_params = get_hidden_params_dict(response)  # get any updated response headers
        additional_headers = hidden_params.get("additional_headers", {}) or {}

        recover_response_cost = not response_cost and hidden_params.get("response_cost") is None
        response_cost_for_headers = (
            self._response_cost_from_logging_obj(response=response, logging_obj=logging_obj) or ""
            if recover_response_cost
            else response_cost
        )

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                call_id=logging_obj.litellm_call_id,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost_for_headers,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                fastest_response_batch_completion=fastest_response_batch_completion,
                request_data=self.data,
                hidden_params=hidden_params,
                litellm_logging_obj=logging_obj,
                **additional_headers,
            )
        )

        if isinstance(response, dict):
            response.pop("_hidden_params", None)

        # Call response headers hook for non-streaming success
        callback_headers = await proxy_logging_obj.post_call_response_headers_hook(
            data=self.data,
            user_api_key_dict=user_api_key_dict,
            response=response,
            request_headers=dict(request.headers),
        )
        if callback_headers:
            fastapi_response.headers.update(callback_headers)

        await check_response_size_is_safe(response=response)

        if route_type in {"aresponses", "aget_responses"}:
            await ProxyBaseLLMRequestProcessing._record_container_owners_from_responses_if_needed(
                response=response,
                user_api_key_dict=user_api_key_dict,
            )

        return response

    @staticmethod
    async def _record_container_owners_from_responses_if_needed(
        response: Any,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        """Register code-interpreter containers so follow-up file APIs pass ownership checks."""
        from litellm.proxy.container_endpoints.ownership import (
            record_container_owners_from_responses_response,
        )

        if response is None:
            return

        try:
            await record_container_owners_from_responses_response(
                response=response,
                user_api_key_dict=user_api_key_dict,
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                "Container ownership recording failed after responses call: %s",
                e,
            )

    @staticmethod
    def _extract_completed_responses_response(stream_response: Any) -> Any:
        """Pull the assembled ``ResponsesAPIResponse`` off a streaming iterator.

        ``ResponsesAPIStreamingIterator`` stores the terminal stream event
        (``response.completed`` / ``response.incomplete`` / ``response.failed``)
        in ``completed_response``; the actual response body hangs off
        that event's ``.response`` attribute. Some iterators store the
        ``ResponsesAPIResponse`` directly. Handle both shapes so the
        container-ownership recording path can walk ``.output`` either way.
        """
        completed = getattr(stream_response, "completed_response", None)
        if completed is None:
            return None
        response_obj = getattr(completed, "response", None)
        if response_obj is not None:
            return response_obj
        return completed

    @staticmethod
    async def _wrap_responses_stream_for_container_ownership(
        original_stream_response: Any,
        wrapped_generator: Any,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        """Forward SSE chunks, then record container ownership at stream end.

        Streaming ``/v1/responses`` short-circuits out of
        ``base_process_llm_request`` before the non-streaming ownership
        tail runs, so without this wrap the
        ``LiteLLM_ManagedObjectTable`` row for any container created
        during the stream is never written and follow-up file API calls
        return 403.
        """
        try:
            async for chunk in wrapped_generator:
                yield chunk
        finally:
            try:
                completed_obj = ProxyBaseLLMRequestProcessing._extract_completed_responses_response(
                    original_stream_response
                )
                if completed_obj is not None:
                    await ProxyBaseLLMRequestProcessing._record_container_owners_from_responses_if_needed(
                        response=completed_obj,
                        user_api_key_dict=user_api_key_dict,
                    )
                else:
                    # Silent skip caused #30210: the proxy's Router wrapper
                    # of the responses streaming iterator wasn't propagating
                    # ``completed_response``, so this hook recorded nothing
                    # and follow-up /v1/containers/<id>/files calls 403'd
                    # for non-admin keys with no proxy-side hint. Log a
                    # warning so future regressions of the same shape
                    # surface in operator logs.
                    verbose_proxy_logger.warning(
                        "Container ownership recording skipped on streaming "
                        "/v1/responses: no completed_response on stream "
                        "iterator %s. If this stream created any tool "
                        "container (e.g. code_interpreter), follow-up "
                        "/v1/containers/<id>/files calls will 403 for "
                        "non-admin keys.",
                        type(original_stream_response).__name__,
                    )
            except Exception as e:
                verbose_proxy_logger.exception(
                    "Container ownership recording failed after streaming responses call: %s",
                    e,
                )

    async def base_passthrough_process_llm_request(
        self,
        request: Request,
        fastapi_response: Response,
        user_api_key_dict: UserAPIKeyAuth,
        proxy_logging_obj: ProxyLogging,
        general_settings: dict,
        proxy_config: ProxyConfig,
        select_data_generator: Callable,
        llm_router: Optional[Router] = None,
        model: Optional[str] = None,
        user_model: Optional[str] = None,
        user_temperature: Optional[float] = None,
        user_request_timeout: Optional[float] = None,
        user_max_tokens: Optional[int] = None,
        user_api_base: Optional[str] = None,
        version: Optional[str] = None,
    ):
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            HttpPassThroughEndpointHelpers,
        )

        result = await self.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="allm_passthrough_route",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=model,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )

        # Check if result is actually a streaming response by inspecting its type
        if isinstance(result, StreamingResponse):
            return result

        # base_process_llm_request may return a FastAPI Response directly after
        # post-call guardrails buffer and rewrite JSON (e.g. Bedrock Converse passthrough).
        if isinstance(result, Response):
            return result

        content = await result.aread()
        return Response(
            content=content,
            status_code=result.status_code,
            headers=HttpPassThroughEndpointHelpers.get_response_headers(
                headers=result.headers,
                custom_headers=dict(fastapi_response.headers),
            ),
        )

    def _is_streaming_response(self, response: Any) -> bool:
        """
        Check if the response object is actually a streaming response by inspecting its type.

        This uses standard Python inspection to detect streaming/async iterator objects
        rather than relying on specific wrapper classes.
        """
        import inspect
        from collections.abc import AsyncGenerator, AsyncIterator

        # Check if it's an async generator (most reliable)
        if inspect.isasyncgen(response):
            return True

        # Check if it implements the async iterator protocol
        if isinstance(response, (AsyncIterator, AsyncGenerator)):
            return True

        return False

    def _is_streaming_request(self, data: dict, is_streaming_request: Optional[bool] = False) -> bool:
        """
        Check if the request is a streaming request.

        1. is_streaming_request is a dynamic param passed in
        2. if "stream" in data and data["stream"] is True
        """
        if is_streaming_request is True:
            return True
        if "stream" in data and data["stream"] is True:
            return True
        return False

    @staticmethod
    def _has_post_call_guardrails() -> bool:
        """
        True when a guardrail explicitly registers post_call. event_hook=None
        matches all hooks in should_run_guardrail but must not defer async logging
        on non-streaming /chat/completions (no post_call_success_hook flush path).
        """
        for cb in litellm.callbacks:
            if not isinstance(cb, CustomGuardrail):
                continue
            if cb.event_hook is None:
                continue
            if cb._event_hook_is_event_type(GuardrailEventHooks.post_call):
                return True
        return False

    def _has_post_call_guardrails_for_passthrough(self) -> bool:
        """
        True when a post_call guardrail will actually run for THIS request.

        Mirrors the gate in ProxyLogging.post_call_success_hook
        (should_run_guardrail against the request's merged guardrails) so that a
        guardrail registered globally but not configured for this key/team does
        not force the passthrough stream to be buffered into a single
        non-streaming response. An event_hook=None guardrail still counts here
        because should_run_guardrail treats it as matching every hook.
        """
        from litellm.proxy.proxy_server import llm_router
        from litellm.proxy.utils import _check_and_merge_model_level_guardrails

        guardrail_data = _check_and_merge_model_level_guardrails(data=self.data, llm_router=llm_router)
        for cb in litellm.callbacks:
            if not isinstance(cb, CustomGuardrail):
                continue
            if cb.should_run_guardrail(
                data=guardrail_data,
                event_type=GuardrailEventHooks.post_call,
            ):
                return True
        return False

    def _passthrough_endpoint_has_stream_guardrail_handler(self) -> bool:
        """
        True when the resolved passthrough provider AND endpoint have an
        event-stream guardrail handler that can rewrite buffered frames. Only such
        endpoints may have their stream buffered for post-call guardrails; every
        other endpoint must keep streaming so the response is not silently turned
        into a non-streaming body when no content modification would occur (e.g.
        Bedrock invoke-with-response-stream, whose frames the Converse handler
        leaves untouched).
        """
        from litellm.llms.pass_through.guardrail_translation.handler import (
            LlmPassthroughRouteHandler,
        )

        return LlmPassthroughRouteHandler.supports_event_stream_de_anonymization(
            self.data.get("custom_llm_provider"),
            self.data.get("endpoint"),
        )

    def _passthrough_event_stream_media_type(self) -> Optional[str]:
        """
        Content-type for a buffered passthrough event-stream response, resolved
        from the provider handler so the proxy stays provider-agnostic. Mirrors
        the upstream content-type the non-streaming path forwards, since the
        buffered streaming generator carries no headers of its own.
        """
        from litellm.llms.pass_through.guardrail_translation.handler import (
            LlmPassthroughRouteHandler,
        )

        return LlmPassthroughRouteHandler.event_stream_media_type(self.data.get("custom_llm_provider"))

    async def _handle_non_streaming_allm_passthrough_route(
        self,
        response: Any,
        proxy_logging_obj: "ProxyLogging",
        user_api_key_dict: "UserAPIKeyAuth",
        custom_headers: dict,
        request_headers: Dict[str, str],
    ) -> Optional[Response]:
        if not self._has_post_call_guardrails_for_passthrough():
            return None

        import json as _json

        from litellm.llms.pass_through.guardrail_translation.handler import (
            LlmPassthroughRouteHandler,
        )
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            HttpPassThroughEndpointHelpers,
        )

        try:
            response_status: int = response.status_code  # type: ignore[union-attr]
            content_type: str = response.headers.get("content-type", "")  # type: ignore[union-attr]
        except AttributeError:
            return None

        if response_status >= 300:
            return None

        is_event_stream = LlmPassthroughRouteHandler.is_event_stream_response(
            self.data.get("custom_llm_provider"), content_type
        )
        if not is_event_stream and "application/json" not in content_type:
            return None

        response_headers = HttpPassThroughEndpointHelpers.get_response_headers(
            headers=response.headers,  # type: ignore[union-attr]
            custom_headers=custom_headers,
        )
        callback_headers = await proxy_logging_obj.post_call_response_headers_hook(
            data=self.data,
            user_api_key_dict=user_api_key_dict,
            response=response,
            request_headers=request_headers,
        )
        if callback_headers:
            response_headers.update(callback_headers)

        if is_event_stream:
            body_bytes = await response.aread()  # type: ignore[union-attr]
            modified_bytes = await self._handle_event_stream_allm_passthrough_route(
                body_bytes=body_bytes,
                proxy_logging_obj=proxy_logging_obj,
                user_api_key_dict=user_api_key_dict,
            )
            return Response(
                content=modified_bytes,
                status_code=response_status,
                media_type=content_type,
                headers=response_headers,
            )

        body_bytes = await response.aread()  # type: ignore[union-attr]
        try:
            parsed = _json.loads(body_bytes)
        except (_json.JSONDecodeError, UnicodeDecodeError):
            return Response(
                content=body_bytes,
                status_code=response_status,
                media_type="application/json",
                headers=response_headers,
            )
        processed = await proxy_logging_obj.post_call_success_hook(
            data=self.data,
            user_api_key_dict=user_api_key_dict,
            response=parsed,
        )
        if isinstance(processed, dict):
            content = _json.dumps(processed).encode()
        else:
            verbose_proxy_logger.debug(
                "allm_passthrough_route: post_call_success_hook returned %s, leaving JSON response unmodified",
                type(processed).__name__,
            )
            content = body_bytes
        return Response(
            content=content,
            status_code=response_status,
            media_type="application/json",
            headers=response_headers,
        )

    async def _handle_event_stream_allm_passthrough_route(
        self,
        body_bytes: bytes,
        proxy_logging_obj: "ProxyLogging",
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bytes:
        from litellm.llms.pass_through.guardrail_translation.handler import (
            LlmPassthroughRouteHandler,
        )

        return await LlmPassthroughRouteHandler.de_anonymize_event_stream(
            body_bytes=body_bytes,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_dict=user_api_key_dict,
            data=self.data,
        )

    @staticmethod
    def _flush_deferred_async_logging(
        logging_obj: Any,
        exception_raised: bool,
    ) -> None:
        """
        Fire the deferred async-success closure stored by wrapper_async, then
        clear the slot.

        Called from the finally block around post_call_success_hook so the
        StandardLoggingPayload is built after post-call guardrails write to
        metadata (deferred logging is enabled for non-streaming requests with
        a registered post_call guardrail).

        On exception (e.g. a post-call guardrail blocks the response), skip
        firing the closure — the exception propagates to post_call_failure_hook
        which writes its own failure spend log via async_failure_handler.
        Firing both produced a duplicate (Success + Failure) entry per request,
        with the Success row exposing the blocked LLM response.

        For streaming early-returns the closure is never stored (wrapper_async
        returns before the deferred block in litellm/utils.py), so this is a
        no-op there.

        Extracted as a static method so tests can exercise the production
        gating logic directly rather than reimplementing the finally block.
        """
        _enqueue_fn = getattr(logging_obj, "_enqueue_deferred_logging", None)
        if _enqueue_fn is None:
            return
        logging_obj._enqueue_deferred_logging = None  # type: ignore[union-attr]
        if exception_raised:
            return
        try:
            _enqueue_fn()
        except Exception as e:
            verbose_proxy_logger.exception("Error firing deferred logging: %s", e)

    @staticmethod
    async def _run_deferred_stream_guardrails(
        captured_data: dict,
        captured_user_api_key_dict: "UserAPIKeyAuth",
        captured_logging_obj: Any,
        assembled_response: Any,
        cache_hit: Any,
    ) -> None:
        """
        Run non-streaming post-call guardrail hooks on an assembled streaming
        response, then fire success logging via ``dispatch_success_handlers``.

        Called by ProxyLogging._fire_deferred_stream_logging after the full
        streaming pipeline (including unified_guardrail end-of-stream blocks)
        has completed.

        Guardrails with apply_guardrail are skipped — they already ran via
        unified_guardrail's streaming iterator.  Only guardrails that override
        async_post_call_success_hook directly (without apply_guardrail) run
        here.

        This is audit-only — content has already been delivered to the client.

        Extracted as a static method so tests can call the production
        implementation directly rather than reimplementing the closure.
        """
        _response = assembled_response
        try:
            from litellm.proxy.proxy_server import llm_router as _global_llm_router
            from litellm.proxy.utils import _check_and_merge_model_level_guardrails

            guardrail_data = _check_and_merge_model_level_guardrails(data=captured_data, llm_router=_global_llm_router)
            for cb in litellm.callbacks:
                if not isinstance(cb, CustomGuardrail):
                    continue
                if not cb.should_run_guardrail(
                    data=guardrail_data,
                    event_type=GuardrailEventHooks.post_call,
                ):
                    continue
                try:
                    guardrail_result = None
                    if "apply_guardrail" in type(cb).__dict__:
                        # Skip — apply_guardrail guardrails already ran via
                        # unified_guardrail's end-of-stream block in the
                        # streaming iterator pipeline.  Running them again
                        # here would duplicate the guardrail API call
                        # (e.g. double OpenAI Moderation charges).
                        continue
                    if "async_post_call_streaming_iterator_hook" in type(cb).__dict__:
                        # Skip — the guardrail already scanned the assembled
                        # response via its own streaming iterator hook in the
                        # streaming pipeline. re running this function async_post_call_success_hook
                        # here would duplicate the scan and can spuriously block the guardrail that already passed / failed.
                        continue
                    else:
                        guardrail_result = await cb.async_post_call_success_hook(
                            user_api_key_dict=captured_user_api_key_dict,
                            data=guardrail_data,
                            response=_response,
                        )
                    if guardrail_result is not None:
                        _response = guardrail_result
                except Exception as e:
                    verbose_proxy_logger.exception(
                        "Error running post-call guardrail %s on streaming response: %s",
                        getattr(cb, "guardrail_name", type(cb).__name__),
                        e,
                    )
                    if isinstance(e, HTTPException) and hasattr(captured_logging_obj, "model_call_details"):
                        captured_logging_obj.model_call_details.setdefault("metadata", {})["guardrail_blocked"] = True
        except Exception as e:
            verbose_proxy_logger.exception(
                "Error in deferred streaming guardrail initialization: %s",
                e,
            )
        finally:
            try:
                # Proxy streaming always runs in async context and proxy spend
                # logging is async-only; force async dispatch so DB/spend
                # callbacks fire regardless of the call-type heuristic in
                # _is_sync_litellm_request (which only recognizes a subset of
                # async markers stored in litellm_params).
                asyncio.create_task(
                    captured_logging_obj.dispatch_success_handlers(
                        _response,
                        cache_hit=cache_hit,
                        start_time=None,
                        end_time=None,
                        prefer_async_handlers=True,
                    )
                )
            except Exception as e:
                verbose_proxy_logger.exception(
                    "Error in deferred streaming success logging: %s",
                    e,
                )

    def _apply_router_cooldown_retry_after(self, headers: dict, e: Exception) -> None:
        if isinstance(e, RouterRateLimitError) and e.cooldown_time > 0:
            headers["retry-after"] = str(math.ceil(e.cooldown_time))

    async def _handle_llm_api_exception(
        self,
        e: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        proxy_logging_obj: ProxyLogging,
        version: Optional[str] = None,
    ):
        """Raises ProxyException (OpenAI API compatible) if an exception is raised"""
        _log_llm_api_exception(e)
        # Allow callbacks to transform the error response
        transformed_exception = await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict,
            original_exception=e,
            request_data=self.data,
        )
        # Use transformed exception if callback returned one, otherwise use original
        if transformed_exception is not None:
            e = transformed_exception
        litellm_debug_info = getattr(e, "litellm_debug_info", "")
        verbose_proxy_logger.debug(
            "\033[1;31mAn error occurred: %s %s\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`",
            e,
            litellm_debug_info,
        )

        timeout = getattr(
            e, "timeout", None
        )  # returns the timeout set by the wrapper. Used for testing if model-specific timeout are set correctly
        _litellm_logging_obj: Optional[LiteLLMLoggingObj] = self.data.get("litellm_logging_obj", None)

        # Attempt to get model_id from logging object
        #
        # Note: We check the direct model_info path first (not nested in metadata) because that's where the router sets it.
        # The nested metadata path is only a fallback for cases where model_info wasn't set at the top level.
        model_id = self.maybe_get_model_id(_litellm_logging_obj)

        custom_headers = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=(
                _litellm_logging_obj.litellm_call_id if _litellm_logging_obj else self.data.get("litellm_call_id")
            ),
            model_id=model_id,
            version=version,
            response_cost=0,
            model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
            request_data=self.data,
            timeout=timeout,
            litellm_logging_obj=_litellm_logging_obj,
        )
        # Extract headers from exception - check both e.headers and e.response.headers
        headers = getattr(e, "headers", None) or {}
        if not headers:
            # Try to get headers from e.response.headers (httpx.Response)
            _response = getattr(e, "response", None)
            if _response is not None:
                _response_headers = getattr(_response, "headers", None)
                if _response_headers:
                    headers = get_response_headers(dict(_response_headers))
        headers.update(custom_headers)

        # Call response headers hook for failure
        try:
            callback_headers = await proxy_logging_obj.post_call_response_headers_hook(
                data=self.data,
                user_api_key_dict=user_api_key_dict,
                response=None,
                request_headers=(self.data.get("proxy_server_request") or {}).get("headers", {}),
            )
            if callback_headers:
                headers.update(callback_headers)
        except Exception:
            pass

        self._apply_router_cooldown_retry_after(headers, e)

        if isinstance(e, ProxyException):
            e.headers = {
                **e.headers,
                **{k: v if isinstance(v, str) else str(v) for k, v in headers.items()},
            }
            raise e

        if isinstance(e, HTTPException):
            raw_detail = getattr(e, "detail", str(e))
            message, structured_fields = _serialize_http_exception_detail(raw_detail)
            existing_fields = getattr(e, "provider_specific_fields", None) or {}
            if structured_fields:
                merged_fields: Optional[dict] = {**existing_fields, **structured_fields}
            else:
                merged_fields = existing_fields or None
            raise ProxyException(
                message=message,
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
                provider_specific_fields=merged_fields,
                headers=headers,
            )
        elif isinstance(e, httpx.HTTPStatusError):
            # Handle httpx.HTTPStatusError - extract actual error from response
            # This matches the original behavior before the refactor in commit 511d435f6f
            http_status_error: httpx.HTTPStatusError = e
            error_body = await http_status_error.response.aread()
            error_text = error_body.decode("utf-8")

            raise HTTPException(
                status_code=http_status_error.response.status_code,
                detail={"error": error_text},
            )
        error_msg = f"{str(e)}"
        # Check for AttributeError in the exception chain.
        # The AttributeError may be wrapped in multiple layers
        # (e.g. AttributeError -> OpenAIException -> APIConnectionError),
        # so walk __cause__, __context__, and original_exception recursively.
        has_attribute_error = _has_attribute_error_in_chain(e)

        if has_attribute_error:
            raise ProxyException(
                message=f"Invalid request format: {error_msg}",
                type="invalid_request_error",
                param=None,
                code=status.HTTP_400_BAD_REQUEST,
                headers=headers,
            )
        # Extract status_code from the exception if it carries one.
        # Provider exceptions (NotFoundError, BadRequestError, GeminiError,
        # VertexAIError, etc.) all have a status_code attribute reflecting
        # the upstream API response. Use it to return the correct HTTP code
        # instead of defaulting to 500.
        _exc_status_code = getattr(e, "status_code", None)
        if _exc_status_code is not None and isinstance(_exc_status_code, int) and 400 <= _exc_status_code <= 599:
            _code = _exc_status_code
        else:
            _code = status.HTTP_500_INTERNAL_SERVER_ERROR
        raise ProxyException(
            message=getattr(e, "message", error_msg),
            type=getattr(e, "type", "None"),
            param=getattr(e, "param", "None"),
            openai_code=getattr(e, "code", None),
            code=_code,
            provider_specific_fields=getattr(e, "provider_specific_fields", None),
            headers=headers,
        )

    #########################################################
    # Proxy Level Streaming Data Generator
    #########################################################

    @staticmethod
    def return_sse_chunk(chunk: Any) -> str:
        """
        Helper function to format streaming chunks for Anthropic API format

        Args:
            chunk: A string or dictionary to be returned in SSE format

        Returns:
            str: A properly formatted SSE chunk string
        """
        if isinstance(chunk, dict):
            # Use safe_dumps for proper JSON serialization with circular reference detection
            chunk_str = safe_dumps(chunk)
            return f"{STREAM_SSE_DATA_PREFIX}{chunk_str}\n\n"
        else:
            return chunk

    @staticmethod
    async def _finalize_streaming_generator_cleanup(
        request: Request | None,
        request_data: dict,
        response: Any,
        stream_completed: bool = False,
        client_disconnected: bool = False,
        user_api_key_dict: UserAPIKeyAuth | None = None,
        proxy_logging_obj: ProxyLogging | None = None,
    ) -> None:
        with anyio.CancelScope(shield=True):
            should_record_client_disconnect = client_disconnected or (not stream_completed)
            recorded_client_disconnect = False
            if should_record_client_disconnect:
                recorded_client_disconnect = await _record_streaming_client_disconnect_if_needed(
                    request,
                    request_data,
                    client_disconnected,
                )
            if recorded_client_disconnect:
                deferred_stream_logging_armed = _deferred_stream_logging_is_armed(request_data)
                ProxyLogging._fire_deferred_stream_logging(request_data)
                # A disconnect-time success event (the deferred-guardrail flush
                # above, or the partial-spend billing below) releases the
                # request's max_parallel_requests slot through the limiter's
                # own success callback. Release the slot explicitly only when
                # no such event fires, so exactly one release happens; two
                # concurrent releases would race and double-decrement under the
                # limiter's in-memory fallback.
                success_event_owns_slot_release = deferred_stream_logging_armed
                if not deferred_stream_logging_armed:
                    success_event_owns_slot_release = await _bill_partial_streamed_spend_on_disconnect(
                        request_data, response
                    )
                if (
                    not success_event_owns_slot_release
                    and proxy_logging_obj is not None
                    and user_api_key_dict is not None
                ):
                    await proxy_logging_obj._arelease_max_parallel_requests_on_disconnect(
                        user_api_key_dict, request_data
                    )

            if hasattr(response, "aclose"):
                try:
                    await response.aclose()
                except BaseException as e:  # noqa: BLE001
                    verbose_proxy_logger.debug(
                        "async_streaming_data_generator: error closing response stream: %s",
                        e,
                    )

    @staticmethod
    async def async_streaming_data_generator(
        response: Any,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: dict,
        proxy_logging_obj: ProxyLogging,
        *,
        serialize_chunk: StreamChunkSerializer,
        serialize_error: StreamErrorSerializer,
        request: Request | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Shared streaming data generator: runs proxy iterator hook, per-chunk hook,
        cost injection, then yields chunks via serialize_chunk; on exception runs
        failure hook and yields via serialize_error. Use for SSE or NDJSON.
        """
        verbose_proxy_logger.debug("inside generator")
        # Resolve per-stream (not per-chunk) whether the heavy per-chunk path
        # is needed. When no callback overrides ``async_post_call_streaming_hook``,
        # no CustomGuardrail is active, and cost injection is disabled, the
        # per-chunk hook returns the chunk unchanged, ``str_so_far`` is never
        # consumed, and cost injection is a no-op -- so the per-chunk coroutine
        # await, response-string materialization, and cost-injection call are
        # pure overhead on the streaming hot path (the default config).
        caps = ProxyLogging._callback_capabilities()
        cost_injection_enabled = bool(getattr(litellm, "include_cost_in_streaming_usage", False))
        fast_path = not caps.has_streaming_chunk_override and not caps.has_guardrail and not cost_injection_enabled
        debug_enabled = verbose_proxy_logger.isEnabledFor(logging.DEBUG)
        stream_completed = False
        client_disconnected = False
        delivered_chunk = False
        try:
            str_so_far = ""
            async for chunk in proxy_logging_obj.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=response,
                request_data=request_data,
            ):
                # ``.format(chunk)`` was previously evaluated for every chunk
                # regardless of log level; gate it behind the level check.
                if debug_enabled:
                    verbose_proxy_logger.debug("async_data_generator: received streaming chunk - %s", chunk)

                if not fast_path:
                    chunk = await proxy_logging_obj.async_post_call_streaming_hook(
                        user_api_key_dict=user_api_key_dict,
                        response=chunk,
                        data=request_data,
                        str_so_far=str_so_far,
                    )

                    if isinstance(chunk, (ModelResponse, ModelResponseStream)):
                        response_str = litellm.get_response_string(response_obj=chunk)
                        str_so_far += response_str
                    elif hasattr(chunk, "model_dump"):
                        try:
                            d = chunk.model_dump(mode="json", exclude_none=True)
                            if isinstance(d, dict):
                                str_so_far += str(d.get("content", ""))
                        except Exception:
                            pass
                    elif isinstance(chunk, dict):
                        str_so_far += str(chunk.get("content", ""))

                    model_name = request_data.get("model", "")
                    chunk = ProxyBaseLLMRequestProcessing._process_chunk_with_cost_injection(chunk, model_name)

                # Set before the yield: an async generator suspends at the yield,
                # so a GeneratorExit on client disconnect is raised there and any
                # statement after the yield never runs. The slow-path hook is
                # awaited above, so a cancellation during it still leaves this
                # False and refunds.
                delivered_chunk = True
                yield serialize_chunk(chunk)
            stream_completed = True
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected mid-stream. CancelledError / GeneratorExit
            # are BaseException and bypass the success/failure logging
            # callbacks that release the pre-call max_parallel_requests +1.
            # Flag the disconnect; the shielded cleanup in `finally` owns the
            # slot release so it can coordinate with disconnect-time success
            # billing and release exactly once. This is the outermost generator
            # Starlette closes on disconnect, so the nested iterator hook (which
            # only sees GeneratorExit on GC) cannot own the refund.
            if not stream_completed:
                client_disconnected = True
            if not delivered_chunk:
                from litellm.proxy.spend_tracking.budget_reservation import (
                    release_budget_reservation_on_cancel,
                )

                await release_budget_reservation_on_cancel(getattr(user_api_key_dict, "budget_reservation", None))
            raise
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.proxy_server.async_data_generator(): Exception occured - {}".format(str(e))
            )
            transformed_exception = await proxy_logging_obj.post_call_failure_hook(
                user_api_key_dict=user_api_key_dict,
                original_exception=e,
                request_data=request_data,
            )
            if transformed_exception is not None:
                e = transformed_exception
            verbose_proxy_logger.debug(
                f"\033[1;31mAn error occurred: {e}\n\n Debug this by setting `--debug`, e.g. `litellm --model gpt-3.5-turbo --debug`"
            )

            if isinstance(e, HTTPException):
                raise e
            error_traceback = _redact_string(traceback.format_exc())
            error_msg = f"{str(e)}\n\n{error_traceback}"
            proxy_exception = ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", 500),
            )
            stream_completed = True
            yield serialize_error(proxy_exception)
        finally:
            await ProxyBaseLLMRequestProcessing._finalize_streaming_generator_cleanup(
                request=request,
                request_data=request_data,
                response=response,
                stream_completed=stream_completed,
                client_disconnected=client_disconnected,
                user_api_key_dict=user_api_key_dict,
                proxy_logging_obj=proxy_logging_obj,
            )

    @staticmethod
    def async_sse_data_generator(
        response: Any,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: dict,
        proxy_logging_obj: ProxyLogging,
        request: Request | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Anthropic /messages and Google /generateContent streaming data generator require SSE events.

        Returns the underlying ``async_streaming_data_generator`` configured with
        SSE serializers directly (rather than re-wrapping it in another
        ``async for: yield`` trampoline), so a streamed chunk traverses one
        fewer async-generator layer / coroutine resume on the hot path.
        """
        return ProxyBaseLLMRequestProcessing.async_streaming_data_generator(
            response=response,
            user_api_key_dict=user_api_key_dict,
            request_data=request_data,
            proxy_logging_obj=proxy_logging_obj,
            serialize_chunk=ProxyBaseLLMRequestProcessing.return_sse_chunk,
            serialize_error=lambda proxy_exc: (
                f"{STREAM_SSE_DATA_PREFIX}{json.dumps({'error': proxy_exc.to_dict()})}\n\n"
            ),
            request=request,
        )

    @staticmethod
    def _process_chunk_with_cost_injection(chunk: Any, model_name: str) -> Any:
        """
        Process a streaming chunk and inject cost information if enabled.

        Args:
            chunk: The streaming chunk (dict, str, bytes, or bytearray)
            model_name: Model name for cost calculation

        Returns:
            The processed chunk with cost information injected if applicable
        """
        if not getattr(litellm, "include_cost_in_streaming_usage", False):
            return chunk

        try:
            if isinstance(chunk, dict):
                maybe_modified = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(chunk, model_name)
                if maybe_modified is not None:
                    return maybe_modified
            elif isinstance(chunk, (bytes, bytearray)):
                # Decode to str, inject, and rebuild as bytes
                try:
                    s = chunk.decode("utf-8", errors="ignore")
                    maybe_mod = ProxyBaseLLMRequestProcessing._inject_cost_into_sse_frame_str(s, model_name)
                    if maybe_mod is not None:
                        return (maybe_mod + ("" if maybe_mod.endswith("\n\n") else "\n\n")).encode("utf-8")
                except Exception:
                    pass
            elif isinstance(chunk, str):
                # Try to parse SSE frame and inject cost into the data line
                maybe_mod = ProxyBaseLLMRequestProcessing._inject_cost_into_sse_frame_str(chunk, model_name)
                if maybe_mod is not None:
                    # Ensure trailing frame separator
                    return maybe_mod if maybe_mod.endswith("\n\n") else (maybe_mod + "\n\n")
        except Exception:
            # Never break streaming on optional cost injection
            pass

        return chunk

    @staticmethod
    def _inject_cost_into_sse_frame_str(frame_str: str, model_name: str) -> Optional[str]:
        """
        Inject cost information into an SSE frame string by modifying the JSON in the 'data:' line.

        Args:
            frame_str: SSE frame string that may contain multiple lines
            model_name: Model name for cost calculation

        Returns:
            Modified SSE frame string with cost injected, or None if no modification needed
        """
        try:
            # Split preserving lines
            lines = frame_str.split("\n")
            for idx, ln in enumerate(lines):
                stripped_ln = ln.strip()
                if stripped_ln.startswith("data:"):
                    json_part = stripped_ln.split("data:", 1)[1].strip()
                    if json_part and json_part != "[DONE]":
                        obj = json.loads(json_part)
                        maybe_modified = ProxyBaseLLMRequestProcessing._inject_cost_into_usage_dict(obj, model_name)
                        if maybe_modified is not None:
                            # Replace just this line with updated JSON using safe_dumps
                            lines[idx] = f"data: {safe_dumps(maybe_modified)}"
                            return "\n".join(lines)
            return None
        except Exception:
            return None

    @staticmethod
    def _inject_cost_into_usage_dict(obj: dict, model_name: str) -> Optional[dict]:
        """
        Inject cost information into a usage dictionary for message_delta events.

        Args:
            obj: Dictionary containing the SSE event data
            model_name: Model name for cost calculation

        Returns:
            Modified dictionary with cost injected, or None if no modification needed
        """
        if obj.get("type") == "message_delta" and isinstance(obj.get("usage"), dict):
            _usage = obj["usage"]
            prompt_tokens = int(_usage.get("input_tokens", 0) or 0)
            completion_tokens = int(_usage.get("output_tokens", 0) or 0)
            total_tokens = int(
                _usage.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens)
            )

            # Extract additional usage fields
            cache_creation_input_tokens = _usage.get("cache_creation_input_tokens")
            cache_read_input_tokens = _usage.get("cache_read_input_tokens")
            web_search_requests = _usage.get("web_search_requests")
            completion_tokens_details = _usage.get("completion_tokens_details")
            prompt_tokens_details = _usage.get("prompt_tokens_details")

            usage_kwargs: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            # Add optional named parameters
            if completion_tokens_details is not None:
                usage_kwargs["completion_tokens_details"] = completion_tokens_details
            if prompt_tokens_details is not None:
                usage_kwargs["prompt_tokens_details"] = prompt_tokens_details

            # Handle web_search_requests by wrapping in ServerToolUse
            if web_search_requests is not None:
                usage_kwargs["server_tool_use"] = ServerToolUse(web_search_requests=web_search_requests)

            # Add cache-related fields to **params (handled by Usage.__init__)
            if cache_creation_input_tokens is not None:
                usage_kwargs["cache_creation_input_tokens"] = cache_creation_input_tokens
            if cache_read_input_tokens is not None:
                usage_kwargs["cache_read_input_tokens"] = cache_read_input_tokens

            _mr = ModelResponse(usage=Usage(**usage_kwargs))

            try:
                cost_val = litellm.completion_cost(
                    completion_response=_mr,
                    model=model_name,
                )
            except Exception:
                cost_val = None

            if cost_val is not None:
                obj.setdefault("usage", {})["cost"] = cost_val
                return obj
        return None

    def maybe_get_model_id(self, _logging_obj: Optional[LiteLLMLoggingObj]) -> Optional[str]:
        """
        Get model_id from logging object or request metadata.

        The router sets model_info.id when selecting a deployment. This tries multiple locations
        where the ID might be stored depending on the request lifecycle stage.
        """
        model_id = None
        if _logging_obj:
            # 1. Try getting from litellm_params (updated during call)
            if hasattr(_logging_obj, "litellm_params") and _logging_obj.litellm_params:
                # First check direct model_info path (set by router.py with selected deployment)
                model_info = _logging_obj.litellm_params.get("model_info") or {}
                model_id = model_info.get("id", None)

                # Fallback to nested metadata path
                if not model_id:
                    metadata = _logging_obj.litellm_params.get("metadata") or {}
                    model_info = metadata.get("model_info") or {}
                    model_id = model_info.get("id", None)

            # 2. Fallback to kwargs (initial)
            if not model_id:
                _kwargs = getattr(_logging_obj, "kwargs", None)
                if _kwargs:
                    litellm_params = _kwargs.get("litellm_params", {})
                    # First check direct model_info path
                    model_info = litellm_params.get("model_info") or {}
                    model_id = model_info.get("id", None)

                    # Fallback to nested metadata path
                    if not model_id:
                        metadata = litellm_params.get("metadata") or {}
                        model_info = metadata.get("model_info") or {}
                        model_id = model_info.get("id", None)

        # 3. Final fallback to self.data["litellm_metadata"] (for routes like /v1/responses that populate data before error)
        if not model_id:
            litellm_metadata = self.data.get("litellm_metadata", {}) or {}
            model_info = litellm_metadata.get("model_info", {}) or {}
            model_id = model_info.get("id", None)

        return model_id
