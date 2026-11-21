"""Spend-log records for JSON-RPC requests the MCP SDK rejects at the protocol layer.

The MCP SDK writes protocol errors (unknown method, malformed params) straight to
the ASGI ``send``, bypassing ``call_mcp_tool``'s logging, so such rejections
produced no ``standard_logging_object``. See
https://github.com/BerriAI/litellm/issues/28929.
"""

import json
import traceback
import types
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Final

from pydantic import TypeAdapter, ValidationError
from starlette.types import Message, Send
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.constants import MAXIMUM_TRACEBACK_LINES_TO_LOG
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    LiteLLMProxyRequestSetup,
    get_chain_id_from_headers,
)
from litellm.types.utils import CallTypes
from litellm.utils import Rules, function_setup

# Upper bound on a response body the send wrapper will parse to detect a
# JSON-RPC rejection. Rejection envelopes are tiny; anything larger is a
# permitted tool result and must not be re-deserialized after it was sent.
_MCP_PROTOCOL_ERROR_PEEK_MAX_BYTES: Final = 64 * 1024
EMPTY_MCP_PROTOCOL_PARAMS: Final[Mapping[str, object]] = types.MappingProxyType({})


class _JSONRPCRequestForLogging(TypedDict, total=False):
    jsonrpc: ReadOnly[object]
    method: ReadOnly[object]
    params: ReadOnly[object]
    id: ReadOnly[object]


class _JSONRPCErrorForLogging(TypedDict, total=False):
    code: ReadOnly[object]
    message: ReadOnly[object]


class _JSONRPCErrorEnvelopeForLogging(TypedDict):
    error: ReadOnly[_JSONRPCErrorForLogging]


_JSONRPC_REQUEST_FOR_LOGGING_ADAPTER: Final[TypeAdapter[_JSONRPCRequestForLogging | tuple[object, ...]]] = TypeAdapter(
    _JSONRPCRequestForLogging | tuple[object, ...]
)
_JSONRPC_PARAMS_FOR_LOGGING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_JSONRPC_ERROR_FOR_LOGGING_ADAPTER: Final[TypeAdapter[_JSONRPCErrorEnvelopeForLogging]] = TypeAdapter(
    _JSONRPCErrorEnvelopeForLogging
)


def _jsonrpc_rejection_reason(error_code: int | None) -> str:
    """Map a JSON-RPC error code to a stable, low-cardinality reason string
    for spend-log metadata. Unknown codes degrade to ``protocol_error``."""
    match error_code:
        case -32700:
            return "parse_error"
        case -32600:
            return "invalid_request"
        case -32601:
            return "unknown_method"
        case -32602:
            return "malformed_params"
        case -32603:
            return "internal_error"
        case _:
            return "protocol_error"


def parse_jsonrpc_request_for_logging(
    body: bytes,
) -> tuple[str | None, Mapping[str, object], object]:
    """Parse a peeked JSON-RPC request body into ``(method, params, id)`` for
    protocol-rejection logging (#28929).

    - A single request object yields its ``method`` (only when ``jsonrpc``
      is ``"2.0"`` and ``method`` is a string), ``params`` (only if a dict),
      and ``id``.
    - A top-level array (JSON-RPC batch) yields the synthetic method
      ``"batch"``: the MCP SDK validates the whole array as one message and,
      if it can't, emits a single error response (it does NOT process
      elements individually), so we record the one rejection without
      attributing a tool name.
    - A truncated/oversized peek or any non-request shape yields
      ``(None, {}, None)`` so the caller installs no logging seam (a wrong
      tool-name attribution is worse than no record).
    """
    try:
        request: Final = _JSONRPC_REQUEST_FOR_LOGGING_ADAPTER.validate_json(body)
    except ValidationError:
        return None, EMPTY_MCP_PROTOCOL_PARAMS, None
    if isinstance(request, tuple):
        return "batch", EMPTY_MCP_PROTOCOL_PARAMS, None
    method: Final = request.get("method")
    if request.get("jsonrpc") != "2.0" or not isinstance(method, str):
        return None, EMPTY_MCP_PROTOCOL_PARAMS, None
    raw_params: Final = request.get("params")
    try:
        params: Final = _JSONRPC_PARAMS_FOR_LOGGING_ADAPTER.validate_python(raw_params)
    except ValidationError:
        return method, EMPTY_MCP_PROTOCOL_PARAMS, request.get("id")
    return method, params, request.get("id")


async def _log_mcp_protocol_rejection(
    request_method: str | None,
    params: Mapping[str, object],
    jsonrpc_error: Mapping[str, object],
    request_id: object,
    user_api_key_auth: UserAPIKeyAuth | None,
    raw_headers: Mapping[str, str] | None,
    start_time: datetime,
) -> None:
    """Emit a real ``standard_logging_object`` for a JSON-RPC request
    rejected at the protocol layer by the MCP SDK (e.g. unknown method,
    malformed params). Without this, such rejections produce no spend-log
    record at all — see https://github.com/BerriAI/litellm/issues/28929.

    This builds a genuine ``LiteLLMLoggingObj`` via ``function_setup`` and
    calls its ``async_failure_handler`` — the same mechanism
    ``list_mcp_tools`` / ``call_mcp_tool`` use — so the
    ``standard_logging_object`` is actually synthesized and dispatched to
    registered callbacks. (``proxy_logging_obj.post_call_failure_hook`` is
    NOT used here: it only synthesizes a ``standard_logging_object`` for
    proxy-only *LLM-API* errors and requires a pre-existing
    ``litellm_logging_obj`` in ``request_data`` — neither holds for a
    protocol-level MCP rejection, so it would deliver a raw dict and no
    record. Driving ``async_failure_handler`` directly also means a
    malformed *client* request never fires the ``llm_exceptions`` alert that
    lives in ``post_call_failure_hook`` -- these are 4xx client errors, not
    LLM API failures.)

    Purely additive and best-effort: any failure here is swallowed so a
    logging/serialization error can never alter or drop the client response.
    """
    try:
        if user_api_key_auth is None:
            return

        raw_tool_name: Final = params.get("name") or params.get("mcp_tool_name")
        tool_name: Final = str(raw_tool_name) if raw_tool_name else None
        raw_error_code: Final = jsonrpc_error.get("code")
        error_code: Final[int | None] = raw_error_code if isinstance(raw_error_code, int) else None
        raw_error_message: Final = jsonrpc_error.get("message")
        error_message: Final = (
            raw_error_message if isinstance(raw_error_message, str) else "MCP JSON-RPC protocol error"
        )

        spend_logs_metadata: Final[dict[str, object]] = {  # mutable-ok: callback metadata requires a concrete dict
            "mcp_operation": request_method,
            "rejection_reason": _jsonrpc_rejection_reason(error_code),
            "jsonrpc_error_code": error_code,
        }
        if tool_name is not None:
            spend_logs_metadata["mcp_tool_name"] = tool_name

        request_data: Final[dict[str, object]] = {  # mutable-ok: auth enrichment mutates this legacy request payload
            "model": "MCP: protocol_error",
            "call_type": CallTypes.call_mcp_tool.value,
            "litellm_call_id": str(uuid.uuid4()),
            "litellm_trace_id": get_chain_id_from_headers(dict(raw_headers) if raw_headers is not None else None),
            "metadata": {
                "spend_logs_metadata": spend_logs_metadata,
            },
            "input": [
                {
                    "role": "system",
                    # Keep message content a string: the downstream
                    # standard-logging builder may run token-counting /
                    # serialization over messages, and a non-string content
                    # can raise there (silently swallowed below -> the very
                    # record this fix emits would be dropped). The structured
                    # fields live in spend_logs_metadata above.
                    "content": json.dumps(
                        {
                            "mcp_operation": request_method,
                            "jsonrpc_id": request_id,
                            "jsonrpc_error_code": error_code,
                        }
                    ),
                }
            ],
        }

        LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(  # pyright: ignore[reportUnknownMemberType]  # legacy helper is untyped
            data=request_data,
            user_api_key_dict=user_api_key_auth,
            _metadata_variable_name="metadata",
        )

        # Build a genuine logging object (mirrors list_mcp_tools at the
        # ``function_setup`` call site) and drive its failure handler, which
        # synthesizes ``model_call_details["standard_logging_object"]`` and
        # dispatches it to every registered async failure callback. A
        # function_setup failure is caught by the outer best-effort handler.
        litellm_logging_obj, _ = function_setup(
            original_function="call_mcp_tool",
            rules_obj=Rules(),
            start_time=start_time,
            **request_data,  # pyright: ignore[reportAny, reportArgumentType]  # function_setup erases its forwarded keyword types
        )

        if litellm_logging_obj is None:  # pyright: ignore[reportUnnecessaryComparison]  # keep best-effort guard for runtime substitutes
            return

        litellm_logging_obj.call_type = CallTypes.call_mcp_tool.value
        litellm_logging_obj.model = "MCP: protocol_error"

        await litellm_logging_obj.async_failure_handler(
            Exception(error_message),
            traceback.format_exc(limit=MAXIMUM_TRACEBACK_LINES_TO_LOG),
            start_time,
            datetime.now(),  # noqa: DTZ005  # matches the logging pipeline start time
        )
    except Exception as log_exc:  # noqa: BLE001  # best-effort: logging must never break the MCP response
        verbose_logger.debug("MCP protocol-error logging failed (continuing): %s", log_exc)


def _parse_jsonrpc_error_response_for_logging(
    raw_body: bytes,
) -> _JSONRPCErrorEnvelopeForLogging | None:
    stripped_body: Final = raw_body.lstrip()
    data_lines: Final = tuple(
        line.removeprefix(b"data:").lstrip() for line in raw_body.splitlines() if line.startswith(b"data:")
    )
    if not stripped_body.startswith(b"{") and len(data_lines) != 1:
        return None
    json_body: Final = stripped_body if stripped_body.startswith(b"{") else data_lines[0]
    try:
        return _JSONRPC_ERROR_FOR_LOGGING_ADAPTER.validate_json(json_body)
    except ValidationError:
        return None


def wrap_send_for_protocol_error_logging(
    inner_send: Send,
    request_method: str | None,
    params: Mapping[str, object],
    request_id: object,
    user_api_key_auth: UserAPIKeyAuth | None,
    raw_headers: Mapping[str, str] | None,
    start_time: datetime,
) -> Send:
    """Wrap an ASGI ``send`` so a JSON-RPC error response emitted by the MCP
    SDK (an ``error`` envelope with no ``result``) produces exactly one
    ``standard_logging_object``. Only JSON-RPC *requests* (carrying a
    ``method``) are loggable; everything else is forwarded untouched.

    The wrapper always forwards the original message FIRST and never mutates
    it, so the client response is byte-for-byte unchanged on every path.
    """
    if request_method is None:
        return inner_send

    already_logged: Final = {"done": False}

    async def _logging_send(message: Message) -> None:
        await inner_send(message)
        if already_logged["done"]:
            return
        try:
            if message.get("type") != "http.response.body":
                return
            raw_body: Final[object] = message.get("body")  # pyright: ignore[reportAny]  # ASGI Message body is untyped
            if not isinstance(raw_body, bytes) or not raw_body:
                return
            # Bound the work: a JSON-RPC protocol-error envelope is tiny.
            # Skip the full deserialization for anything larger than the cap
            # (a permitted tool result) and only parse when the bytes look
            # like an error envelope. This avoids re-deserializing every
            # successful response body after it has already been sent.
            if len(raw_body) > _MCP_PROTOCOL_ERROR_PEEK_MAX_BYTES:
                return
            if b'"error"' not in raw_body or b'"result"' in raw_body:
                return
            payload: Final = _parse_jsonrpc_error_response_for_logging(raw_body)
            if payload is None:
                return
            already_logged["done"] = True
            await _log_mcp_protocol_rejection(
                request_method=request_method,
                params=params,
                jsonrpc_error=payload["error"],
                request_id=request_id,
                user_api_key_auth=user_api_key_auth,
                raw_headers=raw_headers,
                start_time=start_time,
            )
        except (TypeError, ValueError):
            # Non-JSON / streaming chunk — nothing to log here.
            return
        except Exception as wrap_exc:  # noqa: BLE001  # best-effort: never break the response on logging
            verbose_logger.debug(
                "MCP protocol-error send wrapper failed (continuing): %s",
                wrap_exc,
            )

    return _logging_send
