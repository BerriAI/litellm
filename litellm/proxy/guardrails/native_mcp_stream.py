import json
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterable, Mapping
from typing import Final

from openai._streaming import SSEDecoder
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.constants import MCP_MAX_SSE_EVENT_SIZE_BYTES
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.native_mcp_policy import (
    NativeMCPAccess,
    NativeMCPDenial,
    NativeMCPRequestContext,
    NativeMCPResolution,
    evaluate_native_mcp_policy,
    is_gateway_only,
    native_mcp_request_context,
    raise_native_mcp_denial,
    resolve_native_mcp_policy,
)
from litellm.proxy.common_utils.sse_keepalive import split_complete_sse_frames
from litellm.proxy.guardrails.tool_name_extraction import (
    extract_request_mcp_accesses,
    extract_response_mcp_accesses,
    response_mcp_items,
)

_EVENT: Final = TypeAdapter(Mapping[str, object])
_MAX_PENDING_EVENTS: Final = 1024


class NativeMCPStreamValidator:
    def __init__(self, policy: NativeMCPResolution, request_accesses: tuple[NativeMCPAccess, ...]) -> None:
        self._policy: Final = policy
        self._request_accesses = request_accesses
        self._attempt_id: str | None = None
        self._validated_calls: frozenset[str] = frozenset()
        self._pending_calls: frozenset[str] = frozenset()
        self._held: deque[object] = (
            deque()
        )  # mutable-ok: bounded streaming queue avoids copying all held events for every chunk
        self._held_bytes: int = 0
        self._pending_bytes: bytes = b""

    def bind(self, context: NativeMCPRequestContext | None) -> None:
        if context is None or context.attempt_id == self._attempt_id:
            return
        self.finish()
        raise_native_mcp_denial(evaluate_native_mcp_policy(self._policy, context.accesses))
        self._request_accesses = context.accesses
        self._attempt_id = context.attempt_id
        self._validated_calls = frozenset()

    def _check_response(self, response: object) -> None:
        accesses: Final = extract_response_mcp_accesses(response, self._request_accesses)
        raise_native_mcp_denial(evaluate_native_mcp_policy(self._policy, accesses))

    def _check_item(self, item: Mapping[str, object]) -> None:
        if item.get("type") == "mcp_tool_result":
            self._check_call_id(item.get("tool_use_id"))
            return
        if item.get("type") in ("mcp_call", "mcp_tool_use", "mcp_approval_request"):
            call_id: Final = item.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise_native_mcp_denial(NativeMCPDenial("Native MCP stream call has no ID"))
                return
            server: Final = item.get("server_name") if item.get("type") == "mcp_tool_use" else item.get("server_label")
            if item.get("name") in (None, "") or server in (None, ""):
                self._pending_calls = self._pending_calls | {call_id}
                return
            self._check_response({"output": [item]})
            self._validated_calls = self._validated_calls | {call_id}
            self._pending_calls = self._pending_calls - {call_id}
        else:
            self._check_response({"output": [item]})

    def _check_call_id(self, call_id: object) -> None:
        if not isinstance(call_id, str) or not call_id:
            raise_native_mcp_denial(NativeMCPDenial("Native MCP stream references a tool call without an ID"))
            return
        if call_id not in self._validated_calls:
            self._pending_calls = self._pending_calls | {call_id}

    def _release(self, chunk: object) -> tuple[object, ...]:
        if self._pending_calls:
            encoded: Final = (
                chunk
                if isinstance(chunk, bytes)
                else chunk.encode("utf-8")
                if isinstance(chunk, str)
                else json.dumps(chunk.model_dump(mode="json") if isinstance(chunk, BaseModel) else chunk).encode(
                    "utf-8"
                )
            )
            self._held_bytes += len(encoded)
            if self._held_bytes > MCP_MAX_SSE_EVENT_SIZE_BYTES or len(self._held) >= _MAX_PENDING_EVENTS:
                raise_native_mcp_denial(
                    NativeMCPDenial("Native MCP stream exceeded its unvalidated-event buffer limit")
                )
            self._held.append(chunk)
            return ()
        released: Final = (*self._held, chunk)
        self._held.clear()
        self._held_bytes = 0
        return released

    def check_event(self, event: Mapping[str, object]) -> None:
        if "choices" in event or "provider_specific_fields" in event:
            items: Final = response_mcp_items(event)
            if isinstance(items, NativeMCPDenial):
                raise_native_mcp_denial(items)
                return
            for provider_item in items:
                self._check_item(provider_item)
        event_type: Final = event.get("type")
        if event_type in ("response.output_item.added", "response.output_item.done", "content_block_start"):
            item: Final = as_str_mapping(
                event.get("content_block") if event_type == "content_block_start" else event.get("item")
            )
            if item is None:
                raise_native_mcp_denial(NativeMCPDenial("Cannot read a streamed output item"))
                return
            self._check_item(item)
        elif isinstance(event_type, str) and (
            event_type.startswith("response.mcp_call.") or event_type.startswith("response.mcp_call_arguments.")
        ):
            self._check_call_id(event.get("item_id"))
        elif event_type in ("response.completed", "response.incomplete"):
            self._check_response(event.get("response"))
        elif event_type in ("mcp_call", "mcp_tool_use", "mcp_tool_result", "mcp_approval_request"):
            self._check_item(event)

    def feed(self, chunk: object) -> tuple[object, ...]:
        if is_gateway_only(self._request_accesses):
            return (chunk,)
        if isinstance(chunk, (str, bytes)):
            incoming: Final = chunk.encode("utf-8") if isinstance(chunk, str) else chunk
            complete, tail = split_complete_sse_frames(self._pending_bytes + incoming)
            self._pending_bytes = tail
            if len(tail) > MCP_MAX_SSE_EVENT_SIZE_BYTES:
                raise_native_mcp_denial(NativeMCPDenial("Native MCP stream exceeded the incomplete SSE frame limit"))
            if not complete:
                return ()
            try:
                for event in SSEDecoder().iter_bytes(iter((complete,))):
                    if event.data and event.data != "[DONE]":
                        self.check_event(_EVENT.validate_json(event.data))
            except (ValidationError, UnicodeDecodeError):
                raise_native_mcp_denial(NativeMCPDenial("Cannot parse native MCP stream events"))
            return self._release(complete.decode("utf-8") if isinstance(chunk, str) else complete)
        if self._pending_bytes:
            raise_native_mcp_denial(NativeMCPDenial("Native MCP stream changed format inside an SSE frame"))
        data: Final = as_str_mapping(chunk.model_dump() if isinstance(chunk, BaseModel) else chunk)
        if data is None:
            raise_native_mcp_denial(NativeMCPDenial("Cannot read native MCP stream event"))
            return ()
        self.check_event(data)
        return self._release(chunk)

    def finish(self) -> None:
        if self._pending_calls:
            raise_native_mcp_denial(NativeMCPDenial("Native MCP stream ended with unvalidated tool calls"))
        if self._pending_bytes.strip():
            raise_native_mcp_denial(NativeMCPDenial("Native MCP stream ended inside an SSE frame"))


async def validate_native_mcp_stream(
    response: AsyncIterable[object],
    identity: UserAPIKeyAuth,
    request: Mapping[str, object],
) -> AsyncGenerator[object, None]:
    policy: Final = resolve_native_mcp_policy(identity)
    if policy is None:
        async for chunk in response:
            yield chunk
        return
    from litellm.proxy.common_request_processing import close_guarded_stream

    try:
        initial_context: Final = native_mcp_request_context(response)
        request_accesses: Final = (
            initial_context.accesses if initial_context is not None else extract_request_mcp_accesses(request)
        )
        raise_native_mcp_denial(evaluate_native_mcp_policy(policy, request_accesses))
        if isinstance(request_accesses, NativeMCPDenial):
            return
        validator: Final = NativeMCPStreamValidator(policy, request_accesses)
        validator.bind(initial_context)
        async for chunk in response:
            validator.bind(native_mcp_request_context(response) or native_mcp_request_context(chunk))
            for released in validator.feed(chunk):
                yield released
        validator.finish()
    finally:
        await close_guarded_stream(response)
