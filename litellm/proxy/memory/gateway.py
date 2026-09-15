import json
import math
from collections.abc import AsyncGenerator, Mapping
from types import MappingProxyType
from typing import Final
from uuid import uuid4

from fastapi import HTTPException, Request
from openai._streaming import ServerSentEvent, SSEDecoder
from pydantic import TypeAdapter
from starlette.responses import JSONResponse, Response

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.prompt_templates.server_tool_responses import (
    executable_server_calls,
    object_items,
    object_value,
    response_has_client_tools,
    response_messages,
)
from litellm.litellm_core_utils.prompt_templates.server_tool_stream import ServerToolStream, ServerToolStreamError
from litellm.litellm_core_utils.prompt_templates.server_tools import (
    ServerToolRoute,
    append_server_reference,
    continue_server_tools,
    has_server_output_constraint,
    inject_server_tools,
    prepare_server_tool_context,
    restore_client_output,
    trailing_system_messages,
    transcript_items,
    uncached_system_directive,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.sse_keepalive import wrap_passthrough_sse_bytes_with_keepalive_pings
from litellm.proxy.memory.continuation import MemoryContinuation, MemoryContinuations
from litellm.proxy.memory.knowledge import (
    MEMORY_TOOL_NAMES,
    execute_memory_tool,
    memory_functions,
    memory_workflow,
)
from litellm.proxy.memory.policy import (
    MemoryIdentity,
    gateway_memory_is_enabled,
    resolve_memory_access,
)
from litellm.proxy.memory.store import MemoryStore
from litellm.proxy.memory.transport import RoundExecutor, gateway_round, in_gateway_round

_OBJECT: Final = TypeAdapter(dict[str, object])
_MAX_ROUNDS: Final = 8
_MAX_TOOL_CALLS: Final = 16


class GatewayMemoryLoop:
    def __init__(
        self,
        execute: RoundExecutor,
        request: Request,
        data: Mapping[str, object],
        route: ServerToolRoute,
        store: MemoryStore,
        auth: UserAPIKeyAuth,
    ) -> None:
        self.execute = execute
        self.auth = auth
        self.request = request
        self.original = MappingProxyType({**data, "litellm_trace_id": data.get("litellm_trace_id") or str(uuid4())})
        self.route: Final[ServerToolRoute] = route
        self.store = store
        self.continuations = MemoryContinuations(store) if route == "aresponses" else None
        self.stream = ServerToolStream(route, MEMORY_TOOL_NAMES, data)
        self.constrained_output: Final = has_server_output_constraint(data) and (
            data.get("tool_choice") in (None, "auto") or object_value(data.get("tool_choice")).get("type") == "auto"
        )
        self.preparing_output = self.constrained_output
        self.stream.suppress_output = self.preparing_output
        if route == "aresponses":
            self.stream.response_id = "resp_litellm_memory_" + uuid4().hex
        self.streaming = data.get("stream") is True
        self.visible_input = transcript_items(data, route)
        self.data: Mapping[str, object] = data
        self.replaced_input = 0
        self.prepared = False
        self.round_index = 0
        self.completed_responses: tuple[Mapping[str, object], ...] = ()
        self.upstream_ids: tuple[str, ...] = ()
        self.last_response: Mapping[str, object] | None = None
        self.headers: Mapping[str, str] = MappingProxyType({})
        self.costs: tuple[float | None, ...] = ()
        self.pending_results: tuple[Mapping[str, object], ...] = ()

    async def prepare(self) -> None:
        previous: Final = self.original.get("previous_response_id")
        previous_patch: Final = (
            await self.continuations.load_response(previous)
            if self.continuations is not None
            and isinstance(previous, str)
            and previous.startswith("resp_litellm_memory_")
            else None
        )
        if isinstance(previous, str) and previous.startswith("resp_litellm_memory_") and previous_patch is None:
            raise HTTPException(status_code=404, detail="Memory response not found or expired")
        field: Final = "input" if self.route == "aresponses" else "messages"
        functions: Final = memory_functions(self.store.access)
        injected: Final = inject_server_tools(
            {  # mutable-ok: Native provider JSON containers.
                **self.original,
                field: [  # mutable-ok: Native provider JSON containers.
                    *(previous_patch.pending_results if previous_patch else ()),
                    *self.visible_input,
                ],
                **(
                    {  # mutable-ok: Native provider JSON containers.
                        "previous_response_id": previous_patch.upstream_ids[-1]
                    }
                    if previous_patch
                    else {  # mutable-ok: Native provider JSON containers.
                    }
                ),
            },
            self.route,
            functions,
            memory_workflow(self.store.access),
            reserved_names=MEMORY_TOOL_NAMES,
        )
        self.replaced_input = trailing_system_messages(injected, self.route)
        self.data = injected
        if self.preparing_output:
            self.data = append_server_reference(
                prepare_server_tool_context(self.data, MEMORY_TOOL_NAMES),
                self.route,
                "If needed, use the available memory tools for this request. "
                "The final response will be generated separately with the client's output "
                "format and application tools. Do not call application tools during this preparation.",
            )

        self.prepared = True

    async def _call(self) -> AsyncGenerator[bytes, None]:
        response_id: Final = self.stream.response_id if self.route == "aresponses" else None
        self.stream = ServerToolStream(self.route, MEMORY_TOOL_NAMES, self.original)
        self.stream.response_id = response_id
        self.stream.begin_round()
        streaming: Final = self.streaming and not self.preparing_output
        # Claude output directives control the next generated turn. Repeat them
        # on outgoing rounds without adding pending directives to saved history.
        directives: Final = (
            transcript_items(self.original, self.route)[-self.replaced_input :] if self.replaced_input else ()
        )
        messages: Final = transcript_items(self.data, self.route)
        body: Final = {  # mutable-ok: Native provider JSON containers.
            **self.data,
            "stream": streaming,
            **(
                {  # mutable-ok: Native provider JSON containers.
                    "messages": [  # mutable-ok: Provider request JSON.
                        *messages,
                        *(uncached_system_directive(directive) for directive in directives),
                    ],
                }
                if directives and messages[-len(directives) :] != directives
                else {}  # mutable-ok: Native provider JSON containers.
            ),
            "cache": {  # mutable-ok: Native provider JSON containers.
                **object_value(self.data.get("cache")),
                "no-cache": True,
                "no-store": True,
            },
            **(
                {  # mutable-ok: Native provider JSON containers.
                    "stream_options": {  # mutable-ok: Native provider JSON containers.
                        **object_value(self.data.get("stream_options")),
                        "include_usage": True,
                    }
                }
                if streaming and self.route == "acompletion"
                else {  # mutable-ok: Native provider JSON containers.
                }
            ),
        }
        async with gateway_round(
            self.execute,
            self.request,
            body,
            self.auth.model_copy(update=MappingProxyType({"budget_reservation": None})),
            self.round_index,
        ) as call:
            start: Final = await call.started
            status: Final = start.status
            if status >= 400:
                raise HTTPException(
                    status_code=status,
                    detail="The gateway model request was rate limited"
                    if status == 429
                    else "The gateway model request failed",
                    headers={  # mutable-ok: FastAPI's HTTPException accepts a native header dictionary.
                        name.decode("latin-1"): value.decode("latin-1")
                        for name, value in start.headers
                        if name.lower() == b"retry-after"
                    },
                )
            self.headers = MappingProxyType(
                {
                    name.decode("latin-1"): value.decode("latin-1")
                    for name, value in start.headers
                    if name.lower()
                    not in (b"content-length", b"content-type", b"transfer-encoding", b"content-encoding")
                }
            )
            cost: Final = self.headers.get("x-litellm-response-cost")
            try:
                parsed_cost: Final = float(cost) if cost is not None else None
            except ValueError:
                self.costs = (*self.costs, None)
            else:
                self.costs = (
                    *self.costs,
                    parsed_cost if parsed_cost is not None and math.isfinite(parsed_cost) else None,
                )
            if streaming:
                from collections import deque

                buffered: Final = deque[bytes]()
                async for event in SSEDecoder().aiter_bytes(call.chunks()):
                    buffered.extend(self.stream.feed(event))
                response, client_chunks = self.stream.finish_round()
                self.last_response = response
                buffered.extend(client_chunks)
                calls: Final = executable_server_calls(response, self.route, MEMORY_TOOL_NAMES)
                if calls and response_has_client_tools(response, self.route, MEMORY_TOOL_NAMES):
                    original_stream: Final = self.stream
                    self.stream = ServerToolStream(self.route, MEMORY_TOOL_NAMES, self.original)
                    self.stream.response_id = response_id
                    self.stream.hide_text = True
                    for item in original_stream.objects:
                        for chunk in self.stream.feed(
                            ServerSentEvent(data=json.dumps(item), event=str(item.get("type", "")))
                        ):
                            yield chunk
                    _, filtered_chunks = self.stream.finish_round()
                    for chunk in filtered_chunks:
                        yield chunk
                elif not calls:
                    for chunk in buffered:
                        yield chunk
            else:
                content: Final = await call.read()
                self.last_response = _OBJECT.validate_json(content)
                self.stream.hide_text = bool(executable_server_calls(self.last_response, self.route, MEMORY_TOOL_NAMES))
                self.stream.accept_response(self.last_response)
        self.completed_responses = (*self.completed_responses, self.stream.response())
        self.stream.responses = self.completed_responses
        self.round_index += 1

    async def _save_continuation(self) -> None:
        if self.continuations is None or self.original.get("store") is False:
            return
        response: Final = self.stream.response()
        try:
            await self.continuations.save(
                str(response["id"]),
                MemoryContinuation(
                    response=response, upstream_ids=self.upstream_ids, pending_results=self.pending_results
                ),
            )
        except Exception:
            verbose_proxy_logger.warning("Memory response retention unavailable; returning the completed answer")
            self.stream.client_response_fields = MappingProxyType(
                {**self.stream.client_response_fields, "store": False}
            )

    def response_headers(self) -> Mapping[str, str]:
        cost_header: Final = (
            (("x-litellm-response-cost", str(sum(cost for cost in self.costs if cost is not None))),)
            if not self.streaming and self.costs and all(cost is not None for cost in self.costs)
            else ()
        )
        return MappingProxyType(
            {
                key: value
                for key, value in (
                    *((key, value) for key, value in self.headers.items() if key != "x-litellm-response-cost"),
                    *cost_header,
                    ("x-litellm-memory", "active"),
                )
            }
        )

    async def advance(self, round_index: int) -> bool:
        response: Final = self.last_response
        if response is None:
            raise HTTPException(status_code=502, detail="No model response received")
        self.upstream_ids = (*self.upstream_ids, str(response["id"]))
        try:
            memory_calls: Final = executable_server_calls(response, self.route, MEMORY_TOOL_NAMES)
        except ValueError as exc:
            raise HTTPException(
                status_code=502, detail="The model returned incomplete or invalid memory tool calls"
            ) from exc
        client_calls: Final = response_has_client_tools(response, self.route, MEMORY_TOOL_NAMES)
        if self.constrained_output and not self.preparing_output and memory_calls:
            raise HTTPException(status_code=502, detail="The final model response called an unavailable memory tool")
        if len(memory_calls) > _MAX_TOOL_CALLS or any(not call["id"] for call in memory_calls):
            raise HTTPException(status_code=502, detail="Invalid gateway memory tool calls")
        results: Final = tuple(
            [await execute_memory_tool(self.store, call, self.visible_input) for call in memory_calls]
        )
        if memory_calls:
            self.pending_results = (
                tuple(
                    {  # mutable-ok: Native provider JSON containers.
                        "type": "function_call_output",
                        "call_id": call["id"],
                        "output": json.dumps(dict(result)),
                    }
                    for call, result in zip(memory_calls, results)
                )
                if self.route == "aresponses"
                else ()
            )
            self.data = continue_server_tools(
                self.data, self.route, response, memory_calls, tuple(dict(result) for result in results)
            )
        else:
            self.pending_results = ()
            field: Final = "input" if self.route == "aresponses" else "messages"
            self.data = {  # mutable-ok: Native provider JSON containers.
                **self.data,
                field: [  # mutable-ok: Native provider JSON containers.
                    *transcript_items(self.data, self.route),
                    *response_messages(response, self.route),
                ],
            }
        if client_calls or not memory_calls:
            return True
        if round_index + 2 >= _MAX_ROUNDS:
            self.data = restore_client_output(self.data, self.original)
        return False

    async def run(self) -> AsyncGenerator[bytes, None]:
        if not self.prepared:
            await self.prepare()
        for round_index in range(_MAX_ROUNDS):
            async for chunk in self._call():
                yield chunk
            if await self.advance(round_index):
                break
        if self.preparing_output and not (
            (self.last_response or {}).get("status") == "incomplete"
            or (self.last_response or {}).get("stop_reason") == "max_tokens"
            or any(
                choice.get("finish_reason") == "length"
                for choice in object_items((self.last_response or {}).get("choices"))
            )
        ):
            response_id: Final = self.stream.response_id
            self.stream = ServerToolStream(self.route, MEMORY_TOOL_NAMES, self.original)
            if self.route == "aresponses":
                self.stream.response_id = response_id
            self.preparing_output = False
            self.data = append_server_reference(
                restore_client_output(self.data, self.original),
                self.route,
                "Memory preparation is complete. Now respond to the user's request using the required output "
                "format and any application tools provided. Do not describe the memory preparation.",
            )
            async for chunk in self._call():
                yield chunk
            await self.advance(_MAX_ROUNDS - 1)
        await self._save_continuation()
        if self.streaming:
            for chunk in self.stream.finish():
                yield chunk


def validate_memory_request(data: Mapping[str, object], request: Request) -> None:
    choice: Final = object_value(data.get("tool_choice"))
    forced_name: Final = choice.get("name") or object_value(choice.get("function")).get("name")
    if isinstance(forced_name, str) and forced_name in MEMORY_TOOL_NAMES:
        raise HTTPException(status_code=400, detail="Gateway memory tools cannot be forced through tool_choice")
    if request.url.path.startswith("/cursor/"):
        raise HTTPException(
            status_code=400,
            detail="Gateway memory requires a standard /v1/chat/completions, /v1/messages, or /v1/responses endpoint",
        )
    if data.get("functions") is not None or data.get("function_call") is not None:
        raise HTTPException(
            status_code=400, detail="Gateway memory requires tools and tool_choice instead of legacy functions"
        )
    if data.get("background") is True or data.get("n", 1) != 1:
        raise HTTPException(status_code=400, detail="Gateway memory requires a foreground request with one completion")


async def process_gateway_memory(
    data: Mapping[str, object], request: Request, auth: UserAPIKeyAuth, route: str, execute: RoundExecutor
) -> Response | None:
    if in_gateway_round():
        return None
    if route in ("aget_responses", "adelete_responses", "alist_input_items"):
        from litellm.proxy.memory.responses import memory_response_operation

        return await memory_response_operation(data, request, auth, route, execute)
    if route not in ("acompletion", "aresponses", "anthropic_messages"):
        return None
    store: Final = await gateway_memory_store(auth)
    if store is None:
        previous: Final = data.get("previous_response_id")
        if isinstance(previous, str) and previous.startswith("resp_litellm_memory_"):
            raise HTTPException(status_code=404, detail="Memory response not found or expired")
        return None
    validate_memory_request(data, request)
    from litellm.proxy.common_request_processing import (
        _UpstreamClosingStreamingResponse,  # pyright: ignore[reportPrivateUsage]  # Reuse disconnect cleanup for the prefetched stream.
        ttft_keepalive_interval,
    )
    from litellm.proxy.proxy_server import llm_router
    from litellm.proxy.spend_tracking.budget_reservation import release_or_invalidate_budget_reservation

    loop: Final = GatewayMemoryLoop(execute, request, data, route, store, auth)
    try:
        await loop.prepare()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await release_or_invalidate_budget_reservation(auth.budget_reservation)
    iterator: Final = wrap_passthrough_sse_bytes_with_keepalive_pings(
        loop.run(),
        ping_interval_seconds=ttft_keepalive_interval(data, llm_router, default_interval=5.0),
        upstream_headers=MappingProxyType({"content-type": "text/event-stream"}),
    )
    try:
        if not loop.streaming:
            async for _ in iterator:
                pass
            return JSONResponse(loop.stream.response(), headers=loop.response_headers())
        first: Final = await anext(iterator)
    except StopAsyncIteration as exc:
        raise HTTPException(status_code=502, detail="The gateway memory stream was empty") from exc
    except ServerToolStreamError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="The gateway model stream was invalid or incomplete") from exc

    async def stream() -> AsyncGenerator[bytes, None]:
        try:
            yield first
            async for chunk in iterator:
                yield chunk
        except Exception as exc:
            message: Final = (
                str(exc.detail)
                if isinstance(exc, HTTPException)
                else str(exc)
                if isinstance(exc, ServerToolStreamError)
                else "Gateway memory execution failed"
            )
            status: Final = exc.status_code if isinstance(exc, (HTTPException, ServerToolStreamError)) else 502
            yield loop.stream.error(message, status)
        finally:
            await iterator.aclose()

    return _UpstreamClosingStreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={  # mutable-ok: Native ASGI response headers.
            **loop.response_headers(),
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        },
        upstream_generator=iterator,
    )


async def gateway_memory_store(auth: UserAPIKeyAuth) -> MemoryStore | None:
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        return None
    from litellm.proxy.auth.auth_checks import effective_tool_allowlist

    identity: Final = MemoryIdentity.from_auth(auth)
    allowed_tools: Final = effective_tool_allowlist(auth)
    if not identity.user_id and not identity.key_id:
        return None
    try:
        if not await gateway_memory_is_enabled(prisma_client, user_api_key_cache, identity):
            return None
        access: Final = await resolve_memory_access(prisma_client, identity)
        required_tools: Final = frozenset(str(function["name"]) for function in memory_functions(access))
        if allowed_tools is not None and not required_tools.issubset(allowed_tools):
            return None
        return MemoryStore(prisma_client, access) if access.active else None
    except Exception:
        verbose_proxy_logger.warning("Memory access is unavailable; continuing without automatic memory")
        return None
