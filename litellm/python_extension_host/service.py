# pyright: reportAny=false, reportMissingModuleSource=false, reportMissingTypeArgument=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false
from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator, Callable, Iterable
from datetime import datetime, timezone
from typing import Final, cast  # noqa: TID251  # protobuf repeated fields lack precise runtime typing

import grpc

from litellm.integrations.custom_guardrail import is_guardrail_intervention
from litellm.python_extension.generated.v1 import extension_host_pb2 as pb
from litellm.python_extension.generated.v1 import extension_host_pb2_grpc as pb_grpc

from .cache_client import CacheClient
from .compatibility import auth_from_proto, decode_json_object, encode_json, response_from_json
from .constants import PROTOCOL_MAJOR, PROTOCOL_MINOR, SUPPORTED_HOOKS, TOKEN_METADATA_KEY
from .models import LoadedExtension, operation_error, operation_ok
from .revision_store import RevisionStore

_GUARDRAIL_METHODS: Final = {  # mutable-ok: LiteLLM compatibility payload
    pb.HOOK_PHASE_PRE_CALL: "async_pre_call_hook",
    pb.HOOK_PHASE_DURING_CALL: "async_moderation_hook",
    pb.HOOK_PHASE_POST_CALL: "async_post_call_success_hook",
}


class PythonExtensionHostService(pb_grpc.PythonExtensionHostServicer):
    def __init__(
        self,
        token: str,
        store: RevisionStore | None = None,
        gateway_stub: pb_grpc.GatewayServicesStub | None = None,
    ) -> None:
        if not token:
            raise ValueError("extension host token is required")
        self._token: Final = token
        self._store: Final = store or RevisionStore()
        self._gateway_stub: Final = gateway_stub

    async def GetCapabilities(
        self, request: pb.GetCapabilitiesRequest, context: grpc.aio.ServicerContext
    ) -> pb.HostCapabilities:
        await self._authenticate(context)
        if request.protocol_major != PROTOCOL_MAJOR:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"protocol major {request.protocol_major} is incompatible with host major {PROTOCOL_MAJOR}",
            )
        return pb.HostCapabilities(
            protocol_major=PROTOCOL_MAJOR,
            protocol_minor=min(request.protocol_minor, PROTOCOL_MINOR),
            supported_hooks=sorted(SUPPORTED_HOOKS),
            supports_duplex_streaming=True,
            supports_callback_batching=True,
            supports_cache=self._gateway_stub is not None,
            max_callback_batch_size=100,
        )

    async def PrepareRevision(
        self, request: pb.PrepareRevisionRequest, context: grpc.aio.ServicerContext
    ) -> pb.PrepareRevisionResponse:
        await self._authenticate(context)
        operation, extensions = await self._store.prepare(request.revision_id, tuple(request.extensions))
        return pb.PrepareRevisionResponse(
            operation=operation,
            extensions=[  # mutable-ok: LiteLLM compatibility payload
                extension.descriptor() for extension in extensions
            ],  # mutable-ok: LiteLLM compatibility payload
        )

    async def CommitRevision(
        self, request: pb.CommitRevisionRequest, context: grpc.aio.ServicerContext
    ) -> pb.OperationResult:
        await self._authenticate(context)
        return await self._store.commit(request.revision_id)

    async def RetireRevision(
        self, request: pb.RetireRevisionRequest, context: grpc.aio.ServicerContext
    ) -> pb.OperationResult:
        await self._authenticate(context)
        return await self._store.retire(request.revision_id)

    async def ExecuteGuardrail(
        self, request: pb.GuardrailInvocation, context: grpc.aio.ServicerContext
    ) -> pb.GuardrailResult:
        await self._authenticate(context)
        method_name = _GUARDRAIL_METHODS.get(request.hook_phase)  # rebind-ok: invocation-scoped RPC state
        if method_name is None:
            return _guardrail_error(pb.ERROR_CODE_INVALID_ARGUMENT, "invalid guardrail hook phase")
        try:
            async with self._store.acquire(request.context.active_revision, request.plugin_id) as extension:
                if method_name not in extension.hooks:
                    return _guardrail_error(
                        pb.ERROR_CODE_UNSUPPORTED_HOOK,
                        f"extension {request.plugin_id!r} does not implement {method_name}",
                    )
                return await self._execute_guardrail_method(extension, method_name, request)
        except LookupError as error:
            return _guardrail_error(pb.ERROR_CODE_NOT_ACTIVE, str(error))

    async def PublishCallbackEvents(
        self, request: pb.PublishCallbackEventsRequest, context: grpc.aio.ServicerContext
    ) -> pb.PublishCallbackEventsResponse:
        await self._authenticate(context)
        operations: list[  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
            pb.OperationResult
        ] = []  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
        operations.extend(  # mutable-ok: protobuf response requires a repeated field
            [  # mutable-ok: protobuf response requires a repeated field
                await self._publish_callback_event(event) for event in request.events
            ]
        )
        return pb.PublishCallbackEventsResponse(operations=operations)

    async def TransformStream(
        self, request_iterator: AsyncIterator[pb.StreamFrame], context: grpc.aio.ServicerContext
    ) -> AsyncIterator[pb.StreamFrame]:
        await self._authenticate(context)
        try:
            first: Final = await anext(request_iterator)
        except StopAsyncIteration:
            yield _stream_error("", "stream ended before OPEN")
            return
        if first.kind != pb.STREAM_FRAME_KIND_OPEN or not first.HasField("open"):
            yield _stream_error(first.stream_id, "first stream frame must be OPEN")
            return
        stream_open: Final = first.open
        try:
            async with self._store.acquire(stream_open.context.active_revision, stream_open.plugin_id) as extension:
                async for output in self._transform_stream(extension, first.stream_id, stream_open, request_iterator):
                    yield output
        except LookupError as error:
            yield _stream_error(first.stream_id, str(error))
        except Exception as error:  # noqa: BLE001  # isolate arbitrary customer stream failures
            yield _stream_error(first.stream_id, str(error))

    async def _execute_guardrail_method(
        self, extension: LoadedExtension, method_name: str, request: pb.GuardrailInvocation
    ) -> pb.GuardrailResult:
        request_data = decode_json_object(  # rebind-ok: invocation-scoped RPC state
            request.request_json, "request_json"
        )  # rebind-ok: invocation-scoped RPC state
        original_request_json: Final = encode_json(request_data)
        auth = auth_from_proto(request.auth)  # rebind-ok: invocation-scoped RPC state
        cache: Final = self._cache_for(request.cache if request.HasField("cache") else None)
        response = (  # rebind-ok: invocation-scoped RPC state
            response_from_json(request.response_json) if request.HasField("response_json") else None
        )  # rebind-ok: invocation-scoped RPC state
        original_response_json: Final = encode_json(response) if response is not None else None
        method = getattr(extension.target, method_name)  # rebind-ok: invocation-scoped RPC state
        try:
            if method_name == "async_pre_call_hook":
                result = await _maybe_await(  # rebind-ok: invocation-scoped RPC state
                    method(auth, cache, request_data, request.context.call_type)
                )  # rebind-ok: invocation-scoped RPC state
            elif method_name == "async_moderation_hook":
                result = await _maybe_await(  # rebind-ok: invocation-scoped RPC state
                    method(request_data, auth, request.context.call_type)
                )  # rebind-ok: invocation-scoped RPC state
            else:
                result = await _maybe_await(  # rebind-ok: invocation-scoped RPC state
                    method(request_data, auth, response)
                )  # rebind-ok: invocation-scoped RPC state
        except Exception as error:  # noqa: BLE001  # map arbitrary customer guardrail failures
            if is_guardrail_intervention(error):
                return _blocked_result(error)
            return _guardrail_error(pb.ERROR_CODE_EXTENSION_FAILED, str(error))
        if isinstance(result, Exception):
            return _blocked_result(result)
        if isinstance(result, str):
            return _blocked_result(ValueError(result))
        if request.hook_phase == pb.HOOK_PHASE_POST_CALL:
            replacement = result if result is not None else response  # rebind-ok: invocation-scoped RPC state
            replacement_json = (  # rebind-ok: invocation-scoped RPC state
                encode_json(replacement) if replacement is not None else None
            )  # rebind-ok: invocation-scoped RPC state
            if replacement_json is not None and replacement_json != original_response_json:
                return pb.GuardrailResult(
                    operation=operation_ok(),
                    decision=pb.GUARDRAIL_DECISION_REPLACE_RESPONSE,
                    response_json=replacement_json,
                )
        else:
            replacement = result if isinstance(result, dict) else request_data  # rebind-ok: invocation-scoped RPC state
            replacement_json = encode_json(replacement)  # rebind-ok: invocation-scoped RPC state
            if replacement_json != original_request_json:
                return pb.GuardrailResult(
                    operation=operation_ok(),
                    decision=pb.GUARDRAIL_DECISION_REPLACE_REQUEST,
                    request_json=replacement_json,
                )
        return pb.GuardrailResult(operation=operation_ok(), decision=pb.GUARDRAIL_DECISION_ALLOW)

    async def _publish_callback_event(self, event: pb.CallbackEvent) -> pb.OperationResult:
        method_name = (  # rebind-ok: invocation-scoped RPC state
            "async_log_stream_event"
            if event.streaming
            else "async_log_success_event"
            if event.kind == pb.CALLBACK_EVENT_KIND_SUCCESS
            else "async_log_failure_event"
        )
        try:
            async with self._store.acquire(event.context.active_revision, event.plugin_id) as extension:
                if method_name not in extension.hooks:
                    return operation_error(
                        pb.ERROR_CODE_UNSUPPORTED_HOOK,
                        f"extension {event.plugin_id!r} does not implement {method_name}",
                    )
                kwargs: Final = decode_json_object(event.standard_logging_payload_json, "standard_logging_payload_json")
                if event.HasField("error_json"):
                    kwargs["extension_error"] = decode_json_object(event.error_json, "error_json")
                response = (  # rebind-ok: invocation-scoped RPC state
                    response_from_json(event.response_json) if event.HasField("response_json") else None
                )  # rebind-ok: invocation-scoped RPC state
                start_time: Final = datetime.fromtimestamp(event.start_time_seconds, tz=timezone.utc)
                end_time: Final = datetime.fromtimestamp(event.end_time_seconds, tz=timezone.utc)
                target: Final = extension.target
                if extension.callable_target and not hasattr(target, method_name):
                    callback: Final = cast(Callable[..., object], target)  # cast-ok: validated protobuf boundary
                    await _maybe_await(callback(kwargs, response, start_time, end_time))
                else:
                    await _maybe_await(getattr(target, method_name)(kwargs, response, start_time, end_time))
                return operation_ok()
        except LookupError as error:
            return operation_error(pb.ERROR_CODE_NOT_ACTIVE, str(error))
        except Exception as error:  # noqa: BLE001  # map arbitrary customer callback failures
            return operation_error(pb.ERROR_CODE_EXTENSION_FAILED, str(error))

    async def _transform_stream(
        self,
        extension: LoadedExtension,
        stream_id: str,
        stream_open: pb.StreamOpen,
        frames: AsyncIterator[pb.StreamFrame],
    ) -> AsyncIterator[pb.StreamFrame]:
        auth = auth_from_proto(stream_open.auth)  # rebind-ok: invocation-scoped RPC state
        request_data = decode_json_object(  # rebind-ok: invocation-scoped RPC state
            stream_open.request_json, "request_json"
        )  # rebind-ok: invocation-scoped RPC state
        iterator_hook: Final = "async_post_call_streaming_iterator_hook"
        chunk_hook: Final = "async_post_call_streaming_hook"
        if stream_open.iterator_hook:
            if iterator_hook not in extension.hooks:
                yield _stream_error(stream_id, f"extension does not implement {iterator_hook}")
                return
            method = getattr(extension.target, iterator_hook)  # rebind-ok: invocation-scoped RPC state
            transformed: Final = method(auth, _stream_objects(frames, stream_id), request_data)
            async for chunk in transformed:
                yield _output_frame(stream_id, chunk)
            yield pb.StreamFrame(kind=pb.STREAM_FRAME_KIND_END, stream_id=stream_id)
            return
        async for frame in frames:
            if frame.stream_id != stream_id:
                raise ValueError("stream_id changed during TransformStream")
            if frame.kind == pb.STREAM_FRAME_KIND_END:
                yield frame
                return
            if frame.kind == pb.STREAM_FRAME_KIND_ERROR:
                yield frame
                return
            if frame.kind != pb.STREAM_FRAME_KIND_INPUT_CHUNK or not frame.HasField("chunk_json"):
                raise ValueError("expected INPUT_CHUNK, END, or ERROR")
            chunk = response_from_json(frame.chunk_json)
            output = chunk
            if chunk_hook in extension.hooks:
                result = await _maybe_await(getattr(extension.target, chunk_hook)(auth, chunk))
                if result is not None:
                    output = result
            if "async_log_stream_event" in extension.hooks:
                now = datetime.fromtimestamp(time.time(), tz=timezone.utc)
                stream_logger = getattr(  # noqa: B009  # hook name is validated during prepare
                    extension.target, "async_log_stream_event"
                )  # rebind-ok: each stream frame invokes the validated hook
                await _maybe_await(stream_logger(request_data, output, now, now))
            yield _output_frame(stream_id, output)

    def _cache_for(self, cache_ref: pb.CacheRef | None) -> CacheClient | None:
        if cache_ref is None or self._gateway_stub is None:
            return None
        return CacheClient(self._gateway_stub, cache_ref, self._token)

    async def _authenticate(self, context: grpc.aio.ServicerContext) -> None:
        metadata_items: Final = cast(  # cast-ok: validated protobuf boundary
            Iterable[tuple[str, str]], context.invocation_metadata() or ()
        )  # cast-ok: validated protobuf boundary
        metadata: Final = dict(metadata_items)  # mutable-ok: LiteLLM compatibility payload
        if metadata.get(TOKEN_METADATA_KEY) != self._token:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid extension host token")


async def _maybe_await(value: object) -> object:
    return await value if inspect.isawaitable(value) else value


async def _stream_objects(frames: AsyncIterator[pb.StreamFrame], stream_id: str) -> AsyncIterator[object]:
    async for frame in frames:
        if frame.stream_id != stream_id:
            raise ValueError("stream_id changed during TransformStream")
        if frame.kind == pb.STREAM_FRAME_KIND_END:
            return
        if frame.kind == pb.STREAM_FRAME_KIND_ERROR:
            raise RuntimeError(frame.error.message if frame.HasField("error") else "upstream stream failed")
        if frame.kind != pb.STREAM_FRAME_KIND_INPUT_CHUNK or not frame.HasField("chunk_json"):
            raise ValueError("expected INPUT_CHUNK, END, or ERROR")
        yield response_from_json(frame.chunk_json)


def _output_frame(stream_id: str, value: object) -> pb.StreamFrame:
    return pb.StreamFrame(
        kind=pb.STREAM_FRAME_KIND_OUTPUT_CHUNK,
        stream_id=stream_id,
        chunk_json=encode_json(value),
    )


def _stream_error(stream_id: str, message: str) -> pb.StreamFrame:
    return pb.StreamFrame(
        kind=pb.STREAM_FRAME_KIND_ERROR,
        stream_id=stream_id,
        error=pb.PublicError(type="extension_error", message=message),
    )


def _guardrail_error(code: pb.ErrorCode, message: str) -> pb.GuardrailResult:
    return pb.GuardrailResult(
        operation=operation_error(code, message),
        decision=pb.GUARDRAIL_DECISION_ERROR,
    )


def _blocked_result(error: Exception) -> pb.GuardrailResult:
    status_code: Final = getattr(error, "status_code", 400)
    public_error: Final = pb.PublicError(type=type(error).__name__, message=str(error))
    if isinstance(status_code, int):
        public_error.status_code = status_code
    return pb.GuardrailResult(
        operation=operation_ok(),
        decision=pb.GUARDRAIL_DECISION_BLOCK,
        public_error=public_error,
    )
