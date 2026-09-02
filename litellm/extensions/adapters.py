# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false
# pyright: reportMissingParameterType=false
from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from datetime import datetime
from typing import Final

from litellm.caching import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.python_extension.generated.v1 import extension_host_pb2 as pb
from litellm.python_extension_host.compatibility import decode_json, encode_json, response_from_json
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.utils import CallTypesLiteral, ModelResponseStream

from .cache_gateway import InvocationCacheRegistry
from .client import PythonExtensionClient


class _RemoteHooks:
    def __init__(
        self,
        client: PythonExtensionClient,
        plugin_id: str,
        cache_registry: InvocationCacheRegistry,
    ) -> None:
        self._extension_client: Final = client
        self._extension_plugin_id: Final = plugin_id
        self._extension_cache_registry: Final = cache_registry

    async def execute(
        self,
        phase: pb.HookPhase,
        data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        auth: UserAPIKeyAuth,
        call_type: str,
        cache: object | None = None,
        response: object | None = None,
    ) -> object | None:
        context = _invocation_context(  # rebind-ok: invocation-scoped RPC state
            data, self._extension_client.manifest.revision_id, call_type
        )  # rebind-ok: invocation-scoped RPC state
        cache_ref: Final = self._extension_cache_registry.register(context.invocation_id, cache)
        invocation: Final = pb.GuardrailInvocation(
            context=context,
            plugin_id=self._extension_plugin_id,
            hook_phase=phase,
            request_json=encode_json(data),
            auth=_auth_context(auth),
        )
        if response is not None:
            invocation.response_json = encode_json(response)
        if cache_ref is not None:
            invocation.cache.CopyFrom(cache_ref)
        try:
            result = await self._extension_client.execute_guardrail(  # rebind-ok: invocation-scoped RPC state
                invocation
            )  # rebind-ok: invocation-scoped RPC state
        finally:
            self._extension_cache_registry.revoke(cache_ref)
        if result.decision in (
            pb.GUARDRAIL_DECISION_ALLOW,
            pb.GUARDRAIL_DECISION_ERROR,
            pb.GUARDRAIL_DECISION_UNSPECIFIED,
        ):
            return response if phase == pb.HOOK_PHASE_POST_CALL else None
        if result.decision == pb.GUARDRAIL_DECISION_BLOCK:
            public_error: Final = result.public_error if result.HasField("public_error") else None
            raise GuardrailRaisedException(
                guardrail_name=self._extension_plugin_id,
                message=public_error.message if public_error is not None else "request blocked by extension",
                should_wrap_with_default_message=False,
                status_code=(
                    public_error.status_code
                    if public_error is not None and public_error.HasField("status_code")
                    else 400
                ),
                blocked_content=True,
            )
        if result.decision == pb.GUARDRAIL_DECISION_REPLACE_REQUEST and result.HasField("request_json"):
            replacement: Final = decode_json(result.request_json, "request_json")
            if isinstance(replacement, dict):
                data.clear()
                data.update(replacement)
                return data
        if result.decision == pb.GUARDRAIL_DECISION_REPLACE_RESPONSE and result.HasField("response_json"):
            return response_from_json(result.response_json)
        return response if phase == pb.HOOK_PHASE_POST_CALL else None

    def enqueue_event(
        self,
        kind: pb.CallbackEventKind,
        kwargs: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        response: object,
        start_time: object,
        end_time: object,
        streaming: bool = False,
    ) -> None:
        context = _invocation_context(  # rebind-ok: invocation-scoped RPC state
            kwargs,
            self._extension_client.manifest.revision_id,
            str(kwargs.get("call_type", "unknown")),
        )
        payload: Final = kwargs.get("standard_logging_payload", kwargs)
        event: Final = pb.CallbackEvent(
            context=context,
            plugin_id=self._extension_plugin_id,
            kind=kind,
            standard_logging_payload_json=encode_json(payload),
            start_time_seconds=_timestamp(start_time),
            end_time_seconds=_timestamp(end_time),
            auth=_auth_context_from_kwargs(kwargs),
            streaming=streaming,
        )
        if response is not None and not isinstance(response, Exception):
            event.response_json = encode_json(response)
        if isinstance(response, Exception):
            event.error_json = encode_json(
                {"type": type(response).__name__, "message": str(response)}  # mutable-ok: LiteLLM compatibility payload
            )  # mutable-ok: LiteLLM compatibility payload
        self._extension_client.enqueue_callback(event)

    async def transform_iterator(
        self,
        auth: UserAPIKeyAuth,
        response: AsyncIterator[ModelResponseStream],
        request_data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
    ) -> AsyncGenerator[ModelResponseStream, None]:
        context = _invocation_context(  # rebind-ok: invocation-scoped RPC state
            request_data,
            self._extension_client.manifest.revision_id,
            str(request_data.get("call_type", "stream")),
        )
        pending: list[  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
            ModelResponseStream
        ] = []  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state

        async def frames() -> AsyncIterator[pb.StreamFrame]:
            yield pb.StreamFrame(
                kind=pb.STREAM_FRAME_KIND_OPEN,
                stream_id=context.invocation_id,
                open=pb.StreamOpen(
                    context=context,
                    plugin_id=self._extension_plugin_id,
                    request_json=encode_json(request_data),
                    auth=_auth_context(auth),
                    iterator_hook=(
                        "async_post_call_streaming_iterator_hook"
                        in self._extension_client.descriptor_hooks(self._extension_plugin_id)
                    ),
                ),
            )
            async for chunk in response:
                pending.append(chunk)
                yield pb.StreamFrame(
                    kind=pb.STREAM_FRAME_KIND_INPUT_CHUNK,
                    stream_id=context.invocation_id,
                    chunk_json=encode_json(chunk),
                )
            yield pb.StreamFrame(kind=pb.STREAM_FRAME_KIND_END, stream_id=context.invocation_id)

        completed = False  # rebind-ok: invocation-scoped RPC state
        async for frame in self._extension_client.transform_stream(frames()):
            if frame.kind == pb.STREAM_FRAME_KIND_OUTPUT_CHUNK and frame.HasField("chunk_json"):
                transformed = response_from_json(frame.chunk_json)
                if isinstance(transformed, ModelResponseStream):
                    yield transformed
            elif frame.kind == pb.STREAM_FRAME_KIND_END:
                completed = True
            elif frame.kind == pb.STREAM_FRAME_KIND_ERROR:
                break
        if not completed:
            for chunk in pending:
                yield chunk
            async for chunk in response:
                yield chunk


class RemoteCustomLogger(CustomLogger):
    def __init__(
        self,
        client: PythonExtensionClient,
        plugin_id: str,
        cache_registry: InvocationCacheRegistry,
    ) -> None:
        super().__init__()
        self._remote = _RemoteHooks(client, plugin_id, cache_registry)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict[str, object] | None:  # mutable-ok: LiteLLM compatibility payload
        result = await self._remote.execute(  # rebind-ok: invocation-scoped RPC state
            pb.HOOK_PHASE_PRE_CALL,
            data,
            user_api_key_dict,
            call_type,
            cache=cache,
        )
        return result if isinstance(result, Exception | str | dict) else None

    async def async_moderation_hook(
        self,
        data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> object | None:
        return await self._remote.execute(pb.HOOK_PHASE_DURING_CALL, data, user_api_key_dict, call_type)

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
    ) -> object | None:
        return await self._remote.execute(
            pb.HOOK_PHASE_POST_CALL,
            data,
            user_api_key_dict,
            str(data.get("call_type", "unknown")),
            response=response,
        )

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._remote.enqueue_event(pb.CALLBACK_EVENT_KIND_SUCCESS, kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._remote.enqueue_event(pb.CALLBACK_EVENT_KIND_FAILURE, kwargs, response_obj, start_time, end_time)

    async def async_log_stream_event(
        self,
        kwargs: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._remote.enqueue_event(
            pb.CALLBACK_EVENT_KIND_SUCCESS,
            kwargs,
            response_obj,
            start_time,
            end_time,
            streaming=True,
        )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterator[ModelResponseStream],
        request_data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
    ) -> AsyncGenerator[ModelResponseStream, None]:
        async for chunk in self._remote.transform_iterator(user_api_key_dict, response, request_data):
            yield chunk


class RemoteCustomGuardrail(CustomGuardrail):
    def __init__(
        self,
        client: PythonExtensionClient,
        plugin_id: str,
        cache_registry: InvocationCacheRegistry,
        guardrail_name: str | None = None,
        event_hook: GuardrailEventHooks  # mutable-ok: LiteLLM compatibility payload
        | list[GuardrailEventHooks]
        | Mode
        | None = None,  # mutable-ok: LiteLLM compatibility payload
        default_on: bool = False,
        extra_params: Mapping[str, object] | None = None,
    ) -> None:
        params = (  # rebind-ok: invocation-scoped RPC state
            extra_params or {}  # mutable-ok: LiteLLM compatibility payload
        )  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            mask_request_content=_bool_param(params, "mask_request_content", False),
            mask_response_content=_bool_param(params, "mask_response_content", False),
            violation_message_template=_str_param(params, "violation_message_template"),
            end_session_after_n_fails=_int_param(params, "end_session_after_n_fails"),
            on_violation=_str_param(params, "on_violation"),
            realtime_violation_message=_str_param(params, "realtime_violation_message"),
            on_sensitive_data=_str_param(params, "on_sensitive_data"),
            sensitive_data_route_to_model=_str_param(params, "sensitive_data_route_to_model"),
            sticky_session_routing=_bool_param(params, "sticky_session_routing", True),
            run_in_parallel=_bool_param(params, "run_in_parallel", False),
            scan_raw_request=_bool_param(params, "scan_raw_request", False),
            only_scan_new_messages=_bool_param(params, "only_scan_new_messages", False),
            turn_off_message_logging=_bool_param(params, "turn_off_message_logging", False),
            message_logging=_bool_param(params, "message_logging", True),
        )
        self._remote = _RemoteHooks(client, plugin_id, cache_registry)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        call_type: CallTypesLiteral,
    ) -> Exception | str | dict[str, object] | None:  # mutable-ok: LiteLLM compatibility payload
        result = await self._remote.execute(  # rebind-ok: invocation-scoped RPC state
            pb.HOOK_PHASE_PRE_CALL,
            data,
            user_api_key_dict,
            call_type,
            cache=cache,
        )
        return result if isinstance(result, Exception | str | dict) else None

    async def async_moderation_hook(
        self,
        data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> object | None:
        return await self._remote.execute(pb.HOOK_PHASE_DURING_CALL, data, user_api_key_dict, call_type)

    async def async_post_call_success_hook(
        self,
        data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
        user_api_key_dict: UserAPIKeyAuth,
        response: object,
    ) -> object | None:
        return await self._remote.execute(
            pb.HOOK_PHASE_POST_CALL,
            data,
            user_api_key_dict,
            str(data.get("call_type", "unknown")),
            response=response,
        )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterator[ModelResponseStream],
        request_data: dict[str, object],  # mutable-ok: LiteLLM compatibility payload
    ) -> AsyncGenerator[ModelResponseStream, None]:
        async for chunk in self._remote.transform_iterator(user_api_key_dict, response, request_data):
            yield chunk


def _invocation_context(data: Mapping[str, object], revision: str, call_type: str) -> pb.InvocationContext:
    metadata = data.get("metadata")  # rebind-ok: invocation-scoped RPC state
    metadata_mapping = (  # rebind-ok: invocation-scoped RPC state
        metadata if isinstance(metadata, Mapping) else {}  # mutable-ok: LiteLLM compatibility payload
    )  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    request_id: Final = str(
        data.get("request_id") or data.get("litellm_call_id") or metadata_mapping.get("request_id") or ""
    )
    invocation_id: Final = str(
        data.get("litellm_call_id") or request_id or hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:24]
    )
    return pb.InvocationContext(
        request_id=request_id,
        invocation_id=invocation_id,
        active_revision=revision,
        api_surface=str(data.get("api_surface", data.get("call_type", "unknown"))),
        call_type=call_type,
        trace_context=_trace_context(metadata_mapping),
    )


def _auth_context(auth: UserAPIKeyAuth) -> pb.AuthContext:
    key = getattr(auth, "api_key", None) or getattr(auth, "token", None) or ""  # rebind-ok: invocation-scoped RPC state
    metadata = getattr(auth, "metadata", None)  # rebind-ok: invocation-scoped RPC state
    return pb.AuthContext(
        key_hash=_hash_key(str(key)) if key else "",
        user_id=str(getattr(auth, "user_id", None) or ""),
        team_id=str(getattr(auth, "team_id", None) or ""),
        request_metadata=_safe_metadata(
            metadata if isinstance(metadata, Mapping) else {}  # mutable-ok: LiteLLM compatibility payload
        ),  # mutable-ok: LiteLLM compatibility payload
    )


def _auth_context_from_kwargs(kwargs: Mapping[str, object]) -> pb.AuthContext:
    litellm_params: Final = kwargs.get("litellm_params")
    params = (  # rebind-ok: invocation-scoped RPC state
        litellm_params if isinstance(litellm_params, Mapping) else {}  # mutable-ok: LiteLLM compatibility payload
    )  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    metadata = params.get("metadata")  # rebind-ok: invocation-scoped RPC state
    metadata_mapping = (  # rebind-ok: invocation-scoped RPC state
        metadata if isinstance(metadata, Mapping) else {}  # mutable-ok: LiteLLM compatibility payload
    )  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    key = params.get("api_key") or metadata_mapping.get("user_api_key") or ""  # rebind-ok: invocation-scoped RPC state
    return pb.AuthContext(
        key_hash=_hash_key(str(key)) if key else "",
        user_id=str(metadata_mapping.get("user_api_key_user_id") or ""),
        team_id=str(metadata_mapping.get("user_api_key_team_id") or ""),
        request_metadata=_safe_metadata(metadata_mapping),
    )


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _safe_metadata(
    metadata: Mapping[object, object],
) -> dict[str, str]:  # mutable-ok: LiteLLM compatibility payload
    denied: Final = (
        "authorization",
        "api_key",
        "token",
        "cookie",
        "secret",
        "password",
    )
    return {  # mutable-ok: LiteLLM compatibility payload
        str(key): str(value)
        for key, value in metadata.items()
        if not any(part in str(key).lower() for part in denied) and isinstance(value, str | int | float | bool)
    }


def _trace_context(
    metadata: Mapping[object, object],
) -> dict[str, str]:  # mutable-ok: LiteLLM compatibility payload
    return {  # mutable-ok: LiteLLM compatibility payload
        name: str(metadata[name])
        for name in ("traceparent", "tracestate")
        if name in metadata and isinstance(metadata[name], str)
    }


def _timestamp(value: object) -> float:
    if isinstance(value, datetime):
        return float(value.timestamp())
    if isinstance(value, int | float):
        return float(value)
    return time.time()


def _bool_param(params: Mapping[str, object], key: str, default: bool) -> bool:
    value = params.get(key)  # rebind-ok: invocation-scoped RPC state
    return value if isinstance(value, bool) else default


def _str_param(params: Mapping[str, object], key: str) -> str | None:
    value = params.get(key)  # rebind-ok: invocation-scoped RPC state
    return value if isinstance(value, str) else None


def _int_param(params: Mapping[str, object], key: str) -> int | None:
    value = params.get(key)  # rebind-ok: invocation-scoped RPC state
    return value if isinstance(value, int) and not isinstance(value, bool) else None
