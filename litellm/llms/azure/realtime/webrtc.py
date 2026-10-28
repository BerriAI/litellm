import asyncio
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Protocol
from urllib.parse import urlencode, urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid5

import httpx
from pydantic import TypeAdapter
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError, WebSocketException

from litellm._logging import verbose_proxy_logger
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure.common_utils import AzureOpenAIError
from litellm.llms.azure.realtime.http_transformation import AzureRealtimeHTTPConfig
from litellm.llms.custom_httpx.http_handler import realtime_ssl_for_url
from litellm.types.utils import CallTypes

_EVENT: Final = TypeAdapter(Mapping[str, object])
_HEADER_VALUE: Final = TypeAdapter(str)
_TRANSCRIPTION_COMPLETED: Final = "conversation.item.input_audio_transcription.completed"


def _object(value: object) -> Mapping[str, object]:
    return _EVENT.validate_python(value) if isinstance(value, Mapping) else MappingProxyType({})


def azure_webrtc_sideband_url(api_base: str, location: str) -> tuple[str, str]:
    endpoint: Final = urlsplit(AzureRealtimeHTTPConfig().get_realtime_calls_url(api_base, model=""))
    target: Final = urlsplit(location)
    prefix, separator, call_id = target.path.rpartition("/")
    if (
        prefix not in (endpoint.path, "/v1/realtime/calls")
        or not separator
        or re.fullmatch(r"[a-zA-Z0-9_-]+", call_id) is None
        or target.query
        or target.fragment
    ):
        raise AzureOpenAIError(status_code=502, message="Azure realtime returned an invalid call Location")
    scheme: Final = "wss" if endpoint.scheme == "https" else "ws"
    return urlunsplit(
        (scheme, endpoint.netloc, endpoint.path.removesuffix("/calls"), urlencode({"call_id": call_id}), "")
    ), call_id


@dataclass(frozen=True, slots=True)
class AzureWebRTCUsage:
    identity: str
    model: str
    results: tuple[Mapping[str, object], ...]


class AzureWebRTCUsageCollector:
    def __init__(self, model: str, session: Mapping[str, object]) -> None:
        self.model: Final = model
        self.session: Mapping[str, object] = session
        self.seen: frozenset[str] = frozenset()
        self.item_models: Mapping[str, str] = MappingProxyType({})

    def transcription_model(self) -> str | None:
        audio: Final = _object(self.session.get("audio"))
        audio_input: Final = _object(audio.get("input"))
        transcription: Final = _object(
            audio_input.get("transcription") or self.session.get("input_audio_transcription")
        )
        model: Final = transcription.get("model")
        if isinstance(model, str) and model:
            return model
        return self.model if self.session.get("type") == "transcription" else None

    def observe(self, event: Mapping[str, object]) -> AzureWebRTCUsage | None:
        kind: Final = event.get("type")
        if kind in (
            "session.created",
            "session.updated",
            "transcription_session.created",
            "transcription_session.updated",
        ):
            self.session = _object(event.get("session"))
            return None
        item_id: Final = event.get("item_id")
        if kind == "input_audio_buffer.committed":
            model: Final = self.transcription_model()
            if isinstance(item_id, str) and model is not None:
                self.item_models = MappingProxyType({**self.item_models, item_id: model})
            return None
        if kind == "response.done":
            response: Final = _object(event.get("response"))
            response_id: Final = response.get("id")
            if not isinstance(response_id, str) or not isinstance(response.get("usage"), Mapping):
                return None
            return self._record(f"response:{response_id}", self.model, event)
        if kind != _TRANSCRIPTION_COMPLETED or not isinstance(event.get("usage"), Mapping):
            return None
        transcription_model: Final = (
            self.item_models.get(item_id) if isinstance(item_id, str) else None
        ) or self.transcription_model()
        if transcription_model is None:
            raise AzureOpenAIError(status_code=502, message="Azure WebRTC transcription usage has no known model")
        identity: Final = item_id if isinstance(item_id, str) else event.get("event_id")
        if not isinstance(identity, str):
            raise AzureOpenAIError(status_code=502, message="Azure WebRTC transcription usage has no event identity")
        return self._record(
            f"transcription:{identity}:{event.get('content_index', 0)}",
            transcription_model,
            {
                "type": _TRANSCRIPTION_COMPLETED,
                "usage": event["usage"],
                "litellm_transcription_model": transcription_model,
            },
        )

    def _record(self, identity: str, model: str, event: Mapping[str, object]) -> AzureWebRTCUsage | None:
        if identity in self.seen:
            return None
        self.seen = self.seen | {identity}
        return AzureWebRTCUsage(identity, model, ({"type": "session.updated", "session": self.session}, event))


class _SidebandSocket(Protocol):
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...
    async def close(self) -> None: ...


class AzureWebRTCObserver:
    def __init__(
        self,
        socket: _SidebandSocket,
        call_id: str,
        logging_obj: Logging,
        collector: AzureWebRTCUsageCollector,
    ) -> None:
        self.socket: Final = socket
        self.call_id: Final = call_id
        self.logging_obj: Final = logging_obj
        self.collector: Final = collector

    def _logging_call(self, identity: str, status: str, model: str | None = None) -> Logging:
        call: Final = self.logging_obj.copy_for_new_call(
            str(uuid5(NAMESPACE_URL, f"azure-realtime:{self.call_id}:{identity}")), datetime.now(timezone.utc)
        )
        call.call_type = CallTypes.arealtime.value if model is not None else "arealtime_calls"
        details: Final = _EVENT.validate_python(call.model_call_details)
        params: Final = _object(details.get("litellm_params"))
        metadata: Final = _object(params.get("metadata"))
        litellm_metadata: Final = _object(params.get("litellm_metadata"))
        observer_metadata: Final = {
            "webrtc_call_id": self.call_id,
            "webrtc_accounting_status": status,
            "webrtc_usage_model": model,
        }
        new_params: Final = {
            **params,
            "metadata": {
                **metadata,
                **observer_metadata,
                "spend_logs_metadata": {**_object(metadata.get("spend_logs_metadata")), **observer_metadata},
            },
            "litellm_metadata": {
                **litellm_metadata,
                **observer_metadata,
                "spend_logs_metadata": {**_object(litellm_metadata.get("spend_logs_metadata")), **observer_metadata},
            },
        }
        call.litellm_params = new_params
        call.model_call_details = {
            **{
                key: value
                for key, value in details.items()
                if not key.startswith("has_logged_")
                and key not in ("standard_logging_object", "response_cost", "cost_breakdown", "end_time")
            },
            "call_type": call.call_type,
            "litellm_params": new_params,
            "litellm_session_id": self.call_id,
        }
        call.litellm_session_id = self.call_id
        return call

    async def run(self) -> None:
        try:
            async for raw in self.socket:
                await self._observe(raw)
        except asyncio.CancelledError:
            await self._log_incomplete("observer_shutdown")
            raise
        except Exception as error:  # noqa: BLE001  # Provider and logging callback failures must mark accounting incomplete.
            status: Final = "observer_disconnected" if isinstance(error, ConnectionClosedError) else "observer_failed"
            await self._log_incomplete(status)
        else:
            lifecycle: Final = self._logging_call("observer_closed", "observer_closed")
            await lifecycle.dispatch_success_handlers(  # pyright: ignore[reportUnknownMemberType]  # Legacy dispatcher is untyped.
                None, prefer_async_handlers=True
            )
        finally:
            await self.socket.close()

    async def _observe(self, raw: str | bytes) -> None:
        event: Final = _EVENT.validate_json(raw)
        usage: Final = self.collector.observe(event)
        if usage is None:
            return
        call: Final = self._logging_call(usage.identity, "observed", usage.model)
        await call.dispatch_success_handlers(  # pyright: ignore[reportUnknownMemberType]  # Legacy dispatcher is untyped.
            list(usage.results), prefer_async_handlers=True
        )

    async def _log_incomplete(self, status: str) -> None:
        verbose_proxy_logger.warning("Azure WebRTC accounting incomplete (call_id=%s, status=%s)", self.call_id, status)
        call: Final = self._logging_call(status, status)
        await call.dispatch_failure_handlers(
            AzureOpenAIError(status_code=502, message=f"Azure WebRTC accounting incomplete: {status}"),
            traceback_exception="",
            prefer_async_handlers=True,
        )


class AzureWebRTCObservers:
    def __init__(self) -> None:
        self.tasks: frozenset[asyncio.Task[None]] = frozenset()

    async def attach(
        self,
        response: httpx.Response,
        api_base: str,
        auth_headers: Mapping[str, str],
        model: str,
        session: Mapping[str, object],
        logging_obj: Logging,
    ) -> None:
        location: Final = _HEADER_VALUE.validate_python(
            response.headers.get("location", "")  # pyright: ignore[reportAny]  # httpx header stubs return Any.
        )
        url, call_id = azure_webrtc_sideband_url(api_base, location)
        try:
            socket: Final = await connect(
                url,
                additional_headers=auth_headers,
                ssl=realtime_ssl_for_url(url),
                open_timeout=10,
                close_timeout=5,
                max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
            )
        except (OSError, TimeoutError, WebSocketException):
            raise AzureOpenAIError(status_code=502, message="Unable to attach Azure WebRTC usage observer") from None
        observer: Final = AzureWebRTCObserver(socket, call_id, logging_obj, AzureWebRTCUsageCollector(model, session))
        task: Final = asyncio.create_task(observer.run(), name=f"azure-webrtc:{call_id}")
        self.tasks = self.tasks | {task}
        task.add_done_callback(self._finished)
        verbose_proxy_logger.info("Azure WebRTC observer attached (call_id=%s)", call_id)

    def _finished(self, task: asyncio.Task[None]) -> None:
        self.tasks = self.tasks - {task}
        if not task.cancelled() and task.exception() is not None:
            verbose_proxy_logger.error("Azure WebRTC observer cleanup failed")

    async def stop(self) -> None:
        tasks: Final = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


azure_webrtc_observers: Final = AzureWebRTCObservers()
