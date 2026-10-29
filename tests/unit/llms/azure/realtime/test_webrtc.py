import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import datetime, timezone
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.azure.common_utils import AzureOpenAIError
from litellm.llms.azure.realtime import webrtc
from litellm.llms.azure.realtime.webrtc import (
    AzureWebRTCObserver,
    AzureWebRTCObservers,
    AzureWebRTCUsageCollector,
    azure_webrtc_sideband_url,
)
from litellm.types.utils import CallTypes

_OBJECT: Final = TypeAdapter(Mapping[str, object])
_ASR_EVENT: Final = "conversation.item.input_audio_transcription.completed"


@pytest.mark.parametrize("suffix", ["", "/openai", "/openai/v1", "/openai/v1/"])
def test_sideband_attaches_to_call_on_configured_azure_resource(suffix: str) -> None:
    assert azure_webrtc_sideband_url(
        f"https://audio.example{suffix}", "https://audio.example/openai/v1/realtime/calls/call_123"
    ) == ("wss://audio.example/openai/v1/realtime?call_id=call_123", "call_123")


@pytest.mark.parametrize(
    "location",
    [
        "/unrelated/call_123",
        "/v1/realtime/calls/call%2F123",
        "/openai/v1/realtime/calls/",
        "/openai/v1/realtime/calls/call_123/extra",
        "/openai/v1/realtime/calls/call_123?key=secret",
        "",
    ],
)
def test_sideband_refuses_untrusted_or_missing_call_location(location: str) -> None:
    with pytest.raises(AzureOpenAIError, match="invalid call Location"):
        azure_webrtc_sideband_url("https://audio.example", location)


def test_sideband_uses_configured_azure_host_even_when_location_names_another_host() -> None:
    assert azure_webrtc_sideband_url(
        "https://audio.example", "https://other.example/v1/realtime/calls/call_123"
    ) == ("wss://audio.example/openai/v1/realtime?call_id=call_123", "call_123")


def _session(model: str) -> Mapping[str, object]:
    return {"type": "realtime", "model": "rt", "audio": {"input": {"transcription": {"model": model}}}}


def _transcription(item: str, seconds: float) -> Mapping[str, object]:
    return {"type": _ASR_EVENT, "item_id": item, "content_index": 0, "usage": {"type": "duration", "seconds": seconds}}


def test_late_transcription_belongs_to_model_at_audio_commit_and_is_billed_once() -> None:
    collector: Final = AzureWebRTCUsageCollector("rt", _session("asr-first"))
    collector.observe({"type": "input_audio_buffer.committed", "item_id": "item-first"})
    collector.observe({"type": "session.updated", "session": _session("asr-second")})
    collector.observe({"type": "input_audio_buffer.committed", "item_id": "item-second"})
    first: Final = collector.observe(_transcription("item-first", 3.0))
    second: Final = collector.observe(_transcription("item-second", 4.0))
    assert first is not None and second is not None
    assert (first.model, second.model) == ("asr-first", "asr-second")
    assert first.results[-1] == {
        "type": _ASR_EVENT,
        "usage": {"type": "duration", "seconds": 3.0},
        "litellm_transcription_model": "asr-first",
    }
    assert collector.observe(_transcription("item-first", 3.0)) is None


class _AzureSocket:
    def __init__(self, events: tuple[Mapping[str, object], ...], failure: Exception | None = None) -> None:
        self.events: Final = events
        self.failure: Final = failure
        self.closed: bool = False

    async def __aiter__(self) -> AsyncIterator[str]:
        for event in self.events:
            yield json.dumps(event)
        if self.failure is not None:
            raise self.failure

    async def close(self) -> None:
        self.closed = True


class _SpendCapture(CustomLogger):
    def __init__(self) -> None:
        self.success: tuple[Mapping[str, object], ...] = ()
        self.failure: tuple[Mapping[str, object], ...] = ()

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.success = (*self.success, _OBJECT.validate_python(kwargs.get("standard_logging_object") or kwargs))

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.failure = (*self.failure, _OBJECT.validate_python(kwargs["standard_logging_object"]))


@pytest.fixture
def priced_models(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "azure/rt": {
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
                "litellm_provider": "azure",
                "mode": "chat",
            },
            "azure/asr-first": {
                "input_cost_per_second": 0.01,
                "litellm_provider": "azure",
                "mode": "audio_transcription",
            },
            "azure/asr-second": {
                "input_cost_per_second": 0.02,
                "litellm_provider": "azure",
                "mode": "audio_transcription",
            },
        },
    )
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


def _logging(recorder: _SpendCapture) -> Logging:
    metadata: Final = {"user_api_key": "a" * 64, "user_api_key_user_id": "user-1", "user_api_key_team_id": "team-1"}
    logging: Final = Logging(
        model="rt",
        messages=[],
        stream=False,
        call_type="arealtime_calls",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        litellm_call_id="signaling",
        function_id="signaling",
        dynamic_async_success_callbacks=[recorder],
        dynamic_async_failure_callbacks=[recorder],
    )
    logging.update_from_kwargs(
        kwargs={"metadata": metadata}, model="rt", custom_llm_provider="azure", litellm_params={}
    )
    logging.model_call_details["has_logged_async_success"] = True
    return logging


@pytest.mark.asyncio
async def test_observer_reports_parent_tokens_and_each_asr_model_cost_to_original_key(priced_models: None) -> None:
    response: Final[Mapping[str, object]] = {
        "type": "response.done",
        "response": {
            "id": "resp_1",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "output": [],
        },
    }
    socket: Final = _AzureSocket(
        (
            {"type": "session.updated", "session": _session("asr-first")},
            {"type": "input_audio_buffer.committed", "item_id": "item-first"},
            {"type": "session.updated", "session": _session("asr-second")},
            _transcription("item-first", 3.0),
            _transcription("item-first", 3.0),
            _transcription("item-second", 4.0),
            response,
            response,
        )
    )
    recorder: Final = _SpendCapture()
    await AzureWebRTCObserver(socket, "call_1", _logging(recorder), AzureWebRTCUsageCollector("rt", {})).run()
    charged: Final = tuple(
        record for record in recorder.success if record.get("call_type") == CallTypes.arealtime.value
    )
    assert len(charged) == 3
    assert tuple(record["response_cost"] for record in charged) == pytest.approx(
        (3 * 0.01, 4 * 0.02, 10 * 0.001 + 5 * 0.002)
    )
    assert tuple(record["total_tokens"] for record in charged) == (0, 0, 15)
    metadata: Final = tuple(_OBJECT.validate_python(record["metadata"]) for record in charged)
    assert (
        tuple(
            (item["user_api_key_hash"], item["user_api_key_user_id"], item["user_api_key_team_id"]) for item in metadata
        )
        == (("a" * 64, "user-1", "team-1"),) * 3
    )
    assert (
        tuple(_OBJECT.validate_python(item["spend_logs_metadata"])["webrtc_call_id"] for item in metadata)
        == ("call_1",) * 3
    )
    assert len(frozenset(record["id"] for record in charged)) == 3
    assert recorder.failure == ()
    assert socket.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError("connection refused"), TimeoutError("connection timed out")])
async def test_unreachable_observer_refuses_call_admission(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    async def unavailable(url: str, **kwargs: object) -> _AzureSocket:
        raise failure

    monkeypatch.setattr(webrtc, "connect", unavailable)
    response: Final = httpx.Response(201, headers={"location": "/openai/v1/realtime/calls/call_1"})
    with pytest.raises(AzureOpenAIError, match="Unable to attach Azure WebRTC usage observer") as refused:
        await AzureWebRTCObservers().attach(
            response, "https://audio.example", {"api-key": "test-key"}, "rt", {}, _logging(_SpendCapture())
        )
    assert refused.value.status_code == 502


class _ActiveAzureSocket:
    def __init__(self) -> None:
        self.observing: Final = asyncio.Event()
        self.closed: bool = False

    async def __aiter__(self) -> AsyncIterator[str]:
        yield json.dumps(_transcription("item-1", 3.0))
        self.observing.set()
        await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_proxy_shutdown_closes_observer_and_reports_incomplete_accounting_once(
    monkeypatch: pytest.MonkeyPatch, priced_models: None
) -> None:
    socket: Final = _ActiveAzureSocket()

    async def connected(url: str, **kwargs: object) -> _ActiveAzureSocket:
        return socket

    monkeypatch.setattr(webrtc, "connect", connected)
    observers: Final = AzureWebRTCObservers()
    recorder: Final = _SpendCapture()
    response: Final = httpx.Response(201, headers={"location": "/openai/v1/realtime/calls/call_1"})
    await observers.attach(
        response, "https://audio.example", {"api-key": "test-key"}, "rt", _session("asr-first"), _logging(recorder)
    )
    try:
        await socket.observing.wait()
    finally:
        await observers.stop()
    await observers.stop()
    assert socket.closed
    assert tuple(record["response_cost"] for record in recorder.success) == pytest.approx((3 * 0.01,))
    assert len(recorder.failure) == 1
    metadata: Final = _OBJECT.validate_python(recorder.failure[0]["metadata"])
    assert _OBJECT.validate_python(metadata["spend_logs_metadata"])["webrtc_accounting_status"] == "observer_shutdown"


@pytest.mark.asyncio
async def test_observer_failure_is_explicit_and_does_not_rebill_completed_usage(priced_models: None) -> None:
    socket: Final = _AzureSocket((_transcription("item-1", 3.0),), RuntimeError("upstream dropped"))
    recorder: Final = _SpendCapture()
    await AzureWebRTCObserver(
        socket,
        "call_1",
        _logging(recorder),
        AzureWebRTCUsageCollector("rt", _session("asr-first")),
    ).run()
    assert tuple(record["response_cost"] for record in recorder.success) == pytest.approx((3 * 0.01,))
    assert len(recorder.failure) == 1
    assert "accounting incomplete" in str(recorder.failure[0]["error_information"])
    assert socket.closed
