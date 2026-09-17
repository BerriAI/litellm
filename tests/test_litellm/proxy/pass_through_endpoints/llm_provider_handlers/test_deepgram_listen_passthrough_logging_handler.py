"""Deepgram ``/v1/listen`` WebSocket passthrough: duration extraction and duration based cost tracking."""

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.deepgram_listen_passthrough_logging_handler import (
    DeepgramListenPassthroughLoggingHandler,
    deepgram_listen_audio_seconds,
    deepgram_listen_model,
    deepgram_listen_transcript,
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import PassthroughStandardLoggingPayload
from litellm.types.utils import StandardLoggingPayload, TranscriptionResponse

NOVA_3_URL: Final = "wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16&sample_rate=16000"


def _results(start: object, duration: object, transcript: str = "", is_final: object = True) -> dict[str, object]:
    return {
        "type": "Results",
        "start": start,
        "duration": duration,
        "is_final": is_final,
        "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.9}]},
    }


def _metadata(duration: object) -> dict[str, object]:
    return {"type": "Metadata", "request_id": "req-1", "duration": duration, "channels": 1}


@pytest.mark.parametrize(
    ("frames", "expected_seconds"),
    [
        pytest.param((_results(0.0, 2.0), _results(2.0, 3.5), _metadata(6.25)), 6.25, id="metadata wins"),
        pytest.param((_metadata(4.0), _results(0.0, 9.0), _metadata(5.5)), 5.5, id="last metadata wins"),
        pytest.param((_results(0.0, 2.0), _results(2.0, 3.5), _results(1.0, 1.0)), 5.5, id="furthest results end"),
        pytest.param((_results(0.0, 0.0), _metadata(0.0)), 0.0, id="zero metadata is a real zero"),
        pytest.param((_results(0.0, 1.5), _metadata("6.25")), 1.5, id="string metadata is ignored"),
        pytest.param((_results(0.0, 1.5), _metadata(True)), 1.5, id="boolean metadata is ignored"),
        pytest.param((_results(0.0, 1.5), _metadata(-3.0)), 1.5, id="negative metadata is ignored"),
        pytest.param((_results(0.0, 1.5), _metadata(math.nan), _metadata(math.inf)), 1.5, id="nan/inf ignored"),
        pytest.param((_results("0", 2.0), _results(0.0, None), _results(0.0, 0.75)), 0.75, id="malformed results"),
        pytest.param(({"type": "SpeechStarted", "timestamp": 3.0}, {"type": "UtteranceEnd"}), 0.0, id="no usage"),
        pytest.param((), 0.0, id="no frames"),
    ],
)
def test_deepgram_listen_audio_seconds(frames: Sequence[Mapping[str, object]], expected_seconds: float):
    assert deepgram_listen_audio_seconds(frames) == expected_seconds


def test_deepgram_listen_transcript_joins_final_results_only():
    frames = (
        _results(0.0, 1.0, "hello wor", is_final=False),
        _results(0.0, 1.5, "hello world"),
        _results(1.5, 0.5, "", is_final=True),
        _results(2.0, 1.0, "how are you", is_final="yes"),
        {"type": "Results", "start": 3.0, "duration": 1.0, "is_final": True, "channel": {"alternatives": []}},
        _results(4.0, 1.0, "goodbye"),
        _metadata(5.0),
    )
    assert deepgram_listen_transcript(frames) == "hello world goodbye"


@pytest.mark.parametrize(
    ("upstream_url", "expected_model"),
    [
        (NOVA_3_URL, "nova-3"),
        ("wss://api.deepgram.com/v1/listen?encoding=linear16&model=nova-2-medical", "nova-2-medical"),
        ("wss://api.deepgram.com/v1/listen?model=nova-3&model=nova-2", "nova-3"),
        ("wss://api.deepgram.com/v1/listen?encoding=linear16", litellm.constants.DEEPGRAM_LISTEN_DEFAULT_MODEL),
    ],
)
def test_deepgram_listen_model_comes_from_the_upstream_query(upstream_url: str, expected_model: str):
    assert deepgram_listen_model(upstream_url) == expected_model


@pytest.mark.parametrize(
    ("url_route", "expected"),
    [
        ("/deepgram/v1/listen", True),
        ("/deepgram/listen", True),
        ("/deepgram/v1/listen?model=nova-3", True),
        ("/deepgram/v1/speak", False),
        ("/deepgram/v1/listen/extra", False),
        ("/openai/v1/realtime", False),
        ("/vertex_ai/live", False),
        ("", False),
    ],
)
def test_is_deepgram_listen_route(url_route: str, expected: bool):
    assert DeepgramListenPassthroughLoggingHandler.is_deepgram_listen_route(url_route) is expected


def _logging_obj(call_id: str = "call-dg") -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "WebSocket connection"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id=call_id,
        function_id="websocket_passthrough",
    )


def _registry_cost(model: str, seconds: float) -> float:
    """Derives the expected charge from the live cost map rather than pinning a vendor price."""
    per_second: Final = litellm.model_cost[f"deepgram/{model}"]["input_cost_per_second"]
    assert per_second > 0
    return per_second * seconds


def test_handler_bills_metadata_duration_at_the_registry_rate_and_names_the_model():
    frames = (_results(0.0, 5.0, "first sentence"), _results(5.0, 7.5, "second sentence"), _metadata(12.5))
    logging_obj = _logging_obj()

    handler_result = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=frames,
        logging_obj=logging_obj,
        upstream_url=NOVA_3_URL,
        kwargs={"litellm_params": {"metadata": {}}},
    )

    result = handler_result["result"]
    assert isinstance(result, TranscriptionResponse)
    assert result.text == "first sentence second sentence"
    assert result._hidden_params["audio_transcription_duration"] == 12.5
    assert result._hidden_params["response_cost"] == pytest.approx(_registry_cost("nova-3", 12.5))
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(_registry_cost("nova-3", 12.5))
    assert handler_result["kwargs"]["model"] == "nova-3"
    assert handler_result["kwargs"]["custom_llm_provider"] == "deepgram"
    assert handler_result["kwargs"]["litellm_params"] == {"metadata": {}}
    assert logging_obj.model == "nova-3"
    assert logging_obj.model_call_details["model"] == "nova-3"
    assert logging_obj.model_call_details["custom_llm_provider"] == "deepgram"
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(_registry_cost("nova-3", 12.5))


def test_handler_falls_back_to_results_frames_when_the_stream_ends_without_metadata():
    frames = (_results(0.0, 30.0, "a"), _results(30.0, 30.0, "b"), _results(60.0, 12.5, "c"))

    handler_result = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=frames, logging_obj=_logging_obj(), upstream_url=NOVA_3_URL
    )

    assert handler_result["result"]._hidden_params["audio_transcription_duration"] == 72.5
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(_registry_cost("nova-3", 72.5))


def test_handler_charges_more_for_more_audio_on_the_same_model():
    short = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=(_metadata(10.0),), logging_obj=_logging_obj(), upstream_url=NOVA_3_URL
    )
    long = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=(_metadata(30.0),), logging_obj=_logging_obj(), upstream_url=NOVA_3_URL
    )

    assert long["kwargs"]["response_cost"] == pytest.approx(3 * short["kwargs"]["response_cost"])
    assert short["kwargs"]["response_cost"] > 0


def test_handler_keeps_the_spend_row_but_no_cost_for_an_unpriced_model():
    handler_result = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=(_metadata(12.5),),
        logging_obj=_logging_obj(),
        upstream_url="wss://api.deepgram.com/v1/listen?model=nova-99-not-in-registry",
    )

    assert handler_result["kwargs"]["model"] == "nova-99-not-in-registry"
    assert handler_result["kwargs"]["response_cost"] is None
    assert handler_result["result"]._hidden_params["audio_transcription_duration"] == 12.5


class _CapturingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[StandardLoggingPayload] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs["standard_logging_object"])


@pytest.mark.asyncio
async def test_success_handler_dispatches_deepgram_listen_and_logs_duration_based_spend(monkeypatch):
    """Drives the shared passthrough success handler the way the WebSocket relay does at socket close and reads
    what a spend logger receives: Deepgram model and provider, the audio duration billed at the registry rate."""
    capturing_logger = _CapturingLogger()
    monkeypatch.setattr(litellm, "_async_success_callback", [capturing_logger])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj = _logging_obj("call-dg-e2e")
    frames = [_results(0.0, 5.0, "hello world", is_final=False), _results(0.0, 5.0, "hello world"), _metadata(20.0)]
    user_api_key_dict = UserAPIKeyAuth(api_key="hashed-key", team_id="team-stt", user_id="user-1")
    start_time = datetime.now()
    passthrough_logging_payload = PassthroughStandardLoggingPayload(
        url=NOVA_3_URL, request_body={}, request_method="WEBSOCKET", cost_per_request=None
    )
    logging_obj.update_environment_variables(
        model="unknown",
        user="unknown",
        optional_params={},
        litellm_params={
            "metadata": {
                "user_api_key": user_api_key_dict.api_key,
                "user_api_key_team_id": user_api_key_dict.team_id,
                "user_api_key_user_id": user_api_key_dict.user_id,
            }
        },
        call_type="pass_through_endpoint",
    )

    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=SimpleNamespace(
            status_code=200,
            text="WebSocket connection successful",
            headers={},
            request=SimpleNamespace(method="WEBSOCKET", url=NOVA_3_URL),
        ),
        response_body=frames,
        logging_obj=logging_obj,
        url_route="/deepgram/v1/listen",
        result="websocket_connection_successful",
        start_time=start_time,
        end_time=datetime.now(),
        cache_hit=False,
        request_body={},
        passthrough_logging_payload=passthrough_logging_payload,
        litellm_params={
            "metadata": {
                "user_api_key": user_api_key_dict.api_key,
                "user_api_key_team_id": user_api_key_dict.team_id,
                "user_api_key_user_id": user_api_key_dict.user_id,
            }
        },
    )

    assert len(capturing_logger.payloads) == 1
    payload = capturing_logger.payloads[0]
    assert payload["model"] == "nova-3"
    assert payload["custom_llm_provider"] == "deepgram"
    assert payload["response_cost"] == pytest.approx(_registry_cost("nova-3", 20.0))
    assert payload["metadata"]["user_api_key_team_id"] == "team-stt"
    assert payload["id"] == "call-dg-e2e"
