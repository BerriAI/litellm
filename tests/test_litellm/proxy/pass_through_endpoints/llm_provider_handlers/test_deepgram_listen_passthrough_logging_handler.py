"""Deepgram ``/v1/listen`` WebSocket passthrough: duration extraction and duration based cost tracking."""

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
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import PassthroughStandardLoggingPayload
from litellm.types.utils import StandardLoggingPayload, TranscriptionResponse

NOVA_3_URL: Final = "wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16&sample_rate=16000"

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")


def _results(start: object, duration: object, transcript: str = "", is_final: object = True) -> dict[str, object]:
    return {
        "type": "Results",
        "start": start,
        "duration": duration,
        "is_final": is_final,
        "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.9}]},
    }


def _metadata(duration: object, channels: int = 1) -> dict[str, object]:
    return {"type": "Metadata", "request_id": "req-1", "duration": duration, "channels": channels}


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


def _registry_cost(pricing_model: str, seconds: float) -> float:
    """Derives the expected charge from the live cost map rather than pinning a vendor price."""
    per_second: Final = litellm.model_cost[f"deepgram/{pricing_model}"]["input_cost_per_second"]
    assert per_second > 0
    return per_second * seconds


def _cost(upstream_url: str, *frames: dict[str, object]) -> float:
    handler_result = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=frames, logging_obj=_logging_obj(), upstream_url=upstream_url
    )
    response_cost = handler_result["kwargs"]["response_cost"]
    assert isinstance(response_cost, float)
    return response_cost


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
    assert result._hidden_params["response_cost"] == pytest.approx(_registry_cost("streaming/nova-3", 12.5))
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(_registry_cost("streaming/nova-3", 12.5))
    assert handler_result["kwargs"]["model"] == "nova-3"
    assert handler_result["kwargs"]["custom_llm_provider"] == "deepgram"
    assert handler_result["kwargs"]["litellm_params"] == {"metadata": {}}
    assert logging_obj.model == "nova-3"
    assert logging_obj.model_call_details["model"] == "nova-3"
    assert logging_obj.model_call_details["custom_llm_provider"] == "deepgram"
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(_registry_cost("streaming/nova-3", 12.5))


def test_handler_bills_streaming_not_prerecorded_rates():
    """Deepgram prices /v1/listen over a WebSocket separately from pre-recorded transcription, so the streaming entry
    must be the one charged; the two registry rows only need to differ for this to matter, whatever their values."""
    streaming = litellm.model_cost["deepgram/streaming/nova-3"]["input_cost_per_second"]
    prerecorded = litellm.model_cost["deepgram/nova-3"]["input_cost_per_second"]
    assert streaming != prerecorded

    assert _cost(NOVA_3_URL, _metadata(60.0)) == pytest.approx(60.0 * streaming)


def test_handler_bills_multilingual_streaming_when_language_is_multi():
    monolingual = _cost(NOVA_3_URL, _metadata(60.0))
    multilingual = _cost(f"{NOVA_3_URL}&language=multi", _metadata(60.0))

    assert multilingual == pytest.approx(_registry_cost("streaming/nova-3-multilingual", 60.0))
    assert multilingual > monolingual


@pytest.mark.parametrize(
    ("query", "addons"),
    [
        pytest.param("redact=pci", ("redact",), id="redaction"),
        pytest.param("redact=pci&redact=numbers", ("redact",), id="redaction counted once"),
        pytest.param("keyterm=LiteLLM&keyterm=Deepgram", ("keyterm",), id="keyterm prompting"),
        pytest.param("detect_entities=true", ("detect_entities",), id="entity detection"),
        pytest.param("diarize=true", ("diarize",), id="diarization"),
        pytest.param("diarize_model=v1", ("diarize",), id="diarization via diarize_model"),
        pytest.param("diarize=true&diarize_model=v1", ("diarize",), id="diarization counted once"),
        pytest.param(
            "redact=pci&keyterm=x&detect_entities=true&diarize=true",
            ("redact", "keyterm", "detect_entities", "diarize"),
            id="every add-on",
        ),
        pytest.param("detect_entities=false&diarize=False&redact=", (), id="disabled add-ons cost nothing"),
    ],
)
def test_handler_adds_each_priced_add_on_once_on_top_of_the_base_rate(query: str, addons: tuple[str, ...]):
    base = _cost(NOVA_3_URL, _metadata(60.0))
    expected = base + sum(_registry_cost(f"streaming/{addon}", 60.0) for addon in addons)

    assert _cost(f"{NOVA_3_URL}&{query}", _metadata(60.0)) == pytest.approx(expected)


def test_handler_add_ons_scale_with_channels_like_the_base_rate():
    stereo_plain = _cost(f"{NOVA_3_URL}&channels=2", _metadata(60.0, channels=2))
    stereo_redacted = _cost(f"{NOVA_3_URL}&channels=2&redact=pci", _metadata(60.0, channels=2))

    assert stereo_redacted - stereo_plain == pytest.approx(_registry_cost("streaming/redact", 120.0))


def test_handler_falls_back_to_the_prerecorded_rate_for_a_model_without_a_streaming_entry():
    assert "deepgram/streaming/nova-2" not in litellm.model_cost

    handler_result = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=(_metadata(60.0),),
        logging_obj=_logging_obj(),
        upstream_url="wss://api.deepgram.com/v1/listen?model=nova-2",
    )

    assert handler_result["kwargs"]["model"] == "nova-2"
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(_registry_cost("nova-2", 60.0))


def test_handler_falls_back_to_results_frames_when_the_stream_ends_without_metadata():
    frames = (_results(0.0, 30.0, "a"), _results(30.0, 30.0, "b"), _results(60.0, 12.5, "c"))

    handler_result = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=frames, logging_obj=_logging_obj(), upstream_url=NOVA_3_URL
    )

    assert handler_result["result"]._hidden_params["audio_transcription_duration"] == 72.5
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(_registry_cost("streaming/nova-3", 72.5))


def test_handler_charges_more_for_more_audio_on_the_same_model():
    short = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=(_metadata(10.0),), logging_obj=_logging_obj(), upstream_url=NOVA_3_URL
    )
    long = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=(_metadata(30.0),), logging_obj=_logging_obj(), upstream_url=NOVA_3_URL
    )

    assert long["kwargs"]["response_cost"] == pytest.approx(3 * short["kwargs"]["response_cost"])
    assert short["kwargs"]["response_cost"] > 0


def test_handler_bills_every_channel_of_a_multichannel_session():
    """Deepgram bills processed audio per channel (deepgram.com/pricing FAQ, 2026-09-17), so a stereo session must be
    charged for twice its wall-clock duration or budgets can be bypassed by requesting more channels."""
    mono = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=(_metadata(30.0),), logging_obj=_logging_obj(), upstream_url=NOVA_3_URL
    )
    stereo = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=(_metadata(30.0, channels=2),),
        logging_obj=_logging_obj(),
        upstream_url=f"{NOVA_3_URL}&multichannel=true&channels=2",
    )

    assert stereo["result"]._hidden_params["audio_transcription_duration"] == 60.0
    assert stereo["kwargs"]["response_cost"] == pytest.approx(2 * mono["kwargs"]["response_cost"])
    assert stereo["kwargs"]["response_cost"] == pytest.approx(_registry_cost("streaming/nova-3", 60.0))


def test_handler_bills_the_declared_channels_when_the_stream_dies_before_any_frame_reports_them():
    handler_result = DeepgramListenPassthroughLoggingHandler().deepgram_listen_passthrough_handler(
        websocket_messages=(_results(0.0, 10.0, "a"),),
        logging_obj=_logging_obj(),
        upstream_url=f"{NOVA_3_URL}&multichannel=true&channels=3",
    )

    assert handler_result["result"]._hidden_params["audio_transcription_duration"] == 30.0
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(_registry_cost("streaming/nova-3", 30.0))


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
    assert payload["response_cost"] == pytest.approx(_registry_cost("streaming/nova-3", 20.0))
    assert payload["metadata"]["user_api_key_team_id"] == "team-stt"
    assert payload["id"] == "call-dg-e2e"
