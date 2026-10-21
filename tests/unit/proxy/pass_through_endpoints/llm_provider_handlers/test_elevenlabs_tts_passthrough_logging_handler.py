from datetime import datetime
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.elevenlabs.common_utils import elevenlabs_tts_recorded_upstream_frame
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.elevenlabs_tts_passthrough_logging_handler import (
    ElevenLabsTextToSpeechPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import PassthroughStandardLoggingPayload
from litellm.types.utils import StandardLoggingPayload

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")

FLASH_URL: Final = "wss://api.elevenlabs.io/v1/text-to-speech/voice1/stream-input?model_id=eleven_flash_v2_5"
SESSION_TEXT: Final = (" ", "Hello world. ", "¿Qué tal? ", "")
SPOKEN_TEXT: Final = ("Hello world.", "¿Qué tal?")


def _client_frames(*texts: str) -> dict[str, object]:
    return {"client_frames": [{"text": text} for text in texts]}


def _spoken(*texts: str) -> tuple[dict[str, object], ...]:
    return tuple(
        dict(
            elevenlabs_tts_recorded_upstream_frame(
                {"audio": "AAAA", "alignment": {"chars": list(text)}, "isFinal": None}
            )
        )
        for text in texts
    )


def _registry_cost(model_id: str, characters: int) -> float:
    per_character: Final = litellm.model_cost[f"elevenlabs/{model_id}"]["input_cost_per_character"]
    assert per_character > 0
    return per_character * characters


def _logging_obj(call_id: str = "call-el") -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "WebSocket connection"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id=call_id,
        function_id="websocket_passthrough",
    )


@pytest.mark.parametrize(
    ("url_route", "expected"),
    [
        ("/elevenlabs/v1/text-to-speech/voice1/stream-input", True),
        ("/elevenlabs/v1/text-to-speech/voice1/multi-stream-input", True),
        ("/elevenlabs/v1/text-to-speech/voice1/stream-input?model_id=eleven_flash_v2_5", True),
        ("/litellm/elevenlabs/v1/text-to-speech/voice1/stream-input", True),
        ("/elevenlabs/v1/text-to-speech/voice1/stream", False),
        ("/elevenlabs/v1/text-to-speech/voice1", False),
        ("/elevenlabs/v1/speech-to-text", False),
        ("/deepgram/v1/listen", False),
        ("/openai/v1/realtime/stream-input", False),
        ("", False),
    ],
)
def test_is_elevenlabs_tts_websocket_route(url_route, expected):
    assert ElevenLabsTextToSpeechPassthroughLoggingHandler.is_elevenlabs_tts_websocket_route(url_route) is expected


def test_handler_bills_the_characters_elevenlabs_spoke_at_the_registry_rate_and_names_the_model():
    logging_obj = _logging_obj()
    characters = sum(len(text) for text in SPOKEN_TEXT)
    upstream_frames = (*_spoken(*SPOKEN_TEXT), {"isFinal": True})

    handler_result = ElevenLabsTextToSpeechPassthroughLoggingHandler().elevenlabs_tts_passthrough_handler(
        request_body=_client_frames(*SESSION_TEXT),
        upstream_frames=upstream_frames,
        logging_obj=logging_obj,
        upstream_url=FLASH_URL,
        kwargs={"litellm_params": {"metadata": {}}},
    )

    expected_cost = _registry_cost("eleven_flash_v2_5", characters)
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handler_result["kwargs"]["model"] == "eleven_flash_v2_5"
    assert handler_result["kwargs"]["custom_llm_provider"] == "elevenlabs"
    assert handler_result["kwargs"]["litellm_params"] == {"metadata": {}}
    assert logging_obj.model == "eleven_flash_v2_5"
    assert logging_obj.model_call_details["model"] == "eleven_flash_v2_5"
    assert logging_obj.model_call_details["custom_llm_provider"] == "elevenlabs"
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(expected_cost)
    assert handler_result["result"] == {
        "response": {
            "model": "eleven_flash_v2_5",
            "characters": characters,
            "client_text_frames": len(SESSION_TEXT),
            "upstream_frames": len(upstream_frames),
        }
    }


def test_handler_bills_the_model_the_session_ran_on():
    flash = ElevenLabsTextToSpeechPassthroughLoggingHandler().elevenlabs_tts_passthrough_handler(
        request_body=_client_frames("x" * 1000),
        upstream_frames=_spoken("x" * 1000),
        logging_obj=_logging_obj(),
        upstream_url=FLASH_URL,
    )
    multilingual = ElevenLabsTextToSpeechPassthroughLoggingHandler().elevenlabs_tts_passthrough_handler(
        request_body=_client_frames("x" * 1000),
        upstream_frames=_spoken("x" * 1000),
        logging_obj=_logging_obj(),
        upstream_url=FLASH_URL.replace("eleven_flash_v2_5", "eleven_multilingual_v2"),
    )

    assert flash["kwargs"]["response_cost"] == pytest.approx(_registry_cost("eleven_flash_v2_5", 1000))
    assert multilingual["kwargs"]["response_cost"] == pytest.approx(_registry_cost("eleven_multilingual_v2", 1000))
    assert multilingual["kwargs"]["model"] == "eleven_multilingual_v2"


def test_handler_charges_in_proportion_to_the_text_spoken():
    short = ElevenLabsTextToSpeechPassthroughLoggingHandler().elevenlabs_tts_passthrough_handler(
        request_body=_client_frames("abcde"),
        upstream_frames=_spoken("abcde"),
        logging_obj=_logging_obj(),
        upstream_url=FLASH_URL,
    )
    long = ElevenLabsTextToSpeechPassthroughLoggingHandler().elevenlabs_tts_passthrough_handler(
        request_body=_client_frames("abcde", "fghij", "klmno"),
        upstream_frames=_spoken("abcde", "fghij", "klmno"),
        logging_obj=_logging_obj(),
        upstream_url=FLASH_URL,
    )

    assert short["kwargs"]["response_cost"] > 0
    assert long["kwargs"]["response_cost"] == pytest.approx(3 * short["kwargs"]["response_cost"])


def test_handler_records_no_cost_rather_than_a_stand_in_rate_for_an_unpriced_model():
    unpriced_url = FLASH_URL.replace("eleven_flash_v2_5", "eleven_not_in_registry")

    handler_result = ElevenLabsTextToSpeechPassthroughLoggingHandler().elevenlabs_tts_passthrough_handler(
        request_body=_client_frames("Hello "),
        upstream_frames=_spoken("Hello"),
        logging_obj=_logging_obj(),
        upstream_url=unpriced_url,
    )

    assert handler_result["kwargs"]["response_cost"] is None
    assert handler_result["kwargs"]["model"] == "eleven_not_in_registry"
    assert handler_result["result"]["response"]["characters"] == len("Hello")


def test_handler_with_no_audio_back_bills_nothing():
    handler_result = ElevenLabsTextToSpeechPassthroughLoggingHandler().elevenlabs_tts_passthrough_handler(
        request_body=_client_frames(*SESSION_TEXT),
        upstream_frames=({"isFinal": True},),
        logging_obj=_logging_obj(),
        upstream_url=FLASH_URL,
    )

    assert handler_result["kwargs"]["response_cost"] == 0
    assert handler_result["result"]["response"]["characters"] == 0


class _CapturingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[StandardLoggingPayload] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs["standard_logging_object"])


@pytest.mark.asyncio
async def test_success_handler_dispatches_elevenlabs_tts_and_logs_character_spend_without_audio(monkeypatch):
    capturing_logger = _CapturingLogger()
    monkeypatch.setattr(litellm, "_async_success_callback", [capturing_logger])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj = _logging_obj("call-el-e2e")
    request_body = _client_frames(*SESSION_TEXT)
    upstream_frames = [*_spoken(*SPOKEN_TEXT), {"isFinal": True}]
    user_api_key_dict = UserAPIKeyAuth(api_key="hashed-key", team_id="team-tts", user_id="user-1")
    metadata = {
        "user_api_key": user_api_key_dict.api_key,
        "user_api_key_team_id": user_api_key_dict.team_id,
        "user_api_key_user_id": user_api_key_dict.user_id,
    }
    logging_obj.update_environment_variables(
        model="unknown",
        user="unknown",
        optional_params={},
        litellm_params={"metadata": metadata},
        call_type="pass_through_endpoint",
    )

    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=SimpleNamespace(
            status_code=200,
            text="WebSocket connection successful",
            headers={},
            request=SimpleNamespace(method="WEBSOCKET", url=FLASH_URL),
        ),
        response_body=upstream_frames,
        logging_obj=logging_obj,
        url_route="/elevenlabs/v1/text-to-speech/voice1/stream-input",
        result="websocket_connection_successful",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body=request_body,
        passthrough_logging_payload=PassthroughStandardLoggingPayload(
            url=FLASH_URL, request_body=request_body, request_method="WEBSOCKET", cost_per_request=None
        ),
        litellm_params={"metadata": metadata},
    )

    assert len(capturing_logger.payloads) == 1
    payload = capturing_logger.payloads[0]
    assert payload["model"] == "eleven_flash_v2_5"
    assert payload["custom_llm_provider"] == "elevenlabs"
    assert payload["response_cost"] == pytest.approx(
        _registry_cost("eleven_flash_v2_5", sum(len(text) for text in SPOKEN_TEXT))
    )
    assert payload["metadata"]["user_api_key_team_id"] == "team-tts"
    assert payload["id"] == "call-el-e2e"
    assert payload["response"] == {
        "response": {
            "model": "eleven_flash_v2_5",
            "characters": sum(len(text) for text in SPOKEN_TEXT),
            "client_text_frames": len(SESSION_TEXT),
            "upstream_frames": len(upstream_frames),
        }
    }
