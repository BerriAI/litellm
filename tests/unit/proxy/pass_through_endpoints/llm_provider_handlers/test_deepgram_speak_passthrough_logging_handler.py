"""Deepgram ``/v1/speak`` WebSocket passthrough: per-character cost tracking from the relay's client text count."""

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.deepgram_speak_passthrough_logging_handler import (
    DeepgramSpeakPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    WEBSOCKET_CLIENT_TEXT_COUNT_KEY,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import StandardLoggingPayload

AURA_2_URL: Final = "wss://api.deepgram.com/v1/speak?model=aura-2-thalia-en&encoding=linear16&sample_rate=24000"
AURA_1_URL: Final = "wss://api.deepgram.com/v1/speak?model=aura-asteria-en"
SPOKEN_CHARACTERS: Final = len("The quick brown fox jumps over the lazy dog.")
# Pay as you go rates from https://deepgram.com/pricing, checked 2026-10-26
AURA_2_USD_PER_CHARACTER: Final = 0.030 / 1000
AURA_1_USD_PER_CHARACTER: Final = 0.0150 / 1000

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")


@pytest.mark.parametrize(
    ("url_route", "request_method", "expected"),
    [
        ("/deepgram/v1/speak", "WEBSOCKET", True),
        ("/deepgram/speak", "WEBSOCKET", True),
        ("/deepgram/v1/speak?model=aura-2-thalia-en", "WEBSOCKET", True),
        ("/deepgram/v1/listen", "WEBSOCKET", False),
        ("/deepgram/v1/speak/extra", "WEBSOCKET", False),
        ("/openai/v1/audio/speech", "WEBSOCKET", False),
        ("", "WEBSOCKET", False),
        pytest.param("https://gateway.internal/deepgram/v1/speak", "POST", False, id="http passthrough to a speak url"),
    ],
)
def test_is_deepgram_speak_route(url_route: str, request_method: str, expected: bool) -> None:
    assert DeepgramSpeakPassthroughLoggingHandler.is_deepgram_speak_route(url_route, request_method) is expected


def _logging_obj(call_id: str = "call-dg-speak") -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "WebSocket connection"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id=call_id,
        function_id="websocket_passthrough",
    )


def _handler_cost(upstream_url: str, characters: object) -> float | None:
    handler_result: Final = DeepgramSpeakPassthroughLoggingHandler.deepgram_speak_passthrough_handler(
        request_body={WEBSOCKET_CLIENT_TEXT_COUNT_KEY: characters},
        logging_obj=_logging_obj(),
        upstream_url=upstream_url,
    )
    response_cost: Final = handler_result["kwargs"]["response_cost"]
    assert response_cost is None or isinstance(response_cost, float)
    return response_cost


def test_handler_bills_spoken_characters_at_the_aura_2_rate_and_names_the_voice() -> None:
    logging_obj: Final = _logging_obj()

    handler_result: Final = DeepgramSpeakPassthroughLoggingHandler.deepgram_speak_passthrough_handler(
        request_body={WEBSOCKET_CLIENT_TEXT_COUNT_KEY: SPOKEN_CHARACTERS},
        logging_obj=logging_obj,
        upstream_url=AURA_2_URL,
        kwargs={"litellm_params": {"metadata": {}}},
    )

    assert handler_result["kwargs"]["response_cost"] == pytest.approx(SPOKEN_CHARACTERS * AURA_2_USD_PER_CHARACTER)
    assert handler_result["kwargs"]["model"] == "aura-2-thalia-en"
    assert handler_result["kwargs"]["custom_llm_provider"] == "deepgram"
    assert handler_result["kwargs"]["litellm_params"] == {"metadata": {}}
    assert handler_result["result"] == {"response": json.dumps({"characters": SPOKEN_CHARACTERS})}
    assert logging_obj.model == "aura-2-thalia-en"
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(
        SPOKEN_CHARACTERS * AURA_2_USD_PER_CHARACTER
    )


def test_handler_bills_an_aura_1_voice_at_the_aura_1_rate() -> None:
    assert _handler_cost(AURA_1_URL, 1000) == pytest.approx(1000 * AURA_1_USD_PER_CHARACTER)


@pytest.mark.parametrize(
    "characters",
    [pytest.param(0, id="nothing spoken"), pytest.param(None, id="no count")],
)
def test_handler_bills_zero_for_a_session_without_speak_text(characters: object) -> None:
    assert _handler_cost(AURA_2_URL, characters) == 0.0


@pytest.mark.parametrize("model", ["flux-tts-en", "aura-3-x-en", "aura-x", "nova-3"])
def test_handler_reports_no_cost_rather_than_a_wrong_one_for_an_unpriced_voice(model: str) -> None:
    assert _handler_cost(f"wss://api.deepgram.com/v1/speak?model={model}", 1000) is None


class _CapturingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[StandardLoggingPayload] = []  # mutable-ok: the logger appends each payload it receives

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs["standard_logging_object"])


@pytest.mark.parametrize(
    ("characters", "expected_cost"),
    [
        pytest.param(SPOKEN_CHARACTERS, SPOKEN_CHARACTERS * AURA_2_USD_PER_CHARACTER, id="one speak"),
        pytest.param(0, 0.0, id="zero stays zero and is not recomputed"),
    ],
)
async def test_success_handler_dispatches_deepgram_speak_and_logs_character_based_spend(
    monkeypatch: pytest.MonkeyPatch, characters: int, expected_cost: float
) -> None:
    """Drives the shared passthrough success handler the way the relay does at socket close and reads what a spend
    logger receives"""
    capturing_logger: Final = _CapturingLogger()
    monkeypatch.setattr(litellm, "_async_success_callback", [capturing_logger])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj: Final = _logging_obj("call-dg-speak-e2e")
    litellm_params: Final = {"metadata": {"user_api_key": "hashed-key", "user_api_key_team_id": "team-tts"}}
    logging_obj.update_environment_variables(
        model="unknown",
        user="unknown",
        optional_params={},
        litellm_params=litellm_params,
        call_type="pass_through_endpoint",
    )

    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=SimpleNamespace(
            status_code=200,
            text="WebSocket connection successful",
            headers={},
            request=SimpleNamespace(method="WEBSOCKET", url=AURA_2_URL),
        ),
        response_body=[{"type": "Metadata", "request_id": "req-1"}, {"type": "Flushed", "sequence_id": 0}],
        logging_obj=logging_obj,
        url_route="/deepgram/v1/speak",
        result="websocket_connection_successful",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={WEBSOCKET_CLIENT_TEXT_COUNT_KEY: characters},
        passthrough_logging_payload=PassthroughStandardLoggingPayload(
            url=AURA_2_URL, request_body={}, request_method="WEBSOCKET", cost_per_request=None
        ),
        litellm_params=litellm_params,
    )

    assert len(capturing_logger.payloads) == 1
    payload: Final = capturing_logger.payloads[0]
    assert payload["model"] == "aura-2-thalia-en"
    assert payload["custom_llm_provider"] == "deepgram"
    assert payload["response_cost"] == pytest.approx(expected_cost)
    assert payload["response_cost"] is not None
    assert payload["metadata"]["user_api_key_team_id"] == "team-tts"
    assert payload["id"] == "call-dg-speak-e2e"
