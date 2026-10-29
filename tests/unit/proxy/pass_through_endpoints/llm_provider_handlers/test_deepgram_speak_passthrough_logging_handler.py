"""Deepgram ``/v1/speak`` WebSocket passthrough: per-character cost tracking from the relay's client text count."""

import json
from datetime import datetime
from typing import Final

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.deepgram_speak_passthrough_logging_handler import (
    DeepgramSpeakPassthroughLoggingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import WEBSOCKET_CLIENT_TEXT_COUNT_KEY

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


def _logging_obj(characters: object) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "WebSocket connection"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="call-dg-speak",
        function_id="websocket_passthrough",
    )
    logging_obj.model_call_details[WEBSOCKET_CLIENT_TEXT_COUNT_KEY] = characters
    return logging_obj


def _billed_cost(upstream_url: str, characters: object) -> object:
    logging_obj: Final = _logging_obj(characters)
    DeepgramSpeakPassthroughLoggingHandler.deepgram_speak_passthrough_handler(logging_obj, upstream_url)
    return logging_obj.model_call_details["response_cost"]


def test_handler_bills_spoken_characters_at_the_aura_2_rate_and_logs_only_the_count() -> None:
    logging_obj: Final = _logging_obj(SPOKEN_CHARACTERS)

    result: Final = DeepgramSpeakPassthroughLoggingHandler.deepgram_speak_passthrough_handler(logging_obj, AURA_2_URL)

    assert logging_obj.model_call_details["response_cost"] == pytest.approx(
        SPOKEN_CHARACTERS * AURA_2_USD_PER_CHARACTER
    )
    assert logging_obj.model == logging_obj.model_call_details["model"] == "aura-2-thalia-en"
    assert logging_obj.model_call_details["custom_llm_provider"] == "deepgram"
    assert result == {"response": json.dumps({"characters": SPOKEN_CHARACTERS})}


def test_handler_bills_an_aura_1_voice_at_the_aura_1_rate() -> None:
    assert _billed_cost(AURA_1_URL, 1000) == pytest.approx(1000 * AURA_1_USD_PER_CHARACTER)


@pytest.mark.parametrize(
    "characters",
    [pytest.param(0, id="nothing spoken"), pytest.param(None, id="no count")],
)
def test_handler_bills_zero_for_a_session_without_speak_text(characters: object) -> None:
    assert _billed_cost(AURA_2_URL, characters) == 0.0


@pytest.mark.parametrize("model", ["flux-tts-en", "aura-3-x-en", "aura-x", "nova-3"])
def test_handler_reports_no_cost_rather_than_a_wrong_one_for_an_unpriced_voice(model: str) -> None:
    assert _billed_cost(f"wss://api.deepgram.com/v1/speak?model={model}", 1000) is None
