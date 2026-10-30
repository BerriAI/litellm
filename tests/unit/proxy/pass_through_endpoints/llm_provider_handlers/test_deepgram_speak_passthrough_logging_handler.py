"""Deepgram ``/v1/speak`` WebSocket passthrough: per-character cost tracking."""

from datetime import datetime
from types import SimpleNamespace
from typing import Final

import pytest

import litellm
from litellm.constants import WEBSOCKET_CLIENT_TEXT_COUNT_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.deepgram_speak_passthrough_logging_handler import (
    DeepgramSpeakPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.types.passthrough_endpoints.pass_through_endpoints import PassthroughStandardLoggingPayload
from litellm.types.utils import StandardLoggingPayload

AURA_2_URL: Final = "wss://api.deepgram.com/v1/speak?encoding=linear16&model=aura-2-thalia-en"

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")


@pytest.mark.parametrize(
    ("url_route", "expected"),
    [
        ("/deepgram/v1/speak", True),
        ("/deepgram/speak", True),
        ("/deepgram/v1/speak?model=aura-2-thalia-en", True),
        ("/deepgram/v1/listen", False),
        ("/openai/v1/audio/speak", False),
        ("", False),
    ],
)
def test_is_deepgram_speak_route(url_route: str, expected: bool):
    assert DeepgramSpeakPassthroughLoggingHandler.is_deepgram_speak_route(url_route) is expected


class _CapturingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[StandardLoggingPayload] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs["standard_logging_object"])


async def _log_speak_session(monkeypatch: pytest.MonkeyPatch, characters: int | None) -> StandardLoggingPayload:
    """Drives the shared passthrough success handler the way the WebSocket relay does at socket close and returns
    what a spend logger receives."""
    capturing_logger = _CapturingLogger()
    monkeypatch.setattr(litellm, "_async_success_callback", [capturing_logger])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj = LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "WebSocket connection"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id="call-dg-speak",
        function_id="websocket_passthrough",
    )
    litellm_params: Final = {"metadata": {"user_api_key": "hashed-key", "user_api_key_team_id": "team-tts"}}
    logging_obj.update_environment_variables(
        model="unknown",
        user="unknown",
        optional_params={},
        litellm_params=litellm_params,
        call_type="pass_through_endpoint",
    )
    if characters is not None:
        logging_obj.model_call_details[WEBSOCKET_CLIENT_TEXT_COUNT_KEY] = characters

    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=SimpleNamespace(
            status_code=200,
            text="WebSocket connection successful",
            headers={},
            request=SimpleNamespace(method="WEBSOCKET", url=AURA_2_URL),
        ),
        response_body=[{"type": "Metadata", "model_name": "aura-2-thalia-en"}, {"type": "Flushed", "sequence_id": 0}],
        logging_obj=logging_obj,
        url_route="/deepgram/v1/speak",
        result="websocket_connection_successful",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={},
        passthrough_logging_payload=PassthroughStandardLoggingPayload(
            url=AURA_2_URL, request_body={}, request_method="WEBSOCKET", cost_per_request=None
        ),
        litellm_params=litellm_params,
    )

    assert len(capturing_logger.payloads) == 1
    return capturing_logger.payloads[0]


@pytest.mark.asyncio
async def test_success_handler_bills_the_counted_characters_at_the_voice_generation_rate(monkeypatch):
    payload = await _log_speak_session(monkeypatch, characters=47)

    assert payload["model"] == "aura-2-thalia-en"
    assert payload["custom_llm_provider"] == "deepgram"
    assert payload["response_cost"] == pytest.approx(
        47 * litellm.model_cost["deepgram/aura-2"]["input_cost_per_character"]
    )
    assert payload["response"] == {"response": '{"characters": 47}'}
    assert payload["metadata"]["user_api_key_team_id"] == "team-tts"


@pytest.mark.asyncio
async def test_success_handler_bills_nothing_for_a_session_that_spoke_no_text(monkeypatch):
    payload = await _log_speak_session(monkeypatch, characters=None)

    assert payload["model"] == "aura-2-thalia-en"
    assert payload["response_cost"] == 0
