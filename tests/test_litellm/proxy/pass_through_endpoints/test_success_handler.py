from datetime import datetime
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging

REALTIME_FRAMES: Final = [
    {"type": "session.created", "session": {"id": "sess_lit7014", "model": "gpt-realtime-2.1"}},
    {
        "type": "response.done",
        "response": {
            "usage": {
                "input_tokens": 33,
                "output_tokens": 16,
                "total_tokens": 49,
                "input_token_details": {"text_tokens": 33, "audio_tokens": 0, "cached_tokens": 0},
                "output_token_details": {"text_tokens": 16, "audio_tokens": 0},
            }
        },
    },
]


def _normalize(url_route: str, response_body):
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    return PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
        httpx_response=MagicMock(spec=httpx.Response),
        response_body=response_body,
        request_body={},
        logging_obj=logging_obj,
        url_route=url_route,
        result="websocket_connection_successful",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        litellm_call_id="lit-7014-call",
    )


@pytest.mark.parametrize("url_route", ["/openai/v1/realtime", "/openai_passthrough/v1/realtime"])
def test_openai_websocket_frames_reach_the_websocket_cost_handler(url_route):
    normalized = _normalize(url_route, REALTIME_FRAMES)

    assert normalized["kwargs"]["model"] == "gpt-realtime-2.1"
    assert normalized["kwargs"]["custom_llm_provider"] == "openai"
    assert normalized["kwargs"]["response_cost"] == pytest.approx(33 * 4e-06 + 16 * 2.4e-05)
    assert normalized["standard_logging_response_object"].usage.total_tokens == 49


def test_openai_http_passthrough_bodies_are_left_to_the_http_handlers():
    normalized = _normalize("/openai/v1/responses", {"model": "gpt-5.4-mini", "usage": {"input_tokens": 13}})

    assert normalized["standard_logging_response_object"] is None
    assert normalized["kwargs"] == {"litellm_call_id": "lit-7014-call"}
