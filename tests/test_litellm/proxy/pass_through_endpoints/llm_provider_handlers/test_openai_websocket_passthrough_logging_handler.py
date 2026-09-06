from datetime import datetime
from types import MappingProxyType
from typing import Final
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_websocket_passthrough_logging_handler import (
    OpenAIWebsocketPassthroughLoggingHandler,
)

REALTIME_MODEL: Final = "gpt-realtime-2.1"
RESPONSES_MODEL: Final = "gpt-5.4-mini"


def _logging_obj() -> MagicMock:
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    return logging_obj


def _session_created(model: str) -> dict:
    return {"type": "session.created", "session": {"id": "sess_lit7014", "model": model}}


def _realtime_response_done(text_in: int, audio_in: int, text_out: int, audio_out: int) -> dict:
    return {
        "type": "response.done",
        "response": {
            "id": "resp_lit7014",
            "status": "completed",
            "usage": {
                "input_tokens": text_in + audio_in,
                "output_tokens": text_out + audio_out,
                "total_tokens": text_in + audio_in + text_out + audio_out,
                "input_token_details": {"text_tokens": text_in, "audio_tokens": audio_in, "cached_tokens": 0},
                "output_token_details": {"text_tokens": text_out, "audio_tokens": audio_out},
            },
        },
    }


def _responses_completed(model: str, input_tokens: int, output_tokens: int) -> dict:
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_lit7014",
            "model": model,
            "status": "completed",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    }


def _handle(messages, kwargs=MappingProxyType({}), logging_obj=None, url_route="/openai/v1/realtime"):
    return OpenAIWebsocketPassthroughLoggingHandler().openai_websocket_passthrough_handler(
        websocket_messages=messages,
        logging_obj=logging_obj if logging_obj is not None else _logging_obj(),
        url_route=url_route,
        start_time=datetime.now(),
        kwargs=kwargs,
    )


def _realtime_cost(model: str, text_in: int, audio_in: int, text_out: int, audio_out: int) -> float:
    prices = litellm.model_cost[model]
    return (
        text_in * prices["input_cost_per_token"]
        + audio_in * prices["input_cost_per_audio_token"]
        + text_out * prices["output_cost_per_token"]
        + audio_out * prices["output_cost_per_audio_token"]
    )


def test_realtime_session_is_priced_on_the_model_the_session_reported():
    logging_obj = _logging_obj()
    messages = [
        _session_created(REALTIME_MODEL),
        {"type": "response.created", "response": {"id": "resp_lit7014"}},
        _realtime_response_done(text_in=33, audio_in=120, text_out=16, audio_out=240),
    ]

    handled = _handle(messages, logging_obj=logging_obj)

    expected_cost = _realtime_cost(REALTIME_MODEL, 33, 120, 16, 240)
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handled["kwargs"]["model"] == REALTIME_MODEL
    assert handled["kwargs"]["custom_llm_provider"] == "openai"
    assert handled["result"].model == REALTIME_MODEL
    assert handled["result"].usage.total_tokens == 409
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(expected_cost)
    assert logging_obj.model_call_details["model"] == REALTIME_MODEL


def test_realtime_cost_covers_every_turn_of_the_session():
    one_turn = _handle([_session_created(REALTIME_MODEL), _realtime_response_done(10, 0, 5, 0)])
    two_turns = _handle(
        [
            _session_created(REALTIME_MODEL),
            _realtime_response_done(10, 0, 5, 0),
            _realtime_response_done(10, 0, 5, 0),
        ]
    )

    assert two_turns["kwargs"]["response_cost"] == pytest.approx(2 * one_turn["kwargs"]["response_cost"])
    assert two_turns["result"].usage.total_tokens == 30


def test_responses_turn_is_priced_on_the_model_the_response_reported():
    handled = _handle(
        [_responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5)],
        url_route="/openai_passthrough/v1/responses",
    )

    prices = litellm.model_cost[RESPONSES_MODEL]
    expected_cost = 13 * prices["input_cost_per_token"] + 5 * prices["output_cost_per_token"]
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handled["kwargs"]["model"] == RESPONSES_MODEL
    assert handled["result"].usage.prompt_tokens == 13
    assert handled["result"].usage.completion_tokens == 5


def test_responses_cost_covers_every_turn_of_the_connection():
    handled = _handle(
        [
            _responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5),
            _responses_completed(RESPONSES_MODEL, input_tokens=20, output_tokens=7),
        ],
        url_route="/openai_passthrough/v1/responses",
    )

    prices = litellm.model_cost[RESPONSES_MODEL]
    expected_cost = 33 * prices["input_cost_per_token"] + 12 * prices["output_cost_per_token"]
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handled["result"].usage.total_tokens == 45


def test_the_model_a_client_frame_named_prices_a_session_the_provider_never_named():
    handled = _handle(
        [_realtime_response_done(text_in=10, audio_in=0, text_out=5, audio_out=0)],
        kwargs=MappingProxyType({"model": REALTIME_MODEL}),
    )

    assert handled["kwargs"]["model"] == REALTIME_MODEL
    assert handled["result"].model == REALTIME_MODEL
    assert handled["kwargs"]["response_cost"] == pytest.approx(_realtime_cost(REALTIME_MODEL, 10, 0, 5, 0))


def test_a_session_nobody_named_a_model_for_is_left_unpriced():
    logging_obj = _logging_obj()
    handled = _handle(
        [{"type": "response.done", "response": {"usage": {"input_tokens": 10, "output_tokens": 5}}}],
        kwargs=MappingProxyType({"litellm_call_id": "lit-7014-call"}),
        logging_obj=logging_obj,
    )

    assert handled["result"] is None
    assert handled["kwargs"] == {"litellm_call_id": "lit-7014-call"}
    assert logging_obj.model_call_details == {}


def test_a_session_that_reported_no_usage_still_records_its_model():
    handled = _handle([_session_created(REALTIME_MODEL)])

    assert handled["result"] is None
    assert handled["kwargs"] == {"model": REALTIME_MODEL, "custom_llm_provider": "openai"}


def test_a_responses_turn_without_readable_usage_is_not_billed():
    handled = _handle(
        [{"type": "response.completed", "response": {"model": RESPONSES_MODEL, "status": "completed"}}],
        url_route="/openai_passthrough/v1/responses",
    )

    assert handled["result"] is None
    assert handled["kwargs"] == {"model": RESPONSES_MODEL, "custom_llm_provider": "openai"}
