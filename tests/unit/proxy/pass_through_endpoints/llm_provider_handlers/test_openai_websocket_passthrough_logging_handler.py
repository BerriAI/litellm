import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_websocket_passthrough_logging_handler import (
    OpenAIWebsocketPassthroughLoggingHandler,
    openai_websocket_frame_kind,
)
from litellm.proxy.spend_tracking.spend_tracking_utils import get_spend_logs_id
from litellm.types.utils import CallTypes

REALTIME_MODEL: Final = "gpt-realtime-2.1"
RESPONSES_MODEL: Final = "gpt-5.4-mini"
REASONING_MODEL: Final = "gpt-5.4"
TRANSCRIPTION_MODEL: Final = "gpt-4o-mini-transcribe"
SESSION_START: Final = datetime(2026, 1, 1, 12, 0, 0)


def _logging_obj() -> MagicMock:
    logging_obj: Final = MagicMock(model_call_details={}, litellm_call_id=str(uuid.uuid4()))
    return logging_obj


def _session_created(model: str) -> dict[str, object]:
    return {"type": "session.created", "session": {"id": "sess_fake", "model": model}}


def _realtime_response_done(text_in: int, audio_in: int, text_out: int, audio_out: int) -> dict[str, object]:
    return {
        "type": "response.done",
        "response": {
            "id": "resp_fake",
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


def _responses_completed(
    model: str,
    input_tokens: int,
    output_tokens: int,
    event: str = "response.completed",
    served: Mapping[str, object] = MappingProxyType({}),
) -> dict[str, object]:
    return {
        "type": event,
        "response": {
            "id": "resp_fake",
            "object": "response",
            "created_at": 1761700000,
            "model": model,
            "status": "completed",
            "output": [],
            **served,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    }


def _handle(
    messages: Sequence[Mapping[str, object]],
    kwargs: Mapping[str, object] = MappingProxyType({}),
    logging_obj: MagicMock | None = None,
    url_route: str = "/openai/v1/realtime",
):
    return OpenAIWebsocketPassthroughLoggingHandler().openai_websocket_passthrough_handler(
        websocket_messages=messages,
        logging_obj=logging_obj if logging_obj is not None else _logging_obj(),
        url_route=url_route,
        start_time=SESSION_START,
        kwargs=kwargs,
    )


def _realtime_cost(model: str, text_in: int, audio_in: int, text_out: int, audio_out: int) -> float:
    prices: Final = litellm.model_cost[model]
    return (
        text_in * prices["input_cost_per_token"]
        + audio_in * prices["input_cost_per_audio_token"]
        + text_out * prices["output_cost_per_token"]
        + audio_out * prices["output_cost_per_audio_token"]
    )


def test_realtime_session_is_priced_on_the_model_the_session_reported():
    logging_obj: Final = _logging_obj()
    messages: Final = [
        _session_created(REALTIME_MODEL),
        {"type": "response.created", "response": {"id": "resp_fake"}},
        _realtime_response_done(text_in=33, audio_in=120, text_out=16, audio_out=240),
    ]

    handled: Final = _handle(messages, logging_obj=logging_obj)

    expected_cost: Final = _realtime_cost(REALTIME_MODEL, 33, 120, 16, 240)
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handled["kwargs"]["model"] == REALTIME_MODEL
    assert handled["kwargs"]["custom_llm_provider"] == "openai"
    assert handled["result"].model == REALTIME_MODEL
    assert handled["result"].usage.total_tokens == 409
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(expected_cost)
    assert logging_obj.model_call_details["model"] == REALTIME_MODEL


def test_realtime_cost_covers_every_turn_of_the_session():
    one_turn: Final = _handle([_session_created(REALTIME_MODEL), _realtime_response_done(10, 0, 5, 0)])
    two_turns: Final = _handle(
        [
            _session_created(REALTIME_MODEL),
            _realtime_response_done(10, 0, 5, 0),
            _realtime_response_done(10, 0, 5, 0),
        ]
    )

    assert two_turns["kwargs"]["response_cost"] == pytest.approx(2 * one_turn["kwargs"]["response_cost"])
    assert two_turns["result"].usage.total_tokens == 30


def test_responses_turn_is_priced_on_the_model_the_response_reported():
    handled: Final = _handle(
        [_responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5)],
        url_route="/openai_passthrough/v1/responses",
    )

    prices: Final = litellm.model_cost[RESPONSES_MODEL]
    expected_cost: Final = 13 * prices["input_cost_per_token"] + 5 * prices["output_cost_per_token"]
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handled["kwargs"]["model"] == RESPONSES_MODEL
    assert handled["result"].usage.prompt_tokens == 13
    assert handled["result"].usage.completion_tokens == 5


def test_responses_cost_covers_every_turn_of_the_connection():
    handled: Final = _handle(
        [
            _responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5),
            _responses_completed(RESPONSES_MODEL, input_tokens=20, output_tokens=7),
        ],
        url_route="/openai_passthrough/v1/responses",
    )

    prices: Final = litellm.model_cost[RESPONSES_MODEL]
    expected_cost: Final = 33 * prices["input_cost_per_token"] + 12 * prices["output_cost_per_token"]
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handled["result"].usage.total_tokens == 45


def test_a_model_the_caller_asked_for_never_prices_a_session_the_provider_never_named():
    """
    The caller picks what to ask for, and the provider picks what to run. Only
    the provider's own frames may set the price of a session, so a model the
    caller put in the request cannot stand in for one the provider never named.
    """
    logging_obj: Final = _logging_obj()

    handled: Final = _handle(
        [_realtime_response_done(text_in=10, audio_in=0, text_out=5, audio_out=0)],
        kwargs=MappingProxyType({"model": REALTIME_MODEL}),
        logging_obj=logging_obj,
    )

    assert handled["result"] is None
    assert handled["kwargs"] == {"model": REALTIME_MODEL}
    assert logging_obj.model_call_details == {}


def test_each_responses_turn_is_priced_on_the_model_that_turn_ran():
    """
    One socket can run several models one after another, so a connection that
    starts on a big model and finishes on a small one has to pay the big
    model's price for the turn that used it.
    """
    handled: Final = _handle(
        [
            _responses_completed(REASONING_MODEL, input_tokens=13, output_tokens=5),
            _responses_completed(RESPONSES_MODEL, input_tokens=20, output_tokens=7),
        ],
        url_route="/openai_passthrough/v1/responses",
    )

    first: Final = litellm.model_cost[REASONING_MODEL]
    second: Final = litellm.model_cost[RESPONSES_MODEL]
    expected_cost: Final = (
        13 * first["input_cost_per_token"]
        + 5 * first["output_cost_per_token"]
        + 20 * second["input_cost_per_token"]
        + 7 * second["output_cost_per_token"]
    )
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handled["result"].usage.total_tokens == 45


def test_a_responses_turn_that_names_no_model_is_priced_on_the_session_model():
    handled: Final = _handle(
        [
            _session_created(RESPONSES_MODEL),
            _responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5, served={"model": None}),
        ],
        url_route="/openai_passthrough/v1/responses",
    )

    prices: Final = litellm.model_cost[RESPONSES_MODEL]
    expected_cost: Final = 13 * prices["input_cost_per_token"] + 5 * prices["output_cost_per_token"]
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)


def test_a_session_nobody_named_a_model_for_is_left_unpriced():
    logging_obj: Final = _logging_obj()
    handled: Final = _handle(
        [{"type": "response.done", "response": {"usage": {"input_tokens": 10, "output_tokens": 5}}}],
        kwargs=MappingProxyType({"litellm_call_id": "call-fake"}),
        logging_obj=logging_obj,
    )

    assert handled["result"] is None
    assert handled["kwargs"] == {"litellm_call_id": "call-fake"}
    assert logging_obj.model_call_details == {}


def test_a_session_that_reported_no_usage_still_records_its_model():
    logging_obj: Final = _logging_obj()

    handled: Final = _handle([_session_created(REALTIME_MODEL)], logging_obj=logging_obj)

    assert handled["result"] is None
    assert handled["kwargs"] == {"model": REALTIME_MODEL, "custom_llm_provider": "openai"}
    assert logging_obj.model_call_details["model"] == REALTIME_MODEL
    assert logging_obj.model_call_details["custom_llm_provider"] == "openai"


def test_a_responses_turn_without_readable_usage_is_not_billed():
    handled: Final = _handle(
        [{"type": "response.completed", "response": {"model": RESPONSES_MODEL, "status": "completed"}}],
        url_route="/openai_passthrough/v1/responses",
    )

    assert handled["result"] is None
    assert handled["kwargs"] == {"model": RESPONSES_MODEL, "custom_llm_provider": "openai"}


def test_the_price_of_a_multi_model_connection_reaches_the_spend_row():
    """
    The logging layer prices whatever result it is handed, and one result can
    only name one model, so a connection that ran two models has to carry the
    price it already worked out or the cheaper model pays for both turns. The
    row it lands in names the model the connection opened on.
    """
    handled: Final = _handle(
        [
            _responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5),
            _responses_completed(REASONING_MODEL, input_tokens=20, output_tokens=7),
        ],
        url_route="/openai_passthrough/v1/responses",
    )

    logged_cost: Final = litellm.response_cost_calculator(
        response_object=handled["result"],
        model=handled["kwargs"]["model"],
        custom_llm_provider="openai",
        call_type="completion",
        optional_params={},
    )

    session_prices: Final = litellm.model_cost[RESPONSES_MODEL]
    priced_on_the_session_model: Final = (
        33 * session_prices["input_cost_per_token"] + 12 * session_prices["output_cost_per_token"]
    )
    assert logged_cost == pytest.approx(handled["kwargs"]["response_cost"])
    assert logged_cost > priced_on_the_session_model
    assert handled["kwargs"]["model"] == RESPONSES_MODEL


def test_turns_that_start_in_the_same_instant_each_keep_their_own_spend_row():
    """
    Spend logs key each row by the id its result carries and skip a repeated id
    without a word, so two turns that start in the same instant, on one socket
    or on two, each need an id of their own or one of them is never billed.
    """
    turns: Final = [
        _handle(
            [_responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5)],
            url_route="/openai_passthrough/v1/responses",
        )
        for _ in range(2)
    ]

    spend_log_ids: Final = {
        get_spend_logs_id(CallTypes.pass_through.value, turn["result"].model_dump(), turn["kwargs"]) for turn in turns
    }
    assert len(spend_log_ids) == 2
    assert None not in spend_log_ids


def test_a_turn_the_caller_walked_out_of_still_names_the_model_it_ran():
    """
    A client that closes before response.completed leaves no usage behind to
    bill, but response.created already named the model, so the row says which
    model answered instead of falling back to unknown.
    """
    logging_obj: Final = _logging_obj()

    handled: Final = _handle(
        [{"type": "response.created", "response": {"id": "resp_fake", "model": REASONING_MODEL}}],
        logging_obj=logging_obj,
        url_route="/openai_passthrough/v1/responses",
    )

    assert handled["result"] is None
    assert handled["kwargs"] == {"model": REASONING_MODEL, "custom_llm_provider": "openai"}
    assert logging_obj.model_call_details["model"] == REASONING_MODEL


def _transcription_session(model: str) -> tuple[dict[str, object], ...]:
    return (
        {"type": "session.created", "session": {"type": "transcription", "audio": {"input": {"transcription": None}}}},
        {
            "type": "session.updated",
            "session": {"type": "transcription", "audio": {"input": {"transcription": {"model": model}}}},
        },
    )


def _transcription_completed(usage: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item_fake",
        "transcript": "hello",
        "usage": usage,
    }


@pytest.mark.parametrize(
    ("usage", "billed_units"),
    [
        ({"type": "duration", "seconds": 3}, ((3, "input_cost_per_second"),)),
        (
            {
                "type": "tokens",
                "input_tokens": 30,
                "output_tokens": 11,
                "input_token_details": {"text_tokens": 0, "audio_tokens": 30},
            },
            ((30, "input_cost_per_audio_token"), (11, "output_cost_per_token")),
        ),
    ],
)
def test_a_transcription_session_is_billed_for_every_clip_it_transcribed(usage, billed_units):
    handled: Final = _handle(
        [
            *_transcription_session(TRANSCRIPTION_MODEL),
            _transcription_completed(usage),
            _transcription_completed(usage),
        ],
        url_route="/openai_passthrough/v1/realtime",
    )

    logged_cost: Final = litellm.response_cost_calculator(
        response_object=handled["result"],
        model=handled["kwargs"]["model"],
        custom_llm_provider="openai",
        call_type="completion",
        optional_params={},
    )

    prices: Final = litellm.model_cost[TRANSCRIPTION_MODEL]
    expected_cost: Final = 2 * sum(units * prices[rate] for units, rate in billed_units)
    assert expected_cost > 0
    assert handled["kwargs"]["model"] == TRANSCRIPTION_MODEL
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert logged_cost == pytest.approx(expected_cost)


def test_a_conversation_that_also_transcribes_the_caller_is_logged_on_its_own_model():
    handled: Final = _handle(
        [
            {
                "type": "session.created",
                "session": {
                    "model": REALTIME_MODEL,
                    "audio": {"input": {"transcription": {"model": TRANSCRIPTION_MODEL}}},
                },
            },
            _realtime_response_done(text_in=10, audio_in=0, text_out=5, audio_out=0),
        ]
    )

    assert handled["kwargs"]["model"] == REALTIME_MODEL


@pytest.mark.parametrize("event", ["response.incomplete", "response.failed"])
def test_a_responses_turn_that_ended_early_is_billed_for_the_tokens_it_used(event):
    """
    A turn cut short by max_output_tokens or a provider failure still reports
    the tokens it burned, and OpenAI bills them, so the row has to as well.
    """
    handled: Final = _handle(
        [_responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5, event=event)],
        url_route="/openai_passthrough/v1/responses",
    )

    prices: Final = litellm.model_cost[RESPONSES_MODEL]
    expected_cost: Final = 13 * prices["input_cost_per_token"] + 5 * prices["output_cost_per_token"]
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handled["result"].usage.total_tokens == 18


def test_a_failed_responses_turn_that_reported_no_usage_is_not_billed():
    handled: Final = _handle(
        [{"type": "response.failed", "response": {"model": RESPONSES_MODEL, "status": "failed", "usage": None}}],
        url_route="/openai_passthrough/v1/responses",
    )

    assert handled["result"] is None
    assert handled["kwargs"] == {"model": RESPONSES_MODEL, "custom_llm_provider": "openai"}


@pytest.mark.parametrize(
    ("turn_model", "requested_model", "priced_as"),
    [
        ("gpt-unreleased-qa-snapshot", RESPONSES_MODEL, RESPONSES_MODEL),
        ("gpt-5.4-mini-qa-preview", RESPONSES_MODEL, RESPONSES_MODEL),
        (RESPONSES_MODEL, REASONING_MODEL, RESPONSES_MODEL),
    ],
    ids=["unmapped_model", "model_the_cost_map_prices_at_zero", "priced_turn_model"],
)
def test_a_responses_turn_falls_back_to_the_requested_model_only_when_its_own_has_no_price(
    turn_model: str, requested_model: str, priced_as: str
):
    """
    The provider can answer under a model name the cost map does not price
    yet. That turn still ran the model the key was allowed to ask for, so it is
    billed at that model's price instead of $0, while a turn whose own model
    has a price keeps it.
    """
    handled: Final = _handle(
        [_responses_completed(turn_model, input_tokens=13, output_tokens=5)],
        kwargs=MappingProxyType({"litellm_params": {"metadata": {"model_group": requested_model}}}),
        url_route="/openai_passthrough/v1/responses",
    )

    prices: Final = litellm.model_cost[priced_as]
    expected_cost: Final = 13 * prices["input_cost_per_token"] + 5 * prices["output_cost_per_token"]
    assert expected_cost > 0
    assert handled["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handled["kwargs"]["model"] == turn_model


def _http_passthrough_cost(response: Mapping[str, object]) -> float:
    http_response: Final = httpx.Response(
        200, json=response, request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    _, cost = OpenAIPassthroughLoggingHandler._build_responses_api_response_and_cost(
        RESPONSES_MODEL, http_response, _logging_obj(), "openai"
    )
    return cost


@pytest.mark.parametrize(
    "served",
    [
        {"service_tier": "priority"},
        {"output": [{"type": "web_search_call", "id": "ws_fake", "status": "completed", "action": {"type": "search"}}]},
    ],
    ids=["priority_tier", "hosted_web_search"],
)
def test_a_responses_turn_costs_what_the_http_passthrough_charges_for_the_same_response(served):
    """
    The provider bills a turn by the tier that served it and the hosted tools it
    called, not only its tokens, so a websocket turn has to cost exactly what the
    same response costs over HTTP, never the cheaper token-only price.
    """
    turn: Final = _responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5, served=served)
    plain_turn: Final = _responses_completed(RESPONSES_MODEL, input_tokens=13, output_tokens=5)

    handled: Final = _handle([turn], url_route="/openai_passthrough/v1/responses")

    assert handled["kwargs"]["response_cost"] == pytest.approx(_http_passthrough_cost(turn["response"]))
    assert handled["kwargs"]["response_cost"] > _http_passthrough_cost(plain_turn["response"])


@pytest.mark.parametrize(
    ("frame_type", "kind"),
    [
        ("response.completed", "ends_turn"),
        ("response.incomplete", "ends_turn"),
        ("response.failed", "ends_turn"),
        ("response.done", "ends_turn"),
        ("session.created", "opens_session"),
        ("session.updated", "updates_session"),
        ("conversation.item.input_audio_transcription.completed", "bills_usage"),
        ("response.created", "starts_turn"),
        ("response.output_text.delta", "relayed"),
        ("error", "refuses_turn"),
    ],
)
def test_each_frame_is_classified_by_what_it_does_to_a_turn(frame_type, kind):
    assert openai_websocket_frame_kind({"type": frame_type}) == kind
