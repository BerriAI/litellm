import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from litellm.integrations.helicone_v2 import HeliconeLogger


@pytest.fixture
def helicone_logger():
    logger = HeliconeLogger()
    yield logger
    async_client = getattr(logger, "async_httpx_client", None)
    if async_client is not None:
        close_coro = None
        aclose = getattr(async_client, "aclose", None)
        if callable(aclose):
            close_result = aclose()
            if asyncio.iscoroutine(close_result):
                close_coro = close_result
        if close_coro is not None:
            try:
                asyncio.run(close_coro)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(close_coro)
                loop.close()
        else:
            close = getattr(async_client, "close", None)
            if callable(close):
                close()


def _logging_payload(status: str = "success", **overrides):
    payload = {
        "status": status,
        "response": {"completion": "ok"},
        "error_str": "failure",
        "error_information": None,
        "api_base": "https://provider.example",
        "hidden_params": {},
    }
    payload.update(overrides)
    return payload


def _timing():
    start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(seconds=1, milliseconds=500)
    return start, end


def test_pick_request_json_extracts_complete_input_dict(helicone_logger):
    kwargs = {"additional_args": {"complete_input_dict": {"foo": "bar"}}}

    request_json = helicone_logger._pick_request_json(kwargs)

    assert request_json == {"foo": "bar"}


def test_pick_request_json_returns_empty_dict_when_missing(helicone_logger):
    request_json = helicone_logger._pick_request_json({})

    assert request_json == {}


def test_pick_response_returns_success_payload(helicone_logger):
    payload = _logging_payload()

    response = helicone_logger._pick_response(payload)

    assert response == {"completion": "ok"}


def test_pick_response_returns_error_string_when_failure(helicone_logger):
    payload = _logging_payload(status="failure")

    response = helicone_logger._pick_response(payload)

    assert response == "failure"


def test_pick_response_headers_extracts_headers(helicone_logger):
    payload = _logging_payload(hidden_params={"response_headers": {"x-header": "value"}})

    headers = helicone_logger._pick_response_headers(payload)

    assert headers == {"x-header": "value"}


def test_pick_status_code_returns_int_for_numeric_string(helicone_logger):
    payload = _logging_payload(
        error_information={"error_code": "400"},
    )

    status_code = helicone_logger._pick_status_code(payload)

    assert status_code == 400


def test_pick_status_code_defaults_when_missing_error_code(helicone_logger):
    payload = _logging_payload()

    status_code = helicone_logger._pick_status_code(payload)

    assert status_code == 200


def test_pick_status_code_defaults_for_blank_string(helicone_logger):
    payload = _logging_payload(error_information={"error_code": ""})

    status_code = helicone_logger._pick_status_code(payload)

    assert status_code == 200


def test_build_data_compiles_payload_structure(helicone_logger):
    start_time, end_time = _timing()
    payload = _logging_payload(
        status="failure",
        error_information={"error_code": "400"},
    )
    kwargs = {
        "standard_logging_object": payload,
        "additional_args": {"complete_input_dict": {"messages": ["hello"]}},
    }

    built = helicone_logger._build_data(kwargs, {}, start_time, end_time)

    assert built["providerRequest"]["url"] == "https://provider.example"
    assert built["providerRequest"]["json"] == {"messages": ["hello"]}
    assert built["providerResponse"]["json"] == "failure"
    assert built["providerResponse"]["status"] == 400
    assert built["timing"]["startTime"]["seconds"] == int(start_time.timestamp())


def test_log_success_event_forwards_payload(monkeypatch, helicone_logger):
    captured = {}

    def fake_send(data):
        captured["payload"] = data

    monkeypatch.setattr(helicone_logger, "_send_sync", fake_send)
    start_time, end_time = _timing()
    kwargs = {
        "standard_logging_object": _logging_payload(),
        "additional_args": {"complete_input_dict": {"messages": []}},
    }

    helicone_logger.log_success_event(kwargs, {}, start_time, end_time)

    payload = captured["payload"]
    assert payload["providerResponse"]["json"] == {"completion": "ok"}
    assert payload["providerResponse"]["status"] == 200


def test_async_log_failure_event_sends_payload(monkeypatch, helicone_logger):
    captured = {}

    async def fake_send(data):
        captured["payload"] = data

    monkeypatch.setattr(helicone_logger, "_send_async", fake_send)
    helicone_logger.flush_lock = None
    start_time, end_time = _timing()
    kwargs = {
        "standard_logging_object": _logging_payload(
            status="failure",
            error_information={"error_code": "502"},
        ),
        "additional_args": {"complete_input_dict": {"messages": ["retry"]}},
    }

    asyncio.run(
        helicone_logger.async_log_failure_event(kwargs, {}, start_time, end_time)
    )

    payload = captured["payload"]
    assert payload["providerResponse"]["json"] == "failure"
    assert payload["providerResponse"]["status"] == 502
