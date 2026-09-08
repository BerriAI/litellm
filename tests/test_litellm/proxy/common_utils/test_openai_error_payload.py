import json

import pytest
from fastapi import HTTPException

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.common_utils.openai_error_payload import (
    error_status_code,
    openai_error_param,
    openai_error_type,
)


@pytest.mark.parametrize(
    "status_code, expected_type",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "invalid_request_error"),
        (408, "timeout_error"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (499, "invalid_request_error"),
        (500, "internal_server_error"),
        (502, "internal_server_error"),
        (503, "internal_server_error"),
    ],
)
def test_status_code_decides_the_type_when_the_exception_carries_none(status_code: int, expected_type: str):
    """A route that raises a bare HTTPException carries no error type, so the status it
    answered with is the only thing left to name the OpenAI type from."""
    assert openai_error_type(HTTPException(status_code=status_code, detail="boom"), status_code) == expected_type


def test_a_carried_type_wins_over_the_one_the_status_would_imply():
    """A ProxyException raised mid-request already names its own type, and relabelling a
    402 budget_exceeded as the status map's guess would lose what the client branches on."""
    carried = ProxyException(
        message="Budget has been exceeded",
        type=ProxyErrorTypes.budget_exceeded.value,
        param=None,
        code=400,
    )

    assert openai_error_type(carried, 400) == ProxyErrorTypes.budget_exceeded.value


@pytest.mark.parametrize("carried_type", [None, 400, {"type": "invalid_request_error"}, ["invalid_request_error"]])
def test_a_non_string_carried_type_falls_back_to_the_status(carried_type: object):
    """OpenAI types error.type as a string, so anything else on the exception is not one and
    must not reach the wire the way the literal "None" used to."""

    class _Carrier(Exception):
        type = carried_type

    assert openai_error_type(_Carrier("boom"), 401) == "authentication_error"


def test_the_type_is_never_the_string_none_after_a_json_round_trip():
    """The bug this module exists for: json.dumps of a "None" default is indistinguishable
    from a real type to a client's error handler."""
    payload = json.loads(
        json.dumps(
            {
                "type": openai_error_type(HTTPException(status_code=400, detail="boom"), 400),
                "param": openai_error_param(HTTPException(status_code=400, detail="boom")),
            }
        )
    )

    assert payload == {"type": "invalid_request_error", "param": None}


def test_a_carried_param_names_the_offending_field():
    carried = ProxyException(message="Invalid purpose", type="invalid_request_error", param="purpose", code=400)

    assert openai_error_param(carried) == "purpose"


@pytest.mark.parametrize("exc", [HTTPException(status_code=400, detail="boom"), ValueError("boom"), None])
def test_param_is_json_null_when_the_exception_names_no_field(exc: Exception | None):
    assert openai_error_param(exc) is None


def test_a_non_string_carried_param_is_json_null():
    class _Carrier(Exception):
        param = 42

    assert openai_error_param(_Carrier("boom")) is None


def test_a_carried_status_code_wins_over_the_default():
    assert error_status_code(HTTPException(status_code=429, detail="slow down"), 400) == 429


@pytest.mark.parametrize("default", [400, 500])
def test_the_default_status_stands_when_the_exception_carries_none(default: int):
    assert error_status_code(ValueError("boom"), default) == default


@pytest.mark.parametrize("carried_status", [True, False, "429", None, 429.0])
def test_a_non_int_carried_status_falls_back_to_the_default(carried_status: object):
    """True is an int in Python but not an HTTP status, and a stringified one would break
    every caller that compares the code numerically."""

    class _Carrier(Exception):
        status_code = carried_status

    assert error_status_code(_Carrier("boom"), 500) == 500


def test_a_proxy_exception_keeps_the_status_it_was_raised_with():
    """ProxyException stores its status as the string ``code`` rather than ``status_code``,
    so a route tail that rewraps one used to answer a 4xx rejection as a 500."""
    rejection = ProxyException(message="session_id is required", type="bad_request_error", param="session_id", code=400)

    assert error_status_code(rejection, 500) == 400


@pytest.mark.parametrize("carried_code", [None, "None", "", "rate_limited", "4xx", 404])
def test_a_code_that_is_not_a_decimal_string_falls_back_to_the_default(carried_code: object):
    """Only ProxyException's stringified status is a status; ``code`` on anything else
    (OpenAI's ``invalid_api_key``, a stray int) says nothing about the HTTP answer."""

    class _Carrier(Exception):
        code = carried_code

    assert error_status_code(_Carrier("boom"), 500) == 500


def test_a_status_code_wins_over_a_stringified_code():
    class _Carrier(Exception):
        status_code = 429
        code = "400"

    assert error_status_code(_Carrier("boom"), 500) == 429


def test_a_status_carried_by_an_exception_drives_the_type_it_reports():
    """The two helpers compose at every call site: the status the exception carries is what
    names its type, not the default the route would have used."""
    exc = HTTPException(status_code=403, detail="blocked by policy")

    assert openai_error_type(exc, error_status_code(exc, 400)) == "permission_error"
