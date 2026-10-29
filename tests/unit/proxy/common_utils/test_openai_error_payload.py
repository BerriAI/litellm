import json
import sys
from typing import Final

import httpx
import prisma.errors
import pytest
from fastapi import HTTPException
from prisma._async_http import Response
from prisma.engine.utils import handle_response_errors

import litellm
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded
from litellm.proxy.common_utils.openai_error_payload import (
    error_status_code,
    litellm_call_id_headers,
    openai_error_param,
    openai_error_type,
    proxy_exception_for,
    with_litellm_call_id,
)
from litellm.types.router import RouterErrors


@pytest.mark.parametrize(
    "status_code, expected_type",
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "invalid_request_error"),
        (408, "invalid_request_error"),
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


def test_an_upstream_5xx_body_does_not_relabel_the_internal_server_error():
    from litellm.exceptions import InternalServerError

    carried = InternalServerError(
        message="Controlled provider failure",
        model="gpt-5.4-mini",
        llm_provider="openai",
        body={"message": "Controlled provider failure", "type": "server_error", "code": "500"},
    )

    assert carried.body == {"message": "Controlled provider failure", "type": "server_error", "code": "500"}
    assert openai_error_type(carried, error_status_code(carried, 400)) == "internal_server_error"


def test_a_stringified_none_type_or_param_is_treated_as_absent():
    from litellm.exceptions import BadRequestError

    carried = BadRequestError(
        message="Content blocked",
        model="claude-haiku-4-5",
        llm_provider="litellm_proxy",
        body={"message": "Content blocked", "type": "None", "param": "None", "code": "400"},
    )

    assert carried.type == "None"
    assert openai_error_type(carried, 400) == "invalid_request_error"
    assert openai_error_param(carried) is None


def test_a_failed_request_answers_with_the_call_id_it_was_logged_under():
    assert litellm_call_id_headers("call-7836") == {"x-litellm-call-id": "call-7836"}
    assert litellm_call_id_headers(None) is None


def test_an_already_shaped_proxy_error_answers_with_the_call_id_it_was_logged_under():
    raised_without_id = ProxyException(message="budget exceeded", type="budget_exceeded", param="key", code=402)

    carried = with_litellm_call_id(raised_without_id, "call-7836")

    assert carried is raised_without_id
    assert carried.headers == {"x-litellm-call-id": "call-7836"}
    assert (carried.message, carried.type, carried.param, carried.code) == (
        "budget exceeded",
        "budget_exceeded",
        "key",
        "402",
    )


def test_a_proxy_error_keeps_the_call_id_it_was_raised_with():
    raised_with_id = ProxyException(
        message="nope", type="None", param=None, code=400, headers={"x-litellm-call-id": "first"}
    )

    assert with_litellm_call_id(raised_with_id, "second").headers == {"x-litellm-call-id": "first"}
    assert with_litellm_call_id(ProxyException(message="nope", type="None", param=None, code=400), None).headers == {}


@pytest.mark.parametrize(
    "raised_status, expected_type",
    [
        (400, "invalid_request_error"),
        (404, "invalid_request_error"),
        (401, "auth_error"),
        (403, "auth_error"),
        (422, "invalid_request_error"),
        (500, "internal_server_error"),
    ],
)
def test_a_catch_all_handler_keeps_the_status_its_route_answered_with(raised_status: int, expected_type: str) -> None:
    wrapped: Final = proxy_exception_for(
        HTTPException(status_code=raised_status, detail={"message": "boom"}),
        default_status_code=400,
    )

    assert (wrapped.code, wrapped.type) == (str(raised_status), expected_type)


@pytest.mark.parametrize(
    "default_status_code, expected_type", [(400, "invalid_request_error"), (500, "internal_server_error")]
)
def test_an_exception_carrying_no_status_falls_back_to_the_routes_default(default_status_code: int, expected_type: str) -> None:
    wrapped: Final = proxy_exception_for(ValueError("boom"), default_status_code=default_status_code)

    assert (wrapped.code, wrapped.type) == (str(default_status_code), expected_type)


def test_a_proxy_exception_passes_through_with_the_type_it_already_named() -> None:
    raised: Final = ProxyException(
        message="Budget has been exceeded",
        type=ProxyErrorTypes.budget_exceeded.value,
        param="max_budget",
        code=400,
    )

    assert proxy_exception_for(raised, default_status_code=500) is raised


def test_the_message_is_what_the_route_raised_without_an_invented_prefix() -> None:
    assert proxy_exception_for(ValueError("boom"), default_status_code=400).message == "boom"


def test_an_http_exceptions_detail_survives_the_wrapping() -> None:
    wrapped: Final = proxy_exception_for(
        HTTPException(status_code=400, detail={"error": "Invalid budget_duration 'not-a-duration'."}),
        default_status_code=400,
    )

    assert "Invalid budget_duration" in wrapped.message


def test_a_carried_type_survives_the_wrapping() -> None:
    class _Carrier(Exception):
        type: str = "context_window_exceeded"
        status_code: int = 400

    assert proxy_exception_for(_Carrier("boom"), default_status_code=500).type == "context_window_exceeded"


@pytest.mark.parametrize("exc", [HTTPException(status_code=404, detail="nope"), ValueError("boom")])
def test_the_wrapped_error_serializes_as_an_openai_error_object(exc: Exception) -> None:
    body: Final = json.loads(json.dumps({"error": proxy_exception_for(exc, default_status_code=400).to_dict()}))

    assert body["error"]["param"] is None
    assert body["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "provider_failure, expected",
    [
        (
            litellm.AuthenticationError(message="invalid x-api-key", llm_provider="anthropic", model="claude-haiku-5-5"),
            ("401", "authentication_error"),
        ),
        (
            litellm.PermissionDeniedError(
                message="model not allowed",
                llm_provider="anthropic",
                model="claude-haiku-5-5",
                response=httpx.Response(403, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")),
            ),
            ("403", "permission_error"),
        ),
    ],
)
def test_a_provider_credential_failure_is_labelled_the_way_chat_completions_labels_it(
    provider_failure: Exception, expected: tuple[str, str]
) -> None:
    wrapped: Final = proxy_exception_for(provider_failure, default_status_code=400)

    assert (wrapped.code, wrapped.type) == expected


@pytest.mark.parametrize("status_code", [401, 403])
def test_a_rejection_the_route_raised_keeps_the_auth_label(status_code: int) -> None:
    wrapped: Final = proxy_exception_for(HTTPException(status_code=status_code, detail="no"), default_status_code=400)

    assert (wrapped.code, wrapped.type) == (str(status_code), "auth_error")


def test_an_exception_without_a_message_is_named_by_its_type() -> None:
    wrapped: Final = proxy_exception_for(TimeoutError(), default_status_code=500)

    assert wrapped.message == "TimeoutError"


def test_a_database_outage_answers_503_so_callers_retry_instead_of_fixing_their_request() -> None:
    wrapped: Final = proxy_exception_for(
        prisma.errors.PrismaError("Can't reach database server at `db:5432`"), default_status_code=400
    )

    assert (wrapped.code, wrapped.type) == ("503", "no_db_connection")


@pytest.mark.parametrize(
    "postgres_refusal",
    [
        'FATAL: database "litellm" is not currently accepting connections',
        "FATAL: the database system is starting up",
        "FATAL: the database system is shutting down",
        "FATAL: the database system is in recovery mode",
        "FATAL: the database system is not yet accepting connections",
        "FATAL: terminating connection due to administrator command",
        "ERROR: cannot execute UPDATE in a read-only transaction",
    ],
)
def test_a_database_refusing_work_during_a_failover_answers_503(postgres_refusal: str) -> None:
    refused: Final = prisma.errors.DataError(
        {"user_facing_error": {"message": f"Error in connector: Error querying the database: {postgres_refusal}"}}
    )

    wrapped: Final = proxy_exception_for(refused, default_status_code=400)

    assert (wrapped.code, wrapped.type) == ("503", "no_db_connection")


@pytest.mark.parametrize(
    "localized_refusal",
    [
        {
            "error_code": "P2010",
            "message": "Raw query failed. Message: `FEHLER: kann UPDATE nicht in einer Nur-Lese-Transaktion ausführen`",
            "meta": {"code": "25006"},
        },
        {
            "message": 'PostgresError { code: "57P01", message: "Verbindung wird abgebrochen", severity: "FATAL" }',
        },
    ],
)
def test_a_failover_refusal_in_a_localized_postgres_answers_503(localized_refusal: dict[str, object]) -> None:
    refused: Final = prisma.errors.DataError({"user_facing_error": localized_refusal})

    wrapped: Final = proxy_exception_for(refused, default_status_code=400)

    assert (wrapped.code, wrapped.type) == ("503", "no_db_connection")


def test_a_failover_phrase_outside_the_database_is_not_reported_as_a_database_outage() -> None:
    callback_down: Final = ValueError("FATAL: the database system is starting up")

    wrapped: Final = proxy_exception_for(callback_down, default_status_code=400)

    assert (wrapped.code, wrapped.type) == ("400", "invalid_request_error")


def test_a_database_dropped_mid_query_answers_503_though_prisma_fails_to_name_the_error() -> None:
    engine_reply_with_null_meta: Final = [{"user_facing_error": {"error_code": "P1001", "message": "", "meta": None}}]
    with pytest.raises(AttributeError) as raised:
        handle_response_errors(Response(httpx.Response(500)), engine_reply_with_null_meta)

    wrapped: Final = proxy_exception_for(raised.value, default_status_code=500)

    assert (wrapped.code, wrapped.type) == ("503", "no_db_connection")


def test_a_network_failure_outside_the_database_is_not_reported_as_a_database_outage() -> None:
    callback_unreachable: Final = httpx.ConnectError("[Errno 111] Connection refused")

    wrapped: Final = proxy_exception_for(callback_unreachable, default_status_code=500)

    assert (wrapped.code, wrapped.type) == ("500", "internal_server_error")


def _add_member_failing_with(db_error: Exception) -> None:
    try:
        raise db_error
    except Exception as e:
        raise ValueError(f"Error adding member to organization: {e}")


@pytest.mark.parametrize(
    "underlying_error, expected",
    [
        (prisma.errors.PrismaError("Can't reach database server at `db:5432`"), ("503", "no_db_connection")),
        (httpx.ConnectError("[Errno 111] Connection refused"), ("500", "internal_server_error")),
    ],
)
def test_a_route_that_rewraps_its_error_keeps_the_database_outage_it_wrapped(
    underlying_error: Exception, expected: tuple[str, str]
) -> None:
    with pytest.raises(ValueError, match="Error adding member") as raised:
        _add_member_failing_with(underlying_error)

    wrapped: Final = proxy_exception_for(raised.value, default_status_code=500)

    assert (wrapped.code, wrapped.type) == expected


def test_a_proxy_without_prisma_installed_still_answers_with_an_error_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "prisma", None)

    wrapped: Final = proxy_exception_for(httpx.ConnectError("[Errno 111] Connection refused"), default_status_code=500)

    assert (wrapped.code, wrapped.type) == ("500", "internal_server_error")


def _look_up_team_failing_with(lookup_error: Exception) -> None:
    try:
        raise lookup_error
    except Exception:
        raise HTTPException(status_code=404, detail="Team doesn't exist in db")


@pytest.mark.parametrize(
    "lookup_error, expected",
    [
        (DBLookupDeadlineExceeded("team", 10.0), ("503", "no_db_connection")),
        (prisma.errors.PrismaError("Can't reach database server at `db:5432`"), ("503", "no_db_connection")),
        (KeyError("team_id"), ("404", "invalid_request_error")),
    ],
)
def test_a_not_found_answer_raised_while_the_database_was_down_reports_the_outage(
    lookup_error: Exception, expected: tuple[str, str]
) -> None:
    with pytest.raises(HTTPException) as raised:
        _look_up_team_failing_with(lookup_error)

    wrapped: Final = proxy_exception_for(raised.value, default_status_code=400)

    assert (wrapped.code, wrapped.type) == expected


def test_a_lookup_that_missed_its_deadline_answers_503() -> None:
    wrapped: Final = proxy_exception_for(DBLookupDeadlineExceeded("team", 10.0), default_status_code=400)

    assert (wrapped.code, wrapped.type) == ("503", "no_db_connection")


def test_a_database_outage_answer_does_not_reveal_where_the_database_lives() -> None:
    wrapped: Final = proxy_exception_for(
        prisma.errors.PrismaError("Can't reach database server at `prod-db.internal:5432`"), default_status_code=400
    )

    assert wrapped.code == "503"
    assert "prod-db.internal" not in wrapped.message


def test_an_upstream_failure_answers_with_its_own_status_not_the_routes_default() -> None:
    upstream_overloaded: Final = litellm.ServiceUnavailableError(
        message="upstream overloaded", llm_provider="openai", model="gpt-5.4-mini"
    )

    wrapped: Final = proxy_exception_for(upstream_overloaded, default_status_code=400)

    assert (wrapped.code, wrapped.type) == ("503", "internal_server_error")


@pytest.mark.parametrize(
    "router_rejection, expected",
    [
        (f"{RouterErrors.no_deployments_with_tag_routing.value}. Passed model=gpt-5.4-mini", ("401", "auth_error")),
        (f"{RouterErrors.no_deployments_available.value}. Passed model=gpt-5.4-mini", ("429", "rate_limit_error")),
    ],
)
def test_a_router_rejection_is_labelled_by_the_status_it_answers_with(
    router_rejection: str, expected: tuple[str, str]
) -> None:
    wrapped: Final = proxy_exception_for(ValueError(router_rejection), default_status_code=400)

    assert (wrapped.code, wrapped.type) == expected
