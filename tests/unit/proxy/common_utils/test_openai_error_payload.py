import functools
import json
import sys
from collections.abc import Callable
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
    relayed_provider_specific_fields,
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


@pytest.mark.parametrize(
    "carried_status",
    [True, False, "slow down", "", None, 429.0, 200, 1301, 0, "200", "1301", "0", "0429", " 429", "\u0664\u0662\u0669", "4" * 5000],
)
def test_a_carried_status_that_is_not_an_error_status_falls_back_to_the_default(carried_status: object):
    """True is an int in Python but not an HTTP status, and answering a 200 or a 1301 either reads
    as success or makes uvicorn drop the connection."""

    class _Carrier(Exception):
        status_code = carried_status

    assert error_status_code(_Carrier("boom"), 500) == 500


def test_a_status_carried_as_a_decimal_string_is_answered_as_that_status():
    """Provider stream-error chunks pass the vendor's code through as a string ``status_code``,
    and the shared handler used to answer it as is, so dropping it turned a rate limit into a 500."""

    class _Carrier(Exception):
        status_code = "429"

    assert error_status_code(_Carrier("slow down"), 500) == 429
    assert proxy_exception_for(_Carrier("slow down"), default_status_code=500).to_dict() == {
        "message": "slow down",
        "type": "rate_limit_error",
        "param": None,
        "code": "429",
    }


def test_a_proxy_exception_keeps_the_status_it_was_raised_with():
    """ProxyException stores its status as the string ``code`` rather than ``status_code``,
    so a route tail that rewraps one used to answer a 4xx rejection as a 500."""
    rejection = ProxyException(message="session_id is required", type="bad_request_error", param="session_id", code=400)

    assert error_status_code(rejection, 500) == 400


@pytest.mark.parametrize("carried_code", [None, "None", "", "rate_limited", "4xx", 404, "200", "1301"])
def test_a_code_that_is_not_a_stringified_error_status_falls_back_to_the_default(carried_code: object):
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


@pytest.mark.parametrize(
    "leakable",
    [
        "sk-9876543210abcdef9876543210",
        "guard-scan.internal.corp",
        "/srv/secret/app/guard.py",
    ],
)
def test_openai_error_type_and_param_redact_secrets_and_internal_details(leakable: str) -> None:
    class _Carrier(Exception):
        type: str = f"BadTypeError {leakable}"
        param: str = f"bad_field {leakable}"

    carried: Final = _Carrier("boom")
    relayed_type: Final = openai_error_type(carried, 400)
    relayed_param: Final = openai_error_param(carried)

    assert relayed_type == openai_error_type(Exception("boom"), 400)
    assert relayed_param is not None
    assert leakable not in relayed_param
    assert relayed_param.startswith("bad_field ")


def test_a_type_or_param_that_redaction_empties_falls_back_to_the_status_type_and_null() -> None:
    class _Carrier(Exception):
        type: str = "Traceback (most recent call last):"
        param: str = "Traceback (most recent call last):"

    carried: Final = _Carrier("boom")

    assert openai_error_type(carried, 400) == openai_error_type(Exception("boom"), 400)
    assert openai_error_param(carried) is None


@pytest.mark.parametrize("carried_type", ["insufficient_quota", "invalid_request_error", "server_error"])
def test_openai_error_type_relays_a_snake_case_type_the_backend_sent(carried_type: str) -> None:
    class _Carrier(Exception):
        type: str = carried_type

    assert openai_error_type(_Carrier("boom"), 400) == carried_type


@pytest.mark.parametrize("carried_type", ["BadRequestError", "Bad Request", "invalid-request", "1211"])
def test_openai_error_type_answers_the_status_type_for_a_type_outside_openais_vocabulary(carried_type: str) -> None:
    class _Carrier(Exception):
        type: str = carried_type

    assert openai_error_type(_Carrier("boom"), 400) == "invalid_request_error"
    assert openai_error_type(_Carrier("boom"), 429) == "rate_limit_error"


@pytest.mark.parametrize(
    "body",
    [
        {"object": "error", "message": "rejected", "type": "BadRequestError", "param": None, "code": 400},
        {"error": "Input validation error: `max_new_tokens` must be strictly positive", "error_type": "validation"},
        {"detail": [{"loc": ["body", "max_tokens"], "msg": "must be >= 1", "type": "value_error"}]},
    ],
    ids=["vllm", "text_generation_inference", "fastapi_validation"],
)
def test_relayed_provider_specific_fields_ignore_bodies_with_nothing_beyond_openais_error_object(
    body: dict[str, object],  # mutable-ok: BadRequestError.body is a dict
) -> None:
    rejected: Final = litellm.BadRequestError(message="rejected", model="qwen", llm_provider="hosted_vllm", body=body)

    assert relayed_provider_specific_fields(rejected) is None


def test_relayed_provider_specific_fields_keep_what_an_openai_shaped_body_adds() -> None:
    rejected: Final = litellm.BadRequestError(
        message="filtered",
        model="gpt-5.4-nano",
        llm_provider="azure",
        body={
            "message": "filtered",
            "type": None,
            "param": "prompt",
            "code": "content_filter",
            "innererror": {"code": "ResponsibleAIPolicyViolation"},
        },
    )

    assert relayed_provider_specific_fields(rejected) == {
        "code": "content_filter",
        "innererror": {"code": "ResponsibleAIPolicyViolation"},
    }


@pytest.mark.parametrize("provider_code", ["previous_response_not_found", 1211])
def test_relayed_provider_specific_fields_keep_a_provider_code_that_is_not_the_http_status(
    provider_code: str | int,
) -> None:
    rejected: Final = litellm.BadRequestError(
        message="Previous response not found.",
        model="gpt-5.6",
        llm_provider="openai",
        body={
            "message": "Previous response not found.",
            "type": "invalid_request_error",
            "param": "previous_response_id",
            "code": provider_code,
        },
    )

    assert relayed_provider_specific_fields(rejected) == {"code": provider_code}


def test_relayed_provider_specific_fields_keep_the_code_of_an_error_body_without_a_message() -> None:
    rejected: Final = litellm.BadRequestError(
        message="rejected",
        model="grok-4.7",
        llm_provider="xai",
        body={"code": "invalid-argument", "error": "Maximum number of tokens max_tokens must be positive"},
    )

    assert relayed_provider_specific_fields(rejected) == {"code": "invalid-argument"}


def test_relayed_provider_specific_fields_redact_a_secret_from_carried_fields_and_body_keys() -> None:
    secret: Final = "sk-9876543210abcdef9876543210"
    rejected: Final = litellm.ContentPolicyViolationError(
        message="filtered",
        model="gpt-5.4-nano",
        llm_provider="azure",
        provider_specific_fields={"innererror": {"detail": f"blocked for key {secret}"}},
        body={"message": "filtered", "code": "content_filter", f"key {secret}": "leaked"},
    )

    relayed: Final = relayed_provider_specific_fields(rejected)

    assert relayed is not None
    assert relayed["code"] == "content_filter"
    assert secret not in json.dumps(relayed)


@pytest.mark.parametrize(
    "credential",
    [{"value": "opaque-upstream-credential"}, [{"value": "opaque-upstream-credential"}]],
    ids=["object", "list_of_objects"],
)
def test_relayed_provider_specific_fields_redact_a_structured_value_under_a_credential_key(credential: object) -> None:
    rejected: Final = litellm.BadRequestError(
        message="rejected",
        model="qwen",
        llm_provider="hosted_vllm",
        body={"message": "rejected", "client_secret": credential, "upstream": {"region": "eu"}},
    )

    relayed: Final = relayed_provider_specific_fields(rejected)

    assert relayed is not None
    assert relayed["upstream"] == {"region": "eu"}
    assert "opaque-upstream-credential" not in json.dumps(relayed)


@pytest.mark.parametrize("header", ["Cookie", "set-cookie", "Proxy-Authorization", "x_amz_security_token"])
@pytest.mark.parametrize(
    "nesting",
    [
        lambda credential: credential,
        lambda credential: {"request_headers": credential},
        lambda credential: {"sent": [credential]},
    ],
    ids=["top_level", "nested", "inside_a_list"],
)
def test_relayed_provider_specific_fields_redact_a_credential_header(
    header: str,
    nesting: Callable[[dict[str, str]], dict[str, object]],
) -> None:
    rejected: Final = litellm.BadRequestError(
        message="rejected",
        model="qwen",
        llm_provider="hosted_vllm",
        body={"message": "rejected", "region": "eu", **nesting({header: "opaque-upstream-credential"})},
    )

    relayed: Final = relayed_provider_specific_fields(rejected)

    assert relayed is not None
    assert relayed["region"] == "eu"
    assert "opaque-upstream-credential" not in json.dumps(relayed)


@pytest.mark.parametrize(
    ("redaction_enabled", "expected_region"),
    [(True, "eu"), (False, None)],
    ids=["redaction_on", "redaction_off"],
)
def test_relayed_provider_specific_fields_survive_a_body_nested_past_the_recursion_limit(
    redaction_enabled: bool,
    expected_region: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("litellm._logging._ENABLE_SECRET_REDACTION", redaction_enabled)
    nested: Final = functools.reduce(
        lambda inner, _: {"inner": inner},
        range(50_000),
        "deep-leaf",
    )
    rejected: Final = litellm.BadRequestError(
        message="rejected",
        model="qwen",
        llm_provider="hosted_vllm",
        body={"message": "rejected", "region": "eu", "upstream": nested},
    )

    relayed: Final = relayed_provider_specific_fields(rejected)

    assert (relayed or {}).get("region") == expected_region
    assert "deep-leaf" not in json.dumps(relayed)


def test_relayed_provider_specific_fields_survive_a_zero_recursion_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("litellm._logging.DEFAULT_MAX_RECURSE_DEPTH", 0)
    rejected: Final = litellm.BadRequestError(
        message="rejected",
        model="qwen",
        llm_provider="hosted_vllm",
        body={"message": "rejected", "trace_ref": "trace_1", "detail": {"inner": "value"}},
    )

    relayed: Final = relayed_provider_specific_fields(rejected)

    assert relayed is not None
    assert relayed["trace_ref"] == "trace_1"
    assert "value" not in json.dumps(relayed)


@pytest.mark.parametrize(
    "field",
    [
        "token_value",
        "client-secret",
        "clientsecret",
        "privateKey",
        "x-internal-token",
        "accessToken",
        "pass",
        "pwd",
        "signature",
        "session_id",
    ],
)
@pytest.mark.parametrize("nested", [False, True], ids=["top_level", "nested"])
def test_relayed_provider_specific_fields_redact_any_credential_field_name(field: str, nested: bool) -> None:
    fields: Final = {field: "opaque-upstream-credential"}
    rejected: Final = litellm.BadRequestError(
        message="rejected",
        model="qwen",
        llm_provider="hosted_vllm",
        body={"message": "rejected", **({"debug": fields} if nested else fields)},
    )

    relayed: Final = relayed_provider_specific_fields(rejected)

    assert relayed is not None
    assert "opaque-upstream-credential" not in json.dumps(relayed)


@pytest.mark.parametrize(
    "field",
    [
        "api_key_used",
        "password_max",
        "secret_type",
        "token_expires",
        "token_count",
        "subscription_key",
        "license_key",
        "connectionString",
        "accountKey",
        "sharedKey",
        "masterKey",
        "encryption_key",
    ],
)
@pytest.mark.parametrize(
    "nesting",
    [
        lambda fields: fields,
        lambda fields: {"debug": fields},
        lambda fields: {"debug": [fields]},
    ],
    ids=["top_level", "nested", "inside_a_list"],
)
def test_relayed_provider_specific_fields_redact_text_under_a_credential_name_that_also_reads_as_a_quantity_or_key(
    field: str,
    nesting: Callable[[dict[str, str]], dict[str, object]],
) -> None:
    rejected: Final = litellm.BadRequestError(
        message="rejected",
        model="qwen",
        llm_provider="hosted_vllm",
        body={"message": "rejected", "region": "eu", **nesting({field: "opaque-upstream-credential"})},
    )

    relayed: Final = relayed_provider_specific_fields(rejected)

    assert relayed is not None
    assert relayed["region"] == "eu"
    assert "opaque-upstream-credential" not in json.dumps(relayed)


@pytest.mark.parametrize(
    "quantity",
    [3, 2.5, True, None, {"prompt": 3, "completion": 4}, [3, 4]],
    ids=["int", "float", "bool", "null", "object_of_counts", "list_of_counts"],
)
@pytest.mark.parametrize("field", ["token_count", "password_max", "api_key_used", "token_usage", "session_ttl"])
def test_relayed_provider_specific_fields_keep_a_quantity_under_a_credential_sounding_name(
    field: str, quantity: object
) -> None:
    rejected: Final = litellm.BadRequestError(
        message="rejected", model="qwen", llm_provider="hosted_vllm", body={"message": "rejected", field: quantity}
    )

    assert relayed_provider_specific_fields(rejected) == {field: quantity}


@pytest.mark.parametrize(
    "field", ["max_tokens", "prompt_tokens", "request_id", "key_id", "keyword", "passed", "bypass", "cache_key"]
)
def test_relayed_provider_specific_fields_keep_a_plain_field_name(field: str) -> None:
    rejected: Final = litellm.BadRequestError(
        message="rejected", model="qwen", llm_provider="hosted_vllm", body={"message": "rejected", field: "kept"}
    )

    assert relayed_provider_specific_fields(rejected) == {field: "kept"}


def test_relayed_provider_specific_fields_redact_a_credential_inside_json_text() -> None:
    raw: Final = json.dumps({"error": "blocked", "headers": {"Cookie": "session=opaque-upstream-credential"}})
    rejected: Final = litellm.BadRequestError(
        message="rejected",
        model="m",
        llm_provider="openrouter",
        body={"message": "rejected", "code": 400, "metadata": {"raw": raw, "provider_name": "Upstream"}},
    )

    assert relayed_provider_specific_fields(rejected) == {
        "metadata": {
            "raw": json.dumps({"error": "blocked", "headers": {"Cookie": "REDACTED"}}, separators=(",", ":")),
            "provider_name": "Upstream",
        }
    }


def test_relayed_provider_specific_fields_keep_clean_json_text_byte_for_byte() -> None:
    raw: Final = '{"error": "blocked",  "detail": [1, 2]}'
    rejected: Final = litellm.BadRequestError(
        message="rejected",
        model="m",
        llm_provider="openrouter",
        body={"message": "rejected", "code": 400, "metadata": {"raw": raw}},
    )

    assert relayed_provider_specific_fields(rejected) == {"metadata": {"raw": raw}}
