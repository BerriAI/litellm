import functools
import json
from collections.abc import Callable
from typing import Final

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.common_utils.openai_error_payload import (
    error_status_code,
    litellm_call_id_headers,
    openai_error_param,
    openai_error_type,
    relayed_provider_specific_fields,
    with_litellm_call_id,
)


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
    "field", ["max_tokens", "prompt_tokens", "token_count", "request_id", "key_id", "keyword", "passed", "bypass"]
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
