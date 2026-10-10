"""Behavior pins for the proxy_server exception handlers.

Pins covered:
- ``openai_exception_handler``
- ``_close_dangling_otel_server_span``
- ``otel_request_validation_exception_handler``
- ``otel_unhandled_exception_handler``
- ``otlp_http_exception_handler``
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
from starlette.types import Message
from starlette.websockets import WebSocket, WebSocketDisconnect

from litellm.proxy._types import ProxyException
from litellm.proxy.proxy_server import (
    _close_dangling_otel_server_span,
    openai_exception_handler,
    otel_request_validation_exception_handler,
    otel_unhandled_exception_handler,
    otlp_http_exception_handler,
)

from .conftest import normalize


def _make_request(parent_otel_span=None, path="/chat/completions"):
    return Request({
        "type": "http", "method": "POST", "path": path, "headers": [],
        "state": {"parent_otel_span": parent_otel_span},
    })


# ---------------------------------------------------------------------------
# openai_exception_handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_exception_handler_returns_mapped_payload():
    exc = ProxyException(
        message="bad input",
        type="invalid_request_error",
        param="model",
        code=400,
    )
    request = _make_request()

    response = await openai_exception_handler(request=request, exc=exc)
    body = json.loads(response.body)

    assert response.status_code == 400
    assert normalize(body) == {
        "error": {
            "message": "bad input",
            "type": "invalid_request_error",
            "param": "model",
            "code": "400",
        }
    }


@pytest.mark.asyncio
async def test_openai_exception_handler_invalid_empty_code_defaults_to_500():
    """openai_exception_handler falls back to 500 when ``code`` is falsy.

    Constructing via __new__ bypasses __init__ — the production __init__ always
    coerces None to the string "None", which is truthy. To exercise the falsy
    fallback branch we hand-craft an exception with an empty code."""
    exc = ProxyException.__new__(ProxyException)
    exc.message = "boom"
    exc.type = "server_error"
    exc.param = None
    exc.openai_code = None
    exc.code = ""
    exc.headers = {}
    exc.provider_specific_fields = None
    request = _make_request()

    response = await openai_exception_handler(request=request, exc=exc)
    body = json.loads(response.body)

    assert response.status_code == 500
    assert body == {
        "error": {
            "message": "boom",
            "type": "server_error",
            "param": None,
            "code": "",
        }
    }


def _call_id_exception(headers):
    return ProxyException(
        message="bad input",
        type="invalid_request_error",
        param="model",
        code=400,
        headers=headers,
    )


@pytest.mark.asyncio
async def test_openai_exception_handler_copies_the_call_id_into_the_error_when_opted_in(monkeypatch):
    """With include_call_id_in_error_body on, error.litellm_call_id is byte-identical to the
    x-litellm-call-id header, so a pasted str(e) names the request to look up."""
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {"include_call_id_in_error_body": True})
    exc = _call_id_exception({"x-litellm-call-id": "call-8302"})

    response = await openai_exception_handler(request=_make_request(), exc=exc)
    body = json.loads(response.body)

    assert response.headers["x-litellm-call-id"] == "call-8302"
    assert body == {
        "error": {
            "message": "bad input",
            "type": "invalid_request_error",
            "param": "model",
            "code": "400",
            "litellm_call_id": "call-8302",
        }
    }


@pytest.mark.asyncio
async def test_openai_exception_handler_leaves_the_error_alone_when_opted_out(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    exc = _call_id_exception({"x-litellm-call-id": "call-8302"})

    response = await openai_exception_handler(request=_make_request(), exc=exc)
    body = json.loads(response.body)

    assert response.headers["x-litellm-call-id"] == "call-8302"
    assert body == {
        "error": {
            "message": "bad input",
            "type": "invalid_request_error",
            "param": "model",
            "code": "400",
        }
    }


@pytest.mark.asyncio
async def test_openai_exception_handler_never_fabricates_a_call_id(monkeypatch):
    """An error raised before a call id exists (auth failures, say) carries no header,
    and the body must not invent one."""
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {"include_call_id_in_error_body": True})
    exc = _call_id_exception({})

    response = await openai_exception_handler(request=_make_request(), exc=exc)
    body = json.loads(response.body)

    assert "x-litellm-call-id" not in response.headers
    assert "litellm_call_id" not in body["error"]


# ---------------------------------------------------------------------------
# _close_dangling_otel_server_span
# ---------------------------------------------------------------------------


def test_close_dangling_otel_server_span_records_status_and_ends(monkeypatch):
    """Happy path: with a logger and an active span, the handler sets the
    response status, marks ERROR (>=400), ends the span, and clears state."""
    import litellm.proxy.proxy_server as ps

    span = MagicMock()
    fake_logger = MagicMock()
    monkeypatch.setattr(ps, "open_telemetry_logger", fake_logger, raising=False)
    request = _make_request(parent_otel_span=span)

    _close_dangling_otel_server_span(request=request, status_code=502)

    observed = {
        "status_attr_called": fake_logger.set_response_status_code_attribute.called,
        "set_status_called": span.set_status.called,
        "ended": span.end.called,
        "state_cleared": request.state.parent_otel_span is None,
    }
    assert normalize(observed) == {
        "status_attr_called": True,
        "set_status_called": True,
        "ended": True,
        "state_cleared": True,
    }


def test_close_dangling_otel_server_span_v2_stamps_error_without_ending(monkeypatch):
    """LIT-4179: under OTel v2 the FastAPI instrumentor owns the SERVER span, so
    the handler must only stamp error.* on it (via record_error_attributes_on_span)
    and must NOT set status, end the span, or clear request state — otherwise the
    instrumentor's http.* attributes and span close are lost."""
    import litellm.integrations.otel.model.config as otel_config
    import litellm.proxy.proxy_server as ps

    span = MagicMock()
    fake_logger = MagicMock()
    monkeypatch.setattr(ps, "open_telemetry_logger", fake_logger, raising=False)
    monkeypatch.setattr(otel_config, "is_otel_v2_enabled", lambda: True)
    request = _make_request(parent_otel_span=span)
    exc = ProxyException(message="bad", type="bad_request_error", param=None, code=400)

    _close_dangling_otel_server_span(request=request, status_code=422, exc=exc)

    fake_logger.record_error_attributes_on_span.assert_called_once_with(span, exc, 422)
    assert not span.end.called
    assert not span.set_status.called
    assert not fake_logger.set_response_status_code_attribute.called
    assert request.state.parent_otel_span is span


def test_close_dangling_otel_server_span_v2_success_does_not_stamp(monkeypatch):
    """Under v2 a sub-400 status must not stamp an error onto the SERVER span."""
    import litellm.integrations.otel.model.config as otel_config
    import litellm.proxy.proxy_server as ps

    span = MagicMock()
    fake_logger = MagicMock()
    monkeypatch.setattr(ps, "open_telemetry_logger", fake_logger, raising=False)
    monkeypatch.setattr(otel_config, "is_otel_v2_enabled", lambda: True)
    request = _make_request(parent_otel_span=span)

    _close_dangling_otel_server_span(request=request, status_code=200)

    assert not fake_logger.record_error_attributes_on_span.called
    assert not span.end.called


def test_close_dangling_otel_server_span_missing_span_is_noop_error():
    """When parent_otel_span is missing the call short-circuits — no error."""
    request = _make_request(parent_otel_span=None)

    result = _close_dangling_otel_server_span(request=request, status_code=200)
    assert result is None
    assert request.state.parent_otel_span is None


def test_close_dangling_otel_server_span_logger_raises_state_cleared_error(monkeypatch):
    """Logger raising is caught; state.parent_otel_span is cleared regardless."""
    import litellm.proxy.proxy_server as ps

    span = MagicMock()
    fake_logger = MagicMock()
    fake_logger.set_response_status_code_attribute.side_effect = RuntimeError("boom")
    monkeypatch.setattr(ps, "open_telemetry_logger", fake_logger, raising=False)
    request = _make_request(parent_otel_span=span)

    _close_dangling_otel_server_span(request=request, status_code=500)

    assert request.state.parent_otel_span is None


# ---------------------------------------------------------------------------
# otel_request_validation_exception_handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_otel_request_validation_exception_handler_returns_422_detail():
    errors = [{"loc": ["body", "model"], "msg": "field required", "type": "missing", "input": {"messages": []}}]
    exc = RequestValidationError(errors)
    request = _make_request()

    response = await otel_request_validation_exception_handler(request=request, exc=exc)
    body = json.loads(response.body)

    assert response.status_code == 422
    assert body == {"detail": [{"type": "missing", "loc": ["body", "model"], "msg": "field required"}]}


_SUBMITTED_PASSWORD: Final = "hunter2-Sup3rSecret!"
_PASSWORD_LEAKING_ERRORS: Final = (
    {
        "type": "missing",
        "loc": ["body", "new_password"],
        "msg": "Field required",
        "input": {"current_password": _SUBMITTED_PASSWORD},
    },
    {
        "type": "value_error",
        "loc": ["body", "password"],
        "msg": "Value error, password cannot be set via /user/new",
        "input": _SUBMITTED_PASSWORD,
        "ctx": {"error": ValueError(_SUBMITTED_PASSWORD)},
    },
)
_PUBLIC_ERRORS: Final = (
    {"type": "missing", "loc": ["body", "new_password"], "msg": "Field required"},
    {"type": "value_error", "loc": ["body", "password"], "msg": "Value error, password cannot be set via /user/new"},
)


@pytest.mark.asyncio
async def test_otel_request_validation_exception_handler_never_echoes_the_submitted_body():
    """A pydantic error carries the offending value as ``input`` (the whole body for a
    ``missing`` error) and input-derived values in ``ctx``; a caller who mistyped a
    request holding a password must not get that password back."""
    exc = RequestValidationError(list(_PASSWORD_LEAKING_ERRORS))

    response = await otel_request_validation_exception_handler(request=_make_request(), exc=exc)

    assert response.status_code == 422
    assert json.loads(response.body) == {"detail": list(_PUBLIC_ERRORS)}
    assert _SUBMITTED_PASSWORD.encode() not in response.body


@pytest.mark.asyncio
async def test_otel_request_validation_exception_handler_hands_the_span_only_the_public_errors(monkeypatch):
    """The OTEL SERVER span's error message is ``str(exc)``, which FastAPI builds from
    every error dict ``input`` included, so the span gets the same public-only errors
    the caller does, and keeps the traceback the original carried."""
    import litellm.proxy.proxy_server as ps

    fake_logger = MagicMock()
    monkeypatch.setattr(ps, "open_telemetry_logger", fake_logger, raising=False)
    exc = RequestValidationError(list(_PASSWORD_LEAKING_ERRORS))
    try:
        raise exc
    except RequestValidationError as raised:
        original_traceback = raised.__traceback__
    request = _make_request(parent_otel_span=MagicMock())

    await otel_request_validation_exception_handler(request=request, exc=exc)

    (_span, span_exc, status_code) = fake_logger.record_error_attributes_on_span.call_args.args
    assert status_code == 422
    assert isinstance(span_exc, RequestValidationError)
    assert list(span_exc.errors()) == list(_PUBLIC_ERRORS)
    assert _SUBMITTED_PASSWORD not in str(span_exc)
    assert span_exc.__traceback__ is original_traceback


@pytest.mark.asyncio
async def test_otel_request_validation_exception_handler_empty_errors_invalid_payload():
    """An empty error list still returns 422 — the validator emitted nothing
    but the handler must not crash and the body must remain well-formed."""
    exc = RequestValidationError([])
    request = _make_request()

    response = await otel_request_validation_exception_handler(request=request, exc=exc)
    body = json.loads(response.body)

    assert response.status_code == 422
    assert body == {"detail": []}


@pytest.mark.asyncio
async def test_otel_request_validation_exception_handler_returns_a_problem_on_the_control_plane():
    """`/management/v1` answers validation errors as RFC 9457, so a caller there gets a
    400 problem document rather than the proxy-wide 422 `{"detail": [...]}` shape."""
    errors = [
        {"loc": ["query", "page_size"], "msg": "Input should be less than or equal to 100", "type": "less_than_equal"}
    ]
    exc = RequestValidationError(errors)
    request = _make_request(path="/management/v1/spend_logs/end_users")

    response = await otel_request_validation_exception_handler(request=request, exc=exc)
    body = json.loads(response.body)

    assert response.status_code == 400
    assert response.media_type == "application/problem+json"
    assert body["type"].startswith("urn:")
    assert body["status"] == 400
    assert "page_size" in body["detail"]
    assert "detail" in body and not isinstance(body["detail"], list)


@pytest.mark.asyncio
async def test_otel_request_validation_exception_handler_answers_a_bad_control_plane_body_with_422():
    """A request body that fails validation, an unknown field included, is 422 on
    `/management/v1`; only query parameter problems are 400."""
    errors = [
        {"loc": ["body", "users", 0, "user_emial"], "msg": "Extra inputs are not permitted", "type": "extra_forbidden"}
    ]
    exc = RequestValidationError(errors)
    request = _make_request(path="/management/v1/users/bulk")

    response = await otel_request_validation_exception_handler(request=request, exc=exc)
    body = json.loads(response.body)

    assert response.status_code == 422
    assert response.media_type == "application/problem+json"
    assert body["type"] == "urn:litellm:error:invalid-request-body"
    assert body["status"] == 422
    assert "users.0.user_emial: Extra inputs are not permitted" in body["detail"]


@pytest.mark.asyncio
async def test_otel_request_validation_exception_handler_leaves_other_routes_on_422():
    """The problem+json branch is scoped by path prefix. A route that merely contains
    the word management, or sits above the prefix, keeps the shape its callers parse."""
    exc = RequestValidationError([])

    for path in ("/management", "/v1/management/foo", "/customer/list"):
        response = await otel_request_validation_exception_handler(request=_make_request(path=path), exc=exc)

        assert response.status_code == 422, path
        assert json.loads(response.body) == {"detail": []}, path


# ---------------------------------------------------------------------------
# otel_unhandled_exception_handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_otel_unhandled_exception_handler_returns_500_generic_payload():
    exc = RuntimeError("kaboom")
    request = _make_request()

    response = await otel_unhandled_exception_handler(request=request, exc=exc)
    body = json.loads(response.body)

    assert response.status_code == 500
    assert normalize(body) == {
        "error": {
            "message": "Internal server error",
            "type": "internal_server_error",
        }
    }


_DB_OUTAGE_503_BODY: Final = {
    "error": {
        "message": "Service Unavailable, the authentication database is temporarily unreachable. Please retry shortly.",
        "type": "no_db_connection",
        "param": "None",
        "code": "503",
    }
}


def _raised_from(outer: Exception, cause: Exception) -> Exception:
    try:
        raise outer from cause
    except Exception as chained:
        return chained


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("All connection attempts failed"),
        _raised_from(RuntimeError("user read failed"), httpx.ConnectError("All connection attempts failed")),
    ],
    ids=["raw_connect_error", "connect_error_as_cause"],
)
async def test_otel_unhandled_exception_handler_answers_a_db_outage_with_503_no_db_connection(exc):
    response = await otel_unhandled_exception_handler(request=_make_request(path="/v2/team/list"), exc=exc)

    assert response.status_code == 503
    assert json.loads(response.body) == _DB_OUTAGE_503_BODY


@pytest.mark.asyncio
async def test_otel_unhandled_exception_handler_answers_the_control_plane_with_a_500_problem():
    request = _make_request(path="/management/v1/teams/team-1/members")

    response = await otel_unhandled_exception_handler(request=request, exc=RuntimeError("kaboom"))

    assert response.status_code == 500
    assert response.media_type == "application/problem+json"
    assert json.loads(response.body) == {
        "type": "urn:litellm:error:internal-server-error",
        "title": "Internal server error",
        "status": 500,
        "detail": "An unexpected error occurred.",
    }


@pytest.mark.asyncio
async def test_otel_unhandled_exception_handler_answers_a_control_plane_db_outage_with_a_503_problem():
    request = _make_request(path="/management/v1/teams/team-1/members")

    response = await otel_unhandled_exception_handler(
        request=request, exc=httpx.ConnectError("All connection attempts failed")
    )

    assert response.status_code == 503
    assert response.media_type == "application/problem+json"
    assert json.loads(response.body) == {
        "type": "urn:litellm:error:database-not-connected",
        "title": "Database not connected",
        "status": 503,
        "detail": _DB_OUTAGE_503_BODY["error"]["message"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/management", "/v1/management/foo", "/team/info"])
async def test_otel_unhandled_exception_handler_leaves_other_routes_on_the_openai_error_shape(path):
    response = await otel_unhandled_exception_handler(request=_make_request(path=path), exc=RuntimeError("kaboom"))

    assert response.status_code == 500
    assert json.loads(response.body) == {"error": {"message": "Internal server error", "type": "internal_server_error"}}


@pytest.mark.asyncio
async def test_otel_unhandled_exception_handler_reraises_proxy_exception_error():
    """ProxyException / HTTPException / RequestValidationError are re-raised
    so the dedicated handler runs."""
    exc = ProxyException(message="m", type="t", param="p", code=403)
    request = _make_request()

    with pytest.raises(ProxyException):
        await otel_unhandled_exception_handler(request=request, exc=exc)


@pytest.mark.asyncio
async def test_otel_unhandled_exception_handler_reraises_http_exception_invalid():
    request = _make_request()
    with pytest.raises(HTTPException):
        await otel_unhandled_exception_handler(request=request, exc=HTTPException(status_code=418, detail="teapot"))


@pytest.mark.asyncio
@pytest.mark.parametrize("media_type", ["application/json", "application/x-protobuf"])
@pytest.mark.parametrize("root_path", ["", "/tenant-a"])
@pytest.mark.parametrize("native_available", [True, False])
@pytest.mark.parametrize("error", [
    ProxyException("database credentials: secret", "auth_error", None, 401),
    HTTPException(403, "database credentials: secret"),
])
async def test_otlp_auth_errors_hide_internal_details_and_survive_missing_native(
    media_type: str, root_path: str, native_available: bool,
    error: ProxyException | HTTPException, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.rpc.status_pb2 import Status

    from litellm.proxy.proxy_server import otlp_http_exception_handler
    from litellm.rust_bridge import loader

    if not native_available:
        monkeypatch.setattr(loader, "_cached_bridge", None)
    request: Final = Request({
        "type": "http", "method": "POST", "path": root_path + "/v1/traces", "root_path": root_path,
        "headers": [(b"content-type", media_type.encode())],
    })
    response: Final = (
        await openai_exception_handler(request, error)
        if isinstance(error, ProxyException)
        else await otlp_http_exception_handler(request, error)
    )
    assert response is not None
    assert response.status_code == (401 if isinstance(error, ProxyException) else 403)
    assert response.headers["content-type"].startswith(media_type)
    message: Final = (
        json.loads(response.body)["message"]
        if media_type == "application/json"
        else Status.FromString(response.body).message
    )
    expected: Final = "Unauthorized" if isinstance(error, ProxyException) else "Forbidden"
    assert message == (expected if native_available or media_type == "application/json" else "")


def _websocket(sent: list[Message]) -> WebSocket:
    async def receive() -> Message:
        return {"type": "websocket.connect"}

    async def send(message: Message) -> None:
        sent.append(message)

    return WebSocket({"type": "websocket", "path": "/v1/responses", "headers": [], "query_string": b""}, receive, send)


@pytest.mark.asyncio
async def test_http_exception_on_a_plain_http_request_keeps_the_default_json_body() -> None:
    request: Final = _make_request(path="/v1/models")

    response: Final = await otlp_http_exception_handler(request, HTTPException(404, "not found"))

    assert response is not None
    assert response.status_code == 404
    assert json.loads(bytes(response.body)) == {"detail": "not found"}


@pytest.mark.asyncio
async def test_http_exception_on_a_websocket_closed_before_accept_sends_nothing_more() -> None:
    sent: Final[list[Message]] = []
    websocket: Final = _websocket(sent)
    await websocket.close(code=1008)

    response: Final = await otlp_http_exception_handler(websocket, HTTPException(403, "No API key provided"))

    assert response is None
    assert sent == [{"type": "websocket.close", "code": 1008, "reason": ""}]


@pytest.mark.asyncio
async def test_http_exception_on_a_connecting_websocket_denies_the_upgrade_with_its_status() -> None:
    websocket: Final = _websocket([])

    response: Final = await otlp_http_exception_handler(websocket, HTTPException(403, "No API key provided"))

    assert response is not None
    assert response.status_code == 403
    assert json.loads(bytes(response.body)) == {"detail": "No API key provided"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("status_code", "close_code"), [(403, 1008), (429, 1008), (500, 1011), (503, 1011)])
async def test_http_exception_on_an_accepted_websocket_closes_it_with_a_matching_code(
    status_code: int, close_code: int
) -> None:
    sent: Final[list[Message]] = []
    websocket: Final = _websocket(sent)
    await websocket.accept()

    response: Final = await otlp_http_exception_handler(websocket, HTTPException(status_code, "late failure"))

    assert response is None
    assert sent[-1] == {"type": "websocket.close", "code": close_code, "reason": ""}


def test_websocket_auth_rejection_reaches_the_client_as_a_denial_instead_of_a_server_error() -> None:
    async def reject_like_user_api_key_auth_websocket(websocket: WebSocket) -> None:
        await websocket.close(code=1008)
        raise HTTPException(status_code=403, detail="No API key provided")

    async def responses(websocket: WebSocket, _: None = Depends(reject_like_user_api_key_auth_websocket)) -> None:
        await websocket.accept()

    app: Final = FastAPI()
    app.exception_handler(HTTPException)(otlp_http_exception_handler)
    app.add_api_websocket_route("/v1/responses", responses)

    with pytest.raises(WebSocketDisconnect) as disconnect, TestClient(app).websocket_connect("/v1/responses"):
        pass

    assert disconnect.value.code == 1008


def test_websocket_rejected_before_any_close_reaches_the_client_as_an_http_denial_with_the_status() -> None:
    async def reject_without_closing(websocket: WebSocket) -> None:
        raise HTTPException(status_code=403, detail="No API key provided")

    async def responses(websocket: WebSocket, _: None = Depends(reject_without_closing)) -> None:
        await websocket.accept()

    app: Final = FastAPI()
    app.exception_handler(HTTPException)(otlp_http_exception_handler)
    app.add_api_websocket_route("/v1/responses", responses)

    with pytest.raises(WebSocketDenialResponse) as denial, TestClient(app).websocket_connect("/v1/responses"):
        pass

    assert denial.value.status_code == 403
    assert denial.value.json() == {"detail": "No API key provided"}
