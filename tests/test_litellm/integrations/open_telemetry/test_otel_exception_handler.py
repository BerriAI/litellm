"""LIT-3193 — exception-handler path. Closes SERVER spans for requests
that fail after auth but before the route handler runs (e.g. /model/new
TypeError or RequestValidationError)."""

import asyncio
import types

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

import litellm.proxy.proxy_server as proxy_server_module
from litellm.proxy._types import ProxyException
from litellm.proxy.proxy_server import (
    _close_dangling_otel_server_span,
    openai_exception_handler,
    otel_request_validation_exception_handler,
    otel_unhandled_exception_handler,
)

from ._helpers import assert_server_span_attrs


def _fake_request(parent_otel_span=None):
    state = types.SimpleNamespace()
    if parent_otel_span is not None:
        state.parent_otel_span = parent_otel_span
    return types.SimpleNamespace(state=state)


@pytest.fixture
def wired_otel(otel_with_exporter, monkeypatch):
    otel, exporter = otel_with_exporter
    monkeypatch.setattr(proxy_server_module, "open_telemetry_logger", otel)
    return exporter


@pytest.mark.parametrize("status,path", [(500, "/model/new"), (422, "/key/generate")])
def test_close_dangling_span_stamps_status(
    wired_otel, server_span_factory, status, path
):
    request = _fake_request(parent_otel_span=server_span_factory(path))
    _close_dangling_otel_server_span(request, status)
    assert_server_span_attrs(
        wired_otel,
        expected_status=status,
        expected_url_path=path,
        where=f"{path} {status}",
    )
    assert request.state.parent_otel_span is None


def test_close_dangling_span_noop_when_no_span(wired_otel):
    _close_dangling_otel_server_span(_fake_request(), 500)
    assert wired_otel.get_finished_spans() == ()


def test_close_dangling_span_noop_when_otel_absent(server_span_factory, monkeypatch):
    monkeypatch.setattr(proxy_server_module, "open_telemetry_logger", None)
    request = _fake_request(parent_otel_span=server_span_factory("/key/generate"))
    _close_dangling_otel_server_span(request, 500)


@pytest.mark.parametrize(
    "handler,exc,status,path",
    [
        (
            otel_request_validation_exception_handler,
            RequestValidationError(errors=[]),
            422,
            "/key/generate",
        ),
        (
            otel_unhandled_exception_handler,
            TypeError("Deployment.__init__() missing required positional arg"),
            500,
            "/model/new",
        ),
    ],
)
def test_exception_handler_closes_span(
    wired_otel, server_span_factory, handler, exc, status, path
):
    request = _fake_request(parent_otel_span=server_span_factory(path))
    response = asyncio.run(handler(request, exc))
    assert response.status_code == status
    assert_server_span_attrs(
        wired_otel,
        expected_status=status,
        expected_url_path=path,
        where=f"{handler.__name__} ({type(exc).__name__})",
    )


def test_unhandled_handler_reraises_known_exceptions(wired_otel, server_span_factory):
    """ProxyException / HTTPException / RequestValidationError have dedicated handlers."""
    request = _fake_request(parent_otel_span=server_span_factory("/key/generate"))
    with pytest.raises(HTTPException):
        asyncio.run(
            otel_unhandled_exception_handler(
                request, HTTPException(status_code=403, detail="forbidden")
            )
        )


# Covers ProxyException raised after auth stashed the span (e.g., invalid-JSON
# body via _read_request_body) — handler must close the dangling SERVER span.
@pytest.mark.parametrize(
    "code,path",
    [
        (400, "/v1/chat/completions"),
        (400, "/v1/messages"),
        (400, "/v1/responses"),
        (429, "/v1/chat/completions"),
        (503, "/v1/chat/completions"),
    ],
)
def test_openai_exception_handler_closes_span(
    wired_otel, server_span_factory, code, path
):
    request = _fake_request(parent_otel_span=server_span_factory(path))
    exc = ProxyException(
        message="boom",
        type="invalid_request_error",
        param="request_body",
        code=code,
    )
    response = asyncio.run(openai_exception_handler(request, exc))
    assert response.status_code == code
    assert_server_span_attrs(
        wired_otel,
        expected_status=code,
        expected_url_path=path,
        where=f"openai_exception_handler ({path} code={code})",
    )
    assert request.state.parent_otel_span is None
