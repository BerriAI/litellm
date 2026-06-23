"""Behavior pins for the proxy_server exception handlers.

Pins covered:
- ``openai_exception_handler``
- ``_close_dangling_otel_server_span``
- ``otel_request_validation_exception_handler``
- ``otel_unhandled_exception_handler``
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError

from litellm.proxy._types import ProxyException
from litellm.proxy.proxy_server import (
    _close_dangling_otel_server_span,
    openai_exception_handler,
    otel_request_validation_exception_handler,
    otel_unhandled_exception_handler,
)

from .conftest import normalize


def _make_request(parent_otel_span=None):
    state = SimpleNamespace(parent_otel_span=parent_otel_span)
    return SimpleNamespace(state=state)


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
    errors = [{"loc": ["body", "model"], "msg": "field required", "type": "missing"}]
    exc = RequestValidationError(errors)
    request = _make_request()

    response = await otel_request_validation_exception_handler(request=request, exc=exc)
    body = json.loads(response.body)

    assert response.status_code == 422
    assert normalize(body) == {"detail": exc.errors()}


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
        await otel_unhandled_exception_handler(
            request=request, exc=HTTPException(status_code=418, detail="teapot")
        )
