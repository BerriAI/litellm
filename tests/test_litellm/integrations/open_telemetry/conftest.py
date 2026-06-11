"""
Shared fixtures for the LIT-3193 OTEL HTTP-attribute matrix.

The matrix needs every error response — across unified inference, passthrough,
and admin endpoints — to carry ``http.response.status_code``, ``url.path``,
``http.route``, and a non-zero duration on the SERVER (root) span. These
fixtures hook a real ``OpenTelemetry`` callback into ``litellm.callbacks`` so
the tests drive the actual handler / wrapper code under test, not the OTEL
emitter in isolation.

See ``LIT-3193_test_matrix.md`` (same directory) for the cell list.
"""

import os
import sys
from datetime import datetime
from typing import Optional, Tuple
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.integrations.opentelemetry import OpenTelemetry


# ---------------------------------------------------------------------------
# OTEL + exporter
# ---------------------------------------------------------------------------
@pytest.fixture
def otel_with_exporter() -> Tuple[OpenTelemetry, InMemorySpanExporter]:
    """Real OpenTelemetry callback with every span captured in-memory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    otel = OpenTelemetry()
    otel.tracer = provider.get_tracer("lit-3193-tests")
    otel.message_logging = True
    return otel, exporter


@pytest.fixture
def server_span_factory(otel_with_exporter):
    """Factory mirroring user_api_key_auth: SERVER span + url.path + http.route."""
    otel, _exporter = otel_with_exporter

    def _make(url_path: str, http_route: Optional[str] = None):
        span = otel.create_litellm_proxy_request_started_span(
            start_time=datetime.now(), headers={}
        )
        otel.set_proxy_request_route_attributes(
            span, url_path=url_path, http_route=http_route or url_path
        )
        return span

    return _make


@pytest.fixture
def user_api_key_dict_factory():
    """UserAPIKeyAuth-shaped mock; the only attr the failure hooks read is
    parent_otel_span (plus team_id/team_alias for stamping)."""

    def _make(parent_span):
        d = MagicMock()
        d.parent_otel_span = parent_span
        d.team_id = "team-lit-3193"
        d.team_alias = "lit-3193-team"
        d.request_route = None
        return d

    return _make


@pytest.fixture
def register_otel_callback(otel_with_exporter, monkeypatch):
    """Make ProxyLogging.post_call_failure_hook iterate our OTEL instance."""
    otel, _ = otel_with_exporter
    saved = list(litellm.callbacks)
    monkeypatch.setattr(litellm, "callbacks", [otel])
    yield otel
    litellm.callbacks = saved


# Helpers (assertions, exception factories) live in ``_helpers.py`` — pytest
# auto-discovers fixtures here but forbids ``from .conftest import …``.
