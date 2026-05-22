"""
Matrix test: http.response.status_code, url.path, and span duration must be
present on the SERVER (root) span for every endpoint × HTTP-outcome combination
listed in LIT-3191.

Affected paths (previously missing one or more of these attributes):
  A. 5xx on unified LLM endpoints  (/v1/chat/completions, /v1/messages)
  B. 4xx on pass-through endpoints
  C. 4xx / 5xx on admin/management API endpoints

Strategy
  Use the real OpenTelemetry callback with an in-memory span exporter so we
  can inspect the exported spans' attributes directly.  Each cell drives the
  exact same code path the proxy uses in production.
"""

import asyncio
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.opentelemetry import (
    HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE,
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    URL_PATH_ATTRIBUTE,
    OpenTelemetry,
)
from litellm.proxy._types import ManagementEndpointLoggingPayload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PATH = "/v1/chat/completions"


def _make_otel():
    """OTel callback wired to an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel = OpenTelemetry()
    otel.tracer = provider.get_tracer(__name__)
    return otel, exporter


def _server_span(otel, path: str = _PATH):
    """Create the SERVER span that user_api_key_auth opens per request."""
    span = otel.create_litellm_proxy_request_started_span(
        start_time=datetime.now(), headers={}
    )
    # Simulate the route attributes set by auth.
    otel.set_proxy_request_route_attributes(span, url_path=path, http_route=path)
    return span


def _user_api_key_dict(server_span):
    d = MagicMock()
    d.parent_otel_span = server_span
    d.team_id = "t1"
    d.team_alias = "team-a"
    return d


def _finished(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


# Convenience exception classes with explicit HTTP status codes.
class _Generic5xx(Exception):
    """Generic Python exception — no .code / .status_code attribute."""


class _LLMProvider5xx(Exception):
    """Exception with .status_code (how LiteLLM wraps provider errors)."""

    status_code = 503


class _Client4xx(Exception):
    """Auth / validation 4xx."""

    status_code = 401


class _ProxyException4xx(Exception):
    """ProxyException-style error carries .code, not .status_code."""

    code = 403


# ---------------------------------------------------------------------------
# A. Unified LLM endpoint failures (5xx)
# ---------------------------------------------------------------------------


class TestUnifiedEndpoint5xx(unittest.TestCase):
    """
    5xx on /v1/chat/completions and /v1/messages must stamp
    http.response.status_code on the SERVER span.
    """

    def _run(self, exc):
        otel, exporter = _make_otel()
        server_span = _server_span(otel)
        asyncio.run(
            otel.async_post_call_failure_hook(
                request_data={},
                original_exception=exc,
                user_api_key_dict=_user_api_key_dict(server_span),
                traceback_str="tb",
            )
        )
        return _finished(exporter)

    def test_generic_exception_gets_500(self):
        """A bare Python exception (no HTTP code) must map to 500."""
        spans = self._run(_Generic5xx("something exploded"))
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server, "SERVER span not exported")
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(
            status_code,
            500,
            f"Expected 500 for generic exception, got {status_code!r}",
        )

    def test_llm_provider_5xx_preserves_status_code(self):
        """An exception with .status_code=503 must propagate 503."""
        spans = self._run(_LLMProvider5xx("provider timeout"))
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server, "SERVER span not exported")
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(
            status_code,
            503,
            f"Expected 503, got {status_code!r}",
        )

    def test_server_span_is_ended_after_5xx(self):
        """The SERVER span must be exported (ended) so duration is recorded."""
        spans = self._run(_Generic5xx("boom"))
        self.assertIn(
            LITELLM_PROXY_REQUEST_SPAN_NAME,
            spans,
            "SERVER span must be ended (exported) after a 5xx",
        )

    def test_server_span_has_path_after_5xx(self):
        """url.path set during auth must survive the failure hook."""
        spans = self._run(_Generic5xx("boom"))
        server = spans[LITELLM_PROXY_REQUEST_SPAN_NAME]
        self.assertEqual(server.attributes.get(URL_PATH_ATTRIBUTE), _PATH)


# ---------------------------------------------------------------------------
# B. Pass-through endpoint 4xx failures
# ---------------------------------------------------------------------------


class TestPassThroughEndpoint4xx(unittest.TestCase):
    """
    4xx on pass-through endpoints uses the same async_post_call_failure_hook
    path.  Verified separately to pin the behaviour documented in LIT-3191.
    """

    def _run(self, exc):
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path="/openai/v1/chat/completions")
        asyncio.run(
            otel.async_post_call_failure_hook(
                request_data={},
                original_exception=exc,
                user_api_key_dict=_user_api_key_dict(server_span),
                traceback_str="tb",
            )
        )
        return _finished(exporter)

    def test_http_exception_4xx_status_code_stamped(self):
        """HTTPException-style 4xx from an upstream must be stamped correctly."""
        spans = self._run(_Client4xx("unauthorized"))
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server, "SERVER span not exported")
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(
            status_code,
            401,
            f"Expected 401, got {status_code!r}",
        )

    def test_proxy_exception_code_4xx_status_code_stamped(self):
        """ProxyException carries .code; verify it maps to the span attribute."""
        spans = self._run(_ProxyException4xx("forbidden"))
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server)
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(status_code, 403, f"Expected 403, got {status_code!r}")

    def test_generic_exception_falls_back_to_500(self):
        """A generic exception from the upstream proxy must still get 500."""
        spans = self._run(_Generic5xx("upstream unreachable"))
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server)
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(
            status_code, 500, f"Expected fallback 500, got {status_code!r}"
        )

    def test_server_span_has_path(self):
        """url.path must be stamped regardless of 4xx vs 5xx."""
        spans = self._run(_Client4xx("bad key"))
        server = spans[LITELLM_PROXY_REQUEST_SPAN_NAME]
        self.assertEqual(
            server.attributes.get(URL_PATH_ATTRIBUTE),
            "/openai/v1/chat/completions",
        )


# ---------------------------------------------------------------------------
# C. Admin / management API endpoints
# ---------------------------------------------------------------------------


class TestAdminEndpointStatusCodeAndDuration(unittest.TestCase):
    """
    Admin endpoints use the dedicated management hooks, not the LLM callback.
    The parent SERVER span must be stamped with status code and properly ended
    (so duration is captured).
    """

    _MGMT_PATH = "/key/generate"

    def _payload(self, exc=None, response=None):
        return ManagementEndpointLoggingPayload(
            route=self._MGMT_PATH,
            request_data={"key": "sk-test"},
            response=response,
            exception=exc,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    # -- success --

    def test_admin_2xx_status_code_on_server_span(self):
        """Successful admin request must stamp http.response.status_code=200."""
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path=self._MGMT_PATH)
        asyncio.run(
            otel.async_management_endpoint_success_hook(
                logging_payload=self._payload(response={"key": "sk-new"}),
                parent_otel_span=server_span,
            )
        )
        spans = _finished(exporter)
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server, "SERVER span not exported")
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(status_code, 200, f"Expected 200, got {status_code!r}")

    def test_admin_2xx_server_span_is_ended(self):
        """SERVER span must be exported (ended) after a successful admin call."""
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path=self._MGMT_PATH)
        asyncio.run(
            otel.async_management_endpoint_success_hook(
                logging_payload=self._payload(response={"key": "sk-new"}),
                parent_otel_span=server_span,
            )
        )
        spans = _finished(exporter)
        self.assertIn(
            LITELLM_PROXY_REQUEST_SPAN_NAME,
            spans,
            "SERVER span was never ended — duration cannot be recorded",
        )

    def test_admin_2xx_server_span_has_path(self):
        """url.path (set during auth) must survive the success hook."""
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path=self._MGMT_PATH)
        asyncio.run(
            otel.async_management_endpoint_success_hook(
                logging_payload=self._payload(response={}),
                parent_otel_span=server_span,
            )
        )
        spans = _finished(exporter)
        server = spans[LITELLM_PROXY_REQUEST_SPAN_NAME]
        self.assertEqual(server.attributes.get(URL_PATH_ATTRIBUTE), self._MGMT_PATH)

    # -- failure: 4xx --

    def test_admin_4xx_status_code_on_server_span(self):
        """Admin 4xx must stamp http.response.status_code from the exception."""
        exc = _Client4xx("invalid key format")
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path=self._MGMT_PATH)
        asyncio.run(
            otel.async_management_endpoint_failure_hook(
                logging_payload=self._payload(exc=exc),
                parent_otel_span=server_span,
            )
        )
        spans = _finished(exporter)
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server, "SERVER span not exported")
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(status_code, 401, f"Expected 401, got {status_code!r}")

    def test_admin_4xx_server_span_is_ended(self):
        """SERVER span must be exported (ended) after admin 4xx."""
        exc = _Client4xx("bad token")
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path=self._MGMT_PATH)
        asyncio.run(
            otel.async_management_endpoint_failure_hook(
                logging_payload=self._payload(exc=exc),
                parent_otel_span=server_span,
            )
        )
        spans = _finished(exporter)
        self.assertIn(
            LITELLM_PROXY_REQUEST_SPAN_NAME,
            spans,
            "SERVER span was never ended — duration cannot be recorded",
        )

    def test_admin_4xx_server_span_has_path(self):
        """url.path must be preserved on the SERVER span after admin 4xx."""
        exc = _Client4xx("bad token")
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path=self._MGMT_PATH)
        asyncio.run(
            otel.async_management_endpoint_failure_hook(
                logging_payload=self._payload(exc=exc),
                parent_otel_span=server_span,
            )
        )
        spans = _finished(exporter)
        server = spans[LITELLM_PROXY_REQUEST_SPAN_NAME]
        self.assertEqual(server.attributes.get(URL_PATH_ATTRIBUTE), self._MGMT_PATH)

    # -- failure: 5xx --

    def test_admin_5xx_status_code_on_server_span(self):
        """Admin 5xx (DB error) must stamp 500 on the SERVER span."""

        class _DbError(Exception):
            status_code = 500

        exc = _DbError("db connection refused")
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path=self._MGMT_PATH)
        asyncio.run(
            otel.async_management_endpoint_failure_hook(
                logging_payload=self._payload(exc=exc),
                parent_otel_span=server_span,
            )
        )
        spans = _finished(exporter)
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server)
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(status_code, 500, f"Expected 500, got {status_code!r}")

    def test_admin_generic_exception_falls_back_to_500(self):
        """A bare Python exception from an admin handler must map to 500."""
        exc = _Generic5xx("unexpected DB error")
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path=self._MGMT_PATH)
        asyncio.run(
            otel.async_management_endpoint_failure_hook(
                logging_payload=self._payload(exc=exc),
                parent_otel_span=server_span,
            )
        )
        spans = _finished(exporter)
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server)
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(
            status_code, 500, f"Expected fallback 500, got {status_code!r}"
        )

    def test_admin_proxy_exception_code_4xx(self):
        """ProxyException-style .code attribute must be respected."""
        exc = _ProxyException4xx("team not found")
        otel, exporter = _make_otel()
        server_span = _server_span(otel, path=self._MGMT_PATH)
        asyncio.run(
            otel.async_management_endpoint_failure_hook(
                logging_payload=self._payload(exc=exc),
                parent_otel_span=server_span,
            )
        )
        spans = _finished(exporter)
        server = spans.get(LITELLM_PROXY_REQUEST_SPAN_NAME)
        self.assertIsNotNone(server)
        status_code = server.attributes.get(HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE)
        self.assertEqual(status_code, 403, f"Expected 403, got {status_code!r}")

    def test_admin_no_parent_span_does_not_raise(self):
        """When parent_otel_span is None (OTEL disabled) the hook must be a no-op."""
        exc = _Client4xx("bad key")
        otel, _ = _make_otel()
        # Must not raise
        asyncio.run(
            otel.async_management_endpoint_failure_hook(
                logging_payload=self._payload(exc=exc),
                parent_otel_span=None,
            )
        )

    def test_admin_success_no_parent_span_does_not_raise(self):
        """When parent_otel_span is None (OTEL disabled) success hook is a no-op."""
        otel, _ = _make_otel()
        asyncio.run(
            otel.async_management_endpoint_success_hook(
                logging_payload=self._payload(response={}),
                parent_otel_span=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
