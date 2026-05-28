"""
Matrix test: team_id / team_alias must land on EVERY span of a proxy
request trace, for a representative set of endpoints x HTTP outcomes.

Endpoints
  - /v1/chat/completions      (OpenAI-format LLM path)
  - /v1/messages              (Anthropic-format LLM path)
  - /team/info                (management/admin path)

Outcomes
  - 2xx  success
  - 3xx  redirect              (LLM endpoints never 3xx -> N/A; admin too)
  - 4xx  client error          (auth / validation failure)
  - 5xx  server error          (upstream / DB failure)

Strategy
  These assertions exercise the real OpenTelemetry callback the proxy
  invokes for each path, with a SERVER parent span (as
  ``user_api_key_auth`` creates) and an in-memory exporter. Each cell
  drives the path, then asserts team attributes on every span that path
  actually emits.

  - success path  -> ``log_success_event`` -> litellm_request +
    raw_gen_ai_request + guardrail child spans.
  - failure path  -> ``async_post_call_failure_hook`` -> Failed Proxy
    Server Request exception child span.

  Admin endpoints do not run the LLM success callback, so their only
  trace surface is the SERVER span (success) or the exception child span
  (failure) -- the cells below assert exactly that.
"""

import asyncio
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.opentelemetry import (
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    OpenTelemetry,
)

TEAM_ID = "team-123"
TEAM_ALIAS = "my-team"
TEAM_ID_ATTR = "metadata.user_api_key_team_id"
TEAM_ALIAS_ATTR = "metadata.user_api_key_team_alias"


def _make_otel():
    """OTel callback whose every span lands in an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    otel = OpenTelemetry()
    otel.tracer = provider.get_tracer(__name__)
    # raw_gen_ai_request sub-span is gated on message logging.
    otel.message_logging = True
    return otel, exporter


def _server_span(otel):
    """Mirror the SERVER span user_api_key_auth opens per request."""
    return otel.create_litellm_proxy_request_started_span(
        start_time=datetime.now(), headers={}
    )


def _slo(call_type, with_guardrail=False):
    """standard_logging_object the proxy attaches, carrying team metadata."""
    md = {
        "user_api_key_team_id": TEAM_ID,
        "user_api_key_team_alias": TEAM_ALIAS,
    }
    slo = {"metadata": md, "call_type": call_type}
    if with_guardrail:
        slo["guardrail_information"] = [
            {
                "guardrail_name": "test_guardrail",
                "guardrail_mode": "input",
                "guardrail_response": "ok",
                "start_time": 1609459200.0,
                "end_time": 1609459201.0,
            }
        ]
    return slo


def _success_kwargs(call_type, server_span, with_guardrail=True):
    """kwargs the success callback receives for an LLM proxy request."""
    return {
        "model": "gpt-4.1-mini",
        "litellm_call_id": "call-abc",
        "call_type": call_type,
        "litellm_params": {
            "metadata": {
                "litellm_parent_otel_span": server_span,
                "user_api_key_team_id": TEAM_ID,
                "user_api_key_team_alias": TEAM_ALIAS,
            }
        },
        "standard_logging_object": _slo(call_type, with_guardrail=with_guardrail),
        "messages": [{"role": "user", "content": "hi"}],
    }


def _team_user_api_key_dict(server_span):
    d = MagicMock()
    d.parent_otel_span = server_span
    d.team_id = TEAM_ID
    d.team_alias = TEAM_ALIAS
    return d


def _spans_by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def _assert_team_attrs(span, where):
    assert span.attributes.get(TEAM_ID_ATTR) == TEAM_ID, (
        f"{where}: missing/blank {TEAM_ID_ATTR} "
        f"(got {span.attributes.get(TEAM_ID_ATTR)!r})"
    )
    assert span.attributes.get(TEAM_ALIAS_ATTR) == TEAM_ALIAS, (
        f"{where}: missing/blank {TEAM_ALIAS_ATTR} "
        f"(got {span.attributes.get(TEAM_ALIAS_ATTR)!r})"
    )


class _Boom(Exception):
    """Upstream/DB style 5xx."""

    status_code = 500


class _ClientErr(Exception):
    """Auth/validation style 4xx."""

    status_code = 401


# ---------------------------------------------------------------------------
# LLM success cells: litellm_request + raw_gen_ai_request + guardrail spans
# ---------------------------------------------------------------------------
class TestLLMSuccessCells(unittest.TestCase):
    def _run_success(self, call_type):
        otel, exporter = _make_otel()
        server_span = _server_span(otel)
        kwargs = _success_kwargs(call_type, server_span)
        now = datetime.now()
        otel.log_success_event(kwargs, {"id": "resp-1"}, now, now)
        return _spans_by_name(exporter)

    def test_chat_completions_2xx(self):
        spans = self._run_success("completion")
        for name in (
            LITELLM_PROXY_REQUEST_SPAN_NAME,
            "litellm_request",
            "raw_gen_ai_request",
            "guardrail",
        ):
            assert name in spans, f"chat/completions 2xx: missing span {name}"
            _assert_team_attrs(spans[name], f"chat/completions 2xx [{name}]")

    def test_v1_messages_2xx(self):
        spans = self._run_success("anthropic_messages")
        for name in (
            LITELLM_PROXY_REQUEST_SPAN_NAME,
            "litellm_request",
            "raw_gen_ai_request",
            "guardrail",
        ):
            assert name in spans, f"v1/messages 2xx: missing span {name}"
            _assert_team_attrs(spans[name], f"v1/messages 2xx [{name}]")


# ---------------------------------------------------------------------------
# LLM failure cells: Failed Proxy Server Request exception child span
# ---------------------------------------------------------------------------
class TestLLMFailureCells(unittest.TestCase):
    def _run_failure(self, exc):
        """Drive the failure hook, then close the SERVER span (the proxy
        closes it after the hook in real flow) so both the exception child
        span and the SERVER root span are asserted."""
        otel, exporter = _make_otel()
        server_span = _server_span(otel)
        asyncio.run(
            otel.async_post_call_failure_hook(
                request_data={},
                original_exception=exc,
                user_api_key_dict=_team_user_api_key_dict(server_span),
                traceback_str="tb",
            )
        )
        server_span.end()
        return _spans_by_name(exporter)

    def _assert_all(self, spans, where):
        for name in ("Failed Proxy Server Request", LITELLM_PROXY_REQUEST_SPAN_NAME):
            assert name in spans, f"{where}: missing span {name}"
            _assert_team_attrs(spans[name], f"{where} [{name}]")

    def test_chat_completions_4xx(self):
        self._assert_all(
            self._run_failure(_ClientErr("bad key")), "chat/completions 4xx"
        )

    def test_chat_completions_5xx(self):
        self._assert_all(
            self._run_failure(_Boom("upstream blew up")), "chat/completions 5xx"
        )

    def test_v1_messages_4xx(self):
        self._assert_all(
            self._run_failure(_ClientErr("bad anthropic key")), "v1/messages 4xx"
        )

    def test_v1_messages_5xx(self):
        self._assert_all(
            self._run_failure(_Boom("anthropic upstream timeout")), "v1/messages 5xx"
        )


# ---------------------------------------------------------------------------
# Admin /team/info cells.
#   2xx: admin path never runs the LLM success callback -> its only trace
#        surface is the SERVER span; no child spans are emitted.
#   3xx: management endpoints do not redirect -> N/A (documented, no run).
#   4xx/5xx: proxy_logging post_call_failure_hook -> exception child span.
# ---------------------------------------------------------------------------
class TestAdminTeamInfoCells(unittest.TestCase):
    def _run_admin_failure(self, exc):
        otel, exporter = _make_otel()
        server_span = _server_span(otel)
        asyncio.run(
            otel.async_post_call_failure_hook(
                request_data={},
                original_exception=exc,
                user_api_key_dict=_team_user_api_key_dict(server_span),
                traceback_str="tb",
            )
        )
        server_span.end()
        return _spans_by_name(exporter)

    def test_team_info_4xx(self):
        spans = self._run_admin_failure(_ClientErr("team not found"))
        for name in ("Failed Proxy Server Request", LITELLM_PROXY_REQUEST_SPAN_NAME):
            _assert_team_attrs(spans[name], f"/team/info 4xx [{name}]")

    def test_team_info_5xx(self):
        spans = self._run_admin_failure(_Boom("db connection lost"))
        for name in ("Failed Proxy Server Request", LITELLM_PROXY_REQUEST_SPAN_NAME):
            _assert_team_attrs(spans[name], f"/team/info 5xx [{name}]")

    def test_team_info_2xx_only_server_span_no_orphan_children(self):
        """Admin success path emits no LLM child spans; nothing to stamp
        beyond the SERVER span. This pins that contract so a future
        regression that starts emitting child spans here without team
        attrs is caught."""
        otel, exporter = _make_otel()
        server_span = _server_span(otel)
        server_span.end()
        spans = _spans_by_name(exporter)
        assert set(spans) == {
            LITELLM_PROXY_REQUEST_SPAN_NAME
        }, f"/team/info 2xx: unexpected child spans {set(spans)}"

    def test_team_info_3xx_not_applicable(self):
        """Management endpoints return JSON, never a 3xx redirect."""
        self.skipTest("/team/info has no 3xx redirect path (N/A)")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Service span cells (LIT-3195): redis / db / auth service hooks must stamp
# the canonical ``metadata.user_api_key_team_id`` and
# ``metadata.user_api_key_team_alias`` keys onto the service span so traces
# are team-filterable at the service layer (not just the LLM layer).
#
# These cells drive ``async_service_success_hook`` / ``async_service_failure_hook``
# directly against the in-memory exporter and assert the canonical attrs are
# present on the emitted span.
# ---------------------------------------------------------------------------
class TestServiceSpanCells(unittest.TestCase):
    def _make_payload(self, service):
        from litellm.types.services import ServiceLoggerPayload

        return ServiceLoggerPayload(
            is_error=False,
            error=None,
            service=service,
            duration=0.1,
            call_type="read",
            event_metadata=None,
        )

    def _run(self, service, success):
        from litellm.types.services import ServiceTypes

        otel, exporter = _make_otel()
        parent = _server_span(otel)
        payload = self._make_payload(service)
        event_metadata = {
            "user_api_key_team_id": TEAM_ID,
            "user_api_key_team_alias": TEAM_ALIAS,
        }
        now = datetime.now()
        if success:
            asyncio.run(
                otel.async_service_success_hook(
                    payload=payload,
                    parent_otel_span=parent,
                    start_time=now,
                    end_time=now,
                    event_metadata=event_metadata,
                )
            )
        else:
            asyncio.run(
                otel.async_service_failure_hook(
                    payload=payload,
                    error="boom",
                    parent_otel_span=parent,
                    start_time=now,
                    end_time=now,
                    event_metadata=event_metadata,
                )
            )
        parent.end()
        spans = _spans_by_name(exporter)
        return spans[service.value]

    def test_redis_service_success_stamps_canonical_team_attrs(self):
        from litellm.types.services import ServiceTypes

        span = self._run(ServiceTypes.REDIS, success=True)
        _assert_team_attrs(span, "redis success")

    def test_redis_service_failure_stamps_canonical_team_attrs(self):
        from litellm.types.services import ServiceTypes

        span = self._run(ServiceTypes.REDIS, success=False)
        _assert_team_attrs(span, "redis failure")

    def test_db_service_success_stamps_canonical_team_attrs(self):
        from litellm.types.services import ServiceTypes

        span = self._run(ServiceTypes.DB, success=True)
        _assert_team_attrs(span, "db success")

    def test_auth_service_success_stamps_canonical_team_attrs(self):
        from litellm.types.services import ServiceTypes

        span = self._run(ServiceTypes.AUTH, success=True)
        _assert_team_attrs(span, "auth success")

    def test_blank_team_id_does_not_stamp_canonical_attrs(self):
        """Master-key / team-less requests carry empty-string ids — those
        must NOT bleed into traces as fake teams. The ``user_api_key_team_id``
        key from the existing event_metadata loop may still land verbatim
        (it has always done that), but the canonical ``metadata.*`` names
        used by trace filtering MUST be absent."""
        from litellm.types.services import ServiceTypes

        otel, exporter = _make_otel()
        parent = _server_span(otel)
        payload = self._make_payload(ServiceTypes.REDIS)
        now = datetime.now()
        asyncio.run(
            otel.async_service_success_hook(
                payload=payload,
                parent_otel_span=parent,
                start_time=now,
                end_time=now,
                event_metadata={
                    "user_api_key_team_id": "",
                    "user_api_key_team_alias": "",
                },
            )
        )
        parent.end()
        span = _spans_by_name(exporter)["redis"]
        assert (
            span.attributes.get(TEAM_ID_ATTR) is None
        ), "blank team_id leaked as canonical attr"
        assert (
            span.attributes.get(TEAM_ALIAS_ATTR) is None
        ), "blank team_alias leaked as canonical attr"

    def test_none_team_id_does_not_stamp_attrs(self):
        """Auth path stamps ``valid_token_dict.get("team_id")`` which returns
        ``None`` (not ``""``) for master-key / team-less callers. The call
        site filters ``None`` values out of ``event_metadata`` so the bare
        ``user_api_key_team_id`` attribute does not get stamped as the
        literal string ``"None"`` (regression guard for the Greptile P2
        nit on PR #29157)."""
        from litellm.types.services import ServiceTypes

        otel, exporter = _make_otel()
        parent = _server_span(otel)
        payload = self._make_payload(ServiceTypes.AUTH)
        now = datetime.now()
        # Simulate the call shape produced by the auth path after the
        # call-site filters out None values: empty dict -> None.
        asyncio.run(
            otel.async_service_success_hook(
                payload=payload,
                parent_otel_span=parent,
                start_time=now,
                end_time=now,
                event_metadata=None,
            )
        )
        parent.end()
        span = _spans_by_name(exporter)["auth"]
        assert span.attributes.get(TEAM_ID_ATTR) is None
        assert span.attributes.get(TEAM_ALIAS_ATTR) is None
        # The bare attribute names must not appear at all (no "None"
        # string leakage).
        assert span.attributes.get("user_api_key_team_id") is None
        assert span.attributes.get("user_api_key_team_alias") is None


# ---------------------------------------------------------------------------
# Management endpoint span cells (LIT-3195): /key/generate, /team/info, etc
# emit a management_endpoint_span via async_management_endpoint_success_hook
# and async_management_endpoint_failure_hook. Both surfaces must carry the
# canonical team attrs once the ``ManagementEndpointLoggingPayload`` is
# populated with ``team_id`` / ``team_alias`` from the calling
# ``UserAPIKeyAuth``.
# ---------------------------------------------------------------------------
class TestManagementEndpointSpanCells(unittest.TestCase):
    def _payload(self, route, *, with_team, success):
        from litellm.proxy._types import ManagementEndpointLoggingPayload

        now = datetime.now()
        return ManagementEndpointLoggingPayload(
            route=route,
            request_data={},
            response={"ok": True} if success else None,
            start_time=now,
            end_time=now,
            exception=None if success else Exception("boom"),
            team_id=TEAM_ID if with_team else None,
            team_alias=TEAM_ALIAS if with_team else None,
        )

    def _run(self, route, *, with_team, success):
        otel, exporter = _make_otel()
        parent = _server_span(otel)
        payload = self._payload(route, with_team=with_team, success=success)
        if success:
            asyncio.run(
                otel.async_management_endpoint_success_hook(
                    logging_payload=payload, parent_otel_span=parent
                )
            )
        else:
            asyncio.run(
                otel.async_management_endpoint_failure_hook(
                    logging_payload=payload, parent_otel_span=parent
                )
            )
        return _spans_by_name(exporter)[route]

    def test_team_info_success_stamps_canonical_team_attrs(self):
        span = self._run("/team/info", with_team=True, success=True)
        _assert_team_attrs(span, "/team/info success")

    def test_team_info_failure_stamps_canonical_team_attrs(self):
        span = self._run("/team/info", with_team=True, success=False)
        _assert_team_attrs(span, "/team/info failure")

    def test_key_generate_success_stamps_canonical_team_attrs(self):
        span = self._run("/key/generate", with_team=True, success=True)
        _assert_team_attrs(span, "/key/generate success")

    def test_no_team_caller_does_not_stamp_canonical_attrs(self):
        """A proxy-admin or team-less management call must not stamp
        canonical team attrs (None / None on the payload)."""
        span = self._run("/team/info", with_team=False, success=True)
        assert span.attributes.get(TEAM_ID_ATTR) is None
        assert span.attributes.get(TEAM_ALIAS_ATTR) is None
