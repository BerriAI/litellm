"""Live e2e: Arize Phoenix trace delivery for the global ``arize_phoenix`` callback.

Registry cells:
- logging.arize_phoenix.success.logs_spend (messages)
- logging.arize_phoenix.stream.logs_spend  (messages)

The proxy ships OTLP spans to the compose ``phoenix`` service
(PHOENIX_COLLECTOR_HTTP_ENDPOINT); read-back is Phoenix's REST route
``/v1/projects/{project}/spans``. Every request must yield exactly one
``litellm_request`` generation span in Phoenix. ArizePhoenixLogger builds its
own TracerProvider precisely so it can coexist with other OTEL-based callbacks
(datadog, otel, arize) without either dropping or double-shipping traces, so
the burst test counts spans per request instead of just probing for presence.

The burst class simulates Claude Code traffic: consecutive Anthropic-format
``/v1/messages`` calls, mixing streaming and tool use, against a Claude model.

Known-red (registry fail_before_fix: proven): non-streaming /v1/messages
currently ships two complete traces per request. The @client async wrapper
enqueues async_success_handler AND executor-submits the sync success_handler;
the sync CustomLogger gate keys off _is_sync_litellm_request, whose flag list
(acompletion/aresponses/aembedding/...) has no marker for the async-only
anthropic_messages entrypoint, so log_success_event fires a second time.
Streaming escapes because only the async handler assembles the final stream.
test_burst_ships_exactly_one_trace_per_request stays strict so the fix has a
regression gate.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from e2e_config import POLL_INTERVAL, unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from logging_client import (
    CLAUDE_CODE_TOOLS,
    PHOENIX_GENERATION_SPAN,
    LoggingClient,
    PhoenixCreds,
    costs_agree,
    phoenix_span_blob,
)
from models import AnthropicMessagesResponse, AnthropicToolChoice, SpendLogRow, SpendLogsParams

pytestmark = pytest.mark.e2e

DRIVER_MODEL = "claude-haiku-4-5"
CLAUDE_CODE_BURST = 10


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fresh_key(client: LoggingClient, resources: ResourceManager) -> str:
    key = client.key_with_alias(
        f"e2e-phoenix-key-{unique_marker()}", models=[DRIVER_MODEL]
    )
    resources.defer(lambda: client.delete_key(key))
    return key


def _success_rows(rows: list[SpendLogRow]) -> list[SpendLogRow]:
    """Retries against a flaky upstream may add failure rows; those are not
    duplicates, so duplicate accounting only counts non-failure rows."""
    return [row for row in rows if (row.status or "success").lower() != "failure"]


class TestArizePhoenixLogging:
    """The arize_phoenix callback on Anthropic-format /v1/messages traffic."""

    @pytest.mark.covers("logging.arize_phoenix.success.logs_spend", exercised_on=["messages"])
    def test_messages_success_delivers_trace(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        phoenix_creds: PhoenixCreds,
    ) -> None:
        key = _fresh_key(client, resources)
        since = _utc_now_iso()
        marker = unique_marker()
        outcome = client.messages_raw(
            key, DRIVER_MODEL, f"reply with one word only {marker}"
        )
        require_successful_call(outcome)

        spans = client.poll_phoenix_spans(phoenix_creds, marker=marker, since=since)
        assert spans, (
            f"Phoenix never received a {PHOENIX_GENERATION_SPAN} span for "
            f"marker={marker!r} in project {phoenix_creds.project!r}"
        )

    @pytest.mark.covers("logging.arize_phoenix.success.logs_spend", exercised_on=["messages"])
    def test_messages_spend_log_flushed(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        phoenix_creds: PhoenixCreds,
    ) -> None:
        """The Phoenix trace and the proxy's own spend row must both land for one
        request, and their costs must agree with x-litellm-response-cost."""
        key = _fresh_key(client, resources)
        since = _utc_now_iso()
        marker = unique_marker()
        outcome = client.messages_raw(
            key, DRIVER_MODEL, f"reply with one word only {marker}"
        )
        require_successful_call(outcome)

        spans = client.poll_phoenix_spans(phoenix_creds, marker=marker, since=since)
        assert spans, f"Phoenix trace missing for marker={marker!r}"

        spend_row = client.poll_proxy_spend_for_key(key, require_positive_spend=True)
        assert (
            spend_row is not None
            and spend_row.spend is not None
            and spend_row.spend > 0
        ), f"/spend/logs never flushed a positive spend row for the key (marker={marker!r})"
        assert outcome.response_cost is not None and outcome.response_cost > 0, (
            f"proxy must return positive x-litellm-response-cost; got {outcome.response_cost!r}"
        )
        assert costs_agree(outcome.response_cost, spend_row.spend), (
            f"flushed spend {spend_row.spend!r} disagrees with "
            f"x-litellm-response-cost {outcome.response_cost!r}"
        )

    @pytest.mark.covers("logging.arize_phoenix.success.logs_spend", exercised_on=["messages"])
    def test_messages_tool_use_on_trace(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        phoenix_creds: PhoenixCreds,
    ) -> None:
        key = _fresh_key(client, resources)
        since = _utc_now_iso()
        marker = unique_marker()
        outcome = client.messages_raw(
            key,
            DRIVER_MODEL,
            f"Read the file /workspace/config-{marker}.yaml",
            tools=CLAUDE_CODE_TOOLS,
            tool_choice=AnthropicToolChoice(type="any"),
            max_tokens=256,
        )
        require_successful_call(outcome)
        parsed = AnthropicMessagesResponse.model_validate_json(outcome.body)
        tool_names = [
            block.name for block in parsed.content if block.type == "tool_use" and block.name
        ]
        assert tool_names, (
            f"tool_choice=any must force a tool_use block; content={outcome.body[:300]}"
        )

        spans = client.poll_phoenix_spans(phoenix_creds, marker=marker, since=since)
        assert spans, f"Phoenix trace missing for tool-use marker={marker!r}"
        assert any(tool_names[0] in phoenix_span_blob(span) for span in spans), (
            f"Phoenix span must record the {tool_names[0]!r} tool call; "
            f"attributes={phoenix_span_blob(spans[0])[:400]}"
        )


class TestClaudeCodeSimulation:
    """A Claude Code-style burst over /v1/messages: many consecutive calls with
    streaming and tool use mixed in, then exact per-request trace accounting."""

    @pytest.mark.covers("logging.arize_phoenix.stream.logs_spend", exercised_on=["messages"])
    def test_burst_ships_exactly_one_trace_per_request(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        phoenix_creds: PhoenixCreds,
    ) -> None:
        key = _fresh_key(client, resources)
        since = _utc_now_iso()
        run = unique_marker()
        markers = tuple(f"{run}-req{i}" for i in range(CLAUDE_CODE_BURST))

        outcomes: list[StreamingResponse] = []
        for i, marker in enumerate(markers):
            outcome = client.messages_raw(
                key,
                DRIVER_MODEL,
                f"Claude Code session step: reply with one word only {marker}",
                stream=i % 2 == 1,
                tools=CLAUDE_CODE_TOOLS if i % 3 == 0 else None,
                max_tokens=128,
            )
            require_successful_call(outcome)
            outcomes.append(outcome)
        assert sum(1 for o in outcomes if o.is_streaming) > 0, (
            "burst must exercise the streaming path"
        )

        _ = client.poll_phoenix_spans(
            phoenix_creds, marker=run, min_count=CLAUDE_CODE_BURST, since=since
        )
        time.sleep(POLL_INTERVAL)
        spans = [
            span
            for span in client.find_phoenix_spans(phoenix_creds, marker=run, since=since)
            if span.status_code == "OK"
        ]
        counts = {
            marker: sum(1 for span in spans if marker in phoenix_span_blob(span))
            for marker in markers
        }
        wrong = {marker: n for marker, n in counts.items() if n != 1}
        assert not wrong, (
            f"every request must ship exactly one {PHOENIX_GENERATION_SPAN} span to "
            f"Phoenix; off-by-count markers (duplicates > 1, dropped == 0): {wrong}"
        )
        assert len(spans) == CLAUDE_CODE_BURST, (
            f"{len(spans)} generation spans in Phoenix for {CLAUDE_CODE_BURST} requests; "
            f"extras are duplicate traces"
        )

    @pytest.mark.covers("logging.arize_phoenix.stream.logs_spend", exercised_on=["messages"])
    def test_burst_flushes_exactly_one_spend_row_per_request(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        phoenix_creds: PhoenixCreds,
    ) -> None:
        key = _fresh_key(client, resources)
        since = _utc_now_iso()
        run = unique_marker()
        markers = tuple(f"{run}-req{i}" for i in range(CLAUDE_CODE_BURST))
        for i, marker in enumerate(markers):
            require_successful_call(
                client.messages_raw(
                    key,
                    DRIVER_MODEL,
                    f"Claude Code session step: reply with one word only {marker}",
                    stream=i % 2 == 1,
                )
            )

        rows = client.gateway.poll_logs_for_key(
            key,
            min_rows=CLAUDE_CODE_BURST,
            predicate=lambda rs: len(_success_rows(rs)) >= CLAUDE_CODE_BURST,
        )
        time.sleep(POLL_INTERVAL)
        rows = _success_rows(client.gateway.spend_logs(SpendLogsParams(api_key=key)))
        assert len(rows) == CLAUDE_CODE_BURST, (
            f"{len(rows)} success spend rows flushed for {CLAUDE_CODE_BURST} requests; "
            f"extras mean the burst was double-billed, missing means spend was dropped"
        )
        assert all(row.spend is not None and row.spend > 0 for row in rows), (
            f"every flushed row must carry positive spend; "
            f"rows={[(r.request_id, r.spend) for r in rows]}"
        )

        spans = client.poll_phoenix_spans(
            phoenix_creds, marker=run, min_count=CLAUDE_CODE_BURST, since=since
        )
        assert len([s for s in spans if s.status_code == "OK"]) >= CLAUDE_CODE_BURST, (
            f"Phoenix must hold a trace for each of the {CLAUDE_CODE_BURST} spend rows"
        )
