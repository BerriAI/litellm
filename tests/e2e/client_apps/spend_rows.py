"""The proxy-side half of every client_apps cell: the spend rows the proxy wrote
for the key, each a success with real spend and a request body the client sent
with `stream: true`. Every client here makes at least two calls per tool turn
(the tool call, then the reply), so the poll waits for two rows."""

from __future__ import annotations

from client_apps_models import RecordedRequest
from proxy_client import ProxyClient, SpendLogRow


def recorded_request(row: SpendLogRow) -> RecordedRequest:
    return RecordedRequest.model_validate(row.proxy_server_request)


def streamed_tool_turn_rows(proxy: ProxyClient, key: str) -> list[SpendLogRow]:
    rows = proxy.poll_logs_for_key(key, min_rows=2)
    assert len(rows) >= 2, rows
    assert all(row.status == "success" for row in rows), rows
    assert all(row.spend is not None and row.spend > 0 for row in rows), rows
    assert all(recorded_request(row).stream for row in rows), rows
    return rows
