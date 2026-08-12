"""Harness coverage for ProxyClient's spend-log poll contract.

No proxy needed and no ``e2e`` marker: this pins how the poller reports an
unreadable /spend/logs. The poller is what every spend, budget and logging
suite reads its assertions off, so conflating "the endpoint answered and there
are no rows" with "the endpoint never answered" mislabels a read-path outage as
missing spend, and the resulting failure message accuses the wrong subsystem
120s after the fact. The transport is injected, so nothing here talks HTTP or
monkeypatches anything.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from e2e_http import (
    AuthHeaders,
    BinaryStream,
    ProbeResult,
    Result,
    StreamingResponse,
    Success,
    UnknownApiError,
)
from models import SpendLogRow, SpendLogs
from proxy_client import ProxyClient


class _UnusedTransport:
    """Every Transport member the poll path does not touch.

    ProxyClient depends on the whole Transport Protocol, so a fake must satisfy
    all of it to type-check. Raising keeps the fake honest: a test that reaches
    one of these is exercising something it did not set up.
    """

    def post[R: BaseModel](self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]) -> Result[R]:
        raise NotImplementedError(path)

    def stream(self, path: str, *, headers: BaseModel, json: BaseModel) -> StreamingResponse:
        raise NotImplementedError(path)

    def stream_binary(
        self, path: str, *, headers: BaseModel, json: BaseModel, chunk_size: int = 8192
    ) -> BinaryStream:
        raise NotImplementedError(path)

    def send(
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        params: BaseModel | None = None,
        stream: bool = False,
    ) -> StreamingResponse:
        raise NotImplementedError(path)

    def delete[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        json: BaseModel,
        response_type: type[R],
        params: BaseModel | None = None,
    ) -> Result[R]:
        raise NotImplementedError(path)

    def patch[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        raise NotImplementedError(path)

    def put[R: BaseModel](
        self, path: str, *, headers: BaseModel, json: BaseModel, response_type: type[R]
    ) -> Result[R]:
        raise NotImplementedError(path)

    def probe(self, path: str, *, params: BaseModel) -> ProbeResult:
        raise NotImplementedError(path)

    def upload[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        form: BaseModel,
        filename: str,
        content: bytes,
        file_content_type: str = "application/jsonl",
        file_field: str = "file",
        params: BaseModel | None = None,
        response_type: type[R],
    ) -> Result[R]:
        raise NotImplementedError(path)

    def download(self, path: str, *, headers: BaseModel) -> StreamingResponse:
        raise NotImplementedError(path)

    def bearer(self, key: str) -> AuthHeaders:
        raise NotImplementedError(key)

    @property
    def master(self) -> AuthHeaders:
        return AuthHeaders(authorization="Bearer sk-fake-master")


@dataclass
class ScriptedReads(_UnusedTransport):
    """Serves one scripted /spend/logs outcome per GET, then repeats the last."""

    reads: Sequence[Result[SpendLogs]]
    calls: int = 0

    def get[R: BaseModel](
        self,
        path: str,
        *,
        headers: BaseModel,
        params: BaseModel,
        response_type: type[R],
        timeout: float | None = None,
    ) -> Result[R]:
        assert path == "/spend/logs", f"fake only serves /spend/logs, got {path}"
        outcome = self.reads[min(self.calls, len(self.reads) - 1)]
        self.calls += 1
        return outcome  # pyright: ignore[reportReturnType]  # scripted per-test as Result[SpendLogs]


def _row(request_id: str, spend: float) -> SpendLogRow:
    return SpendLogRow(request_id=request_id, spend=spend)


def _served(rows: Iterator[SpendLogRow] | list[SpendLogRow]) -> Success[SpendLogs]:
    return Success(status_code=200, data=SpendLogs(list(rows)))


def _client(reads: Sequence[Result[SpendLogs]]) -> tuple[ProxyClient, ScriptedReads]:
    transport = ScriptedReads(reads=reads)
    # Short budget with no sleep: the contract under test is which outcome the
    # window produces, not how long it waits.
    return ProxyClient(transport=transport, poll_timeout=0.05, poll_interval=0.0), transport


class TestSpendLogPollDistinguishesUnreadableFromEmpty:
    def test_rows_are_returned_once_the_predicate_holds(self) -> None:
        rows = [_row("r1", 0.01)]
        client, transport = _client([_served([]), _served(rows)])

        polled = client.poll_logs_for_key("sk-test", predicate=lambda rs: any((r.spend or 0) > 0 for r in rs))

        assert [r.request_id for r in polled] == ["r1"]
        assert transport.calls == 2, "poller must keep reading until the predicate holds"

    def test_a_window_of_only_failed_reads_raises_instead_of_reporting_no_rows(self) -> None:
        """The regression: 24 consecutive 500s used to come back as an empty list, so
        the caller asserted "no spend row was written" for a request whose row it had
        never been able to read."""
        client, _ = _client([UnknownApiError(status_code=500, body='{"error":"deadlock detected"}')])

        with pytest.raises(AssertionError) as failure:
            client.poll_logs_for_key("sk-test")

        message = str(failure.value)
        assert "never returned a successful read" in message
        assert "500" in message and "deadlock detected" in message, (
            f"the failure must carry the read that failed, not just its own verdict: {message}"
        )

    def test_a_readable_endpoint_with_no_rows_still_reports_no_rows(self) -> None:
        """The other side of the same contract: emptiness the proxy actually served is
        a real answer about the spend pipeline and must reach the caller unchanged."""
        client, _ = _client([_served([])])

        assert client.poll_logs_for_key("sk-test") == []

    def test_a_transient_read_failure_is_polled_through(self) -> None:
        rows = [_row("r1", 0.02)]
        client, transport = _client([UnknownApiError(status_code=500, body="blip"), _served(rows)])

        polled = client.poll_logs_for_key("sk-test")

        assert [r.request_id for r in polled] == ["r1"]
        assert transport.calls == 2, "one failed read must not end the poll"

    def test_request_id_polling_shares_the_contract(self) -> None:
        client, _ = _client([UnknownApiError(status_code=503, body="upstream unavailable")])

        with pytest.raises(AssertionError, match="never returned a successful read"):
            client.poll_logs_for_request_id("chatcmpl-abc")
