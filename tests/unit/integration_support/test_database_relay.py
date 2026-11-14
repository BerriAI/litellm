from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.database_relay import StatementMarkerScanner, TriggerScanner

TRIGGER: Final = b'SELECT "startTime" FROM "LiteLLM_SpendLogs"'


@pytest.mark.parametrize("split_at", range(1, len(TRIGGER)))
def test_trigger_scanner_matches_a_trigger_split_across_two_reads(split_at: int) -> None:
    scanner: Final = TriggerScanner(TRIGGER)
    assert not scanner.feed(TRIGGER[:split_at])
    assert scanner.feed(TRIGGER[split_at:])


def test_trigger_scanner_matches_a_trigger_arriving_one_byte_at_a_time() -> None:
    scanner: Final = TriggerScanner(TRIGGER)
    hits: Final = tuple(scanner.feed(TRIGGER[i : i + 1]) for i in range(len(TRIGGER)))
    assert hits == (False,) * (len(TRIGGER) - 1) + (True,)


def test_trigger_scanner_reports_a_match_once() -> None:
    scanner: Final = TriggerScanner(TRIGGER)
    assert scanner.feed(b"x" + TRIGGER + b"y")
    assert not scanner.feed(b"z")


STARTUP: Final = b"\x00\x00\x00\x08\x00\x03\x00\x00"


def _message(kind: bytes, payload: bytes) -> bytes:
    return kind + (len(payload) + 4).to_bytes(4, "big") + payload


PARSE: Final = _message(b"P", b"s1\x00" + TRIGGER + b"\x00\x00\x00")
BIND: Final = _message(b"B", b"\x00s1\x00\x00\x00\x00\x00\x00\x00")


@pytest.mark.parametrize("split_at", range(1, len(STARTUP + PARSE + BIND)))
def test_statement_markers_survive_every_protocol_frame_split(split_at: int) -> None:
    scanner: Final = StatementMarkerScanner((TRIGGER,))
    stream: Final = STARTUP + PARSE + BIND
    assert (*scanner.feed(stream[:split_at]), *scanner.feed(stream[split_at:])) == (
        frozenset({TRIGGER}),
        frozenset({TRIGGER}),
    )


def test_statement_markers_are_local_to_a_connection_and_forget_closed_statements() -> None:
    scanner: Final = StatementMarkerScanner((TRIGGER,))
    other: Final = StatementMarkerScanner((TRIGGER,))
    assert tuple(scanner.feed(STARTUP + PARSE)) == (frozenset({TRIGGER}),)
    assert tuple(other.feed(STARTUP + BIND)) == (frozenset(),)
    assert tuple(scanner.feed(_message(b"C", b"Ss1\x00") + BIND)) == (frozenset(),)


def test_replaced_and_unnamed_statements_do_not_reuse_old_markers() -> None:
    scanner: Final = StatementMarkerScanner((TRIGGER,))
    assert tuple(scanner.feed(STARTUP + PARSE)) == (frozenset({TRIGGER}),)
    replacement: Final = _message(b"P", b"s1\x00SELECT 1\x00\x00\x00")
    unnamed: Final = _message(b"P", b"\x00" + TRIGGER + b"\x00\x00\x00")
    assert tuple(scanner.feed(replacement + BIND + unnamed)) == (frozenset(), frozenset(), frozenset({TRIGGER}))


def test_statement_scanner_rejects_invalid_frame_lengths() -> None:
    scanner: Final = StatementMarkerScanner((TRIGGER,))
    assert tuple(scanner.feed(STARTUP)) == ()
    with pytest.raises(ValueError, match="Invalid PostgreSQL message length"):
        tuple(scanner.feed(b"P\x00\x00\x00\x03"))


def test_statement_scanner_matches_bound_parameter_markers() -> None:
    scanner: Final = StatementMarkerScanner((TRIGGER,))
    bind: Final = _message(
        b"B", b"\x00s2\x00\x00\x00\x00\x01" + len(TRIGGER).to_bytes(4, "big") + TRIGGER + b"\x00\x00"
    )
    assert tuple(scanner.feed(STARTUP + bind)) == (frozenset({TRIGGER}),)
