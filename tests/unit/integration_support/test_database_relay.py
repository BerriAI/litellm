from __future__ import annotations

from typing import Final

import pytest

from tests.integration._support.database_relay import TriggerScanner

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
