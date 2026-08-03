"""Payload-size batching for spend-log writes.

The Prisma query engine's resident memory is a high-water mark set by the
largest single statement it executes, so these tests pin that no batch
exceeds the byte budget while every row is still written exactly once.

Symbols pinned here:
  - ``spend_log_write_batches``
  - ``_row_payload_bytes``
"""

import json
from typing import Any, Dict, List

from litellm.proxy.db.spend_log_batching import (
    _row_payload_bytes,
    spend_log_write_batches,
)


def _row(request_id: str, blob_bytes: int = 0) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "spend": 0.01,
        "total_tokens": 10,
        "messages": json.dumps({"content": "x" * blob_bytes}) if blob_bytes else "{}",
    }


def test_rows_are_split_when_the_payload_exceeds_the_budget() -> None:
    rows = [_row(f"r{i}", blob_bytes=1000) for i in range(10)]
    # The encoded size of a three-row statement, so three rows fit and four do not.
    budget = len(json.dumps(rows[:3], default=str))
    batches = list(spend_log_write_batches(rows, max_bytes=budget))

    assert [len(batch) for batch in batches] == [3, 3, 3, 1]
    assert all(len(json.dumps(list(batch), default=str)) <= budget for batch in batches)


def test_every_row_is_written_exactly_once_and_in_order() -> None:
    rows = [_row(f"r{i}", blob_bytes=500) for i in range(37)]
    flattened: List[Any] = [row for batch in spend_log_write_batches(rows, max_bytes=1700) for row in batch]

    assert [row["request_id"] for row in flattened] == [row["request_id"] for row in rows]


def test_small_rows_stay_in_one_statement() -> None:
    rows = [_row(f"r{i}") for i in range(1000)]
    batches = list(spend_log_write_batches(rows, max_bytes=2_000_000))

    assert [len(batch) for batch in batches] == [1000]


def test_a_row_larger_than_the_budget_is_written_alone_not_dropped() -> None:
    rows = [_row("small"), _row("huge", blob_bytes=50_000), _row("small2")]
    batches = list(spend_log_write_batches(rows, max_bytes=1000))

    assert [[row["request_id"] for row in batch] for batch in batches] == [
        ["small"],
        ["huge"],
        ["small2"],
    ]


def test_field_names_and_separators_are_counted() -> None:
    """A spend-log row carries ~25 columns, so for rows of many small values
    the field names and separators outweigh the values themselves. Counting
    only the values would let such a statement run well past the budget."""
    row = {f"column_with_a_long_name_{i}": "v" for i in range(25)}
    value_bytes_only = sum(len(json.dumps(value)) for value in row.values())

    assert _row_payload_bytes(row) > 4 * value_bytes_only

    # Every row fits the budget counting values alone, and only three fit once
    # the keys are counted, so the split is what proves they are counted.
    budget = len(json.dumps([row] * 3, default=str))
    assert [len(batch) for batch in spend_log_write_batches([row] * 6, max_bytes=budget)] == [3, 3]


def test_an_unserializable_value_does_not_break_the_flush() -> None:
    """Measuring a row must never be what loses spend data. A value the
    serializer refuses (a self-referencing list is the reachable case) counts
    as zero and the row is still written, rather than raising out of the
    flush and dropping every row queued behind it."""
    circular: List[Any] = []
    circular.append(circular)
    row = {"request_id": "r", "messages": circular}

    assert _row_payload_bytes(row) == 0
    assert [[r["request_id"] for r in batch] for batch in spend_log_write_batches([row], max_bytes=10)] == [["r"]]


def test_every_statement_fits_the_budget_when_encoded_whole() -> None:
    """The budget bounds the statement, not the sum of its rows. Rows become
    one collection on the wire, so a slice also carries the brackets around it
    and a separator between each pair; summing rows alone runs a slice of many
    small rows over the budget by roughly its row count."""
    rows = [_row(f"r{i}", blob_bytes=20 * (i % 7)) for i in range(400)]
    # Exactly the encoded size of the first 40 rows, so a batcher that ignored
    # the framing would fit 40 of them and overshoot by the 39 separators.
    budget = len(json.dumps(rows[:40], default=str))

    batches = [list(batch) for batch in spend_log_write_batches(rows, max_bytes=budget)]
    encoded = [len(json.dumps(batch, default=str)) for batch in batches]

    assert len(batches) > 1
    assert max(len(batch) for batch in batches) > 1
    assert max(encoded) <= budget


def test_empty_input_yields_no_statements() -> None:
    assert list(spend_log_write_batches([], max_bytes=1000)) == []


def test_non_ascii_payloads_are_measured_in_bytes_not_characters() -> None:
    """A prompt in a non-Latin script encodes to several bytes per character,
    so counting characters would let a statement carry a multiple of the
    budget, which is the whole thing the budget exists to prevent."""
    row = {"request_id": "r", "messages": "你好" * 2000}
    characters = len(row["messages"])

    assert _row_payload_bytes(row) >= len(row["messages"].encode("utf-8"))

    budget = characters + 1000  # comfortably over the character count, under the encoded size
    assert [len(batch) for batch in spend_log_write_batches([row, row], max_bytes=budget)] == [1, 1]


def test_json_escaping_growth_is_counted() -> None:
    """An all-ASCII prompt still grows when JSON escapes its quotes and
    newlines, so a raw character count would let a statement exceed the
    budget by that expansion factor."""
    row = {"request_id": "r", "messages": '"quoted"\n' * 1000}
    characters = len(row["messages"])

    assert _row_payload_bytes(row) > characters

    # Both rows fit the budget when counted as raw characters, and do not once
    # the escaping is counted, so the split is what proves the escaping is measured.
    budget = 2 * characters + 200
    assert [len(batch) for batch in spend_log_write_batches([row, row], max_bytes=budget)] == [1, 1]


def test_unserialized_list_payloads_are_measured_not_ignored() -> None:
    """``jsonify_object`` only stringifies dicts, so a list-valued ``messages``
    reaches the batcher raw; counting it as zero would let the largest rows
    bypass the budget entirely."""
    row = {"request_id": "r", "messages": [{"content": "x" * 5000}]}

    assert _row_payload_bytes(row) > 5000
    assert [len(batch) for batch in spend_log_write_batches([row, row], max_bytes=5100)] == [1, 1]
