"""Split a spend-log flush into statements the Prisma query engine can afford.

The query engine is a separate Rust process whose resident memory is a
high-water mark: it grows with the payload of the largest single statement it
is asked to execute and glibc never returns that memory to the OS, so a pod's
memory floor ratchets up to its worst-ever write and stays there for the life
of the worker. Under ``store_prompts_in_spend_logs`` a single spend-log row
carries the full prompt and response, so a fixed 1000-row ``create_many``
hands the engine tens of megabytes in one statement and permanently costs
hundreds of megabytes of RSS, which is what makes memory-based autoscaling
read the wrong number.

Bounding each statement by payload size instead caps that floor. Row-count
batching alone cannot: the same 1000 rows range from well under a megabyte
(spend counters only) to tens of megabytes (prompts stored), and only the
byte budget tracks what the engine actually allocates.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

SpendLogRow = Mapping[str, object]

_STATEMENT_FRAMING_BYTES: Final = len(json.dumps([]))
_ROW_SEPARATOR_BYTES: Final = len(json.dumps([0, 0])) - len(json.dumps([0])) - len(json.dumps(0))


def _row_payload_bytes(row: SpendLogRow) -> int:
    """Bytes this row contributes to the encoded write statement.

    The whole row is serialized rather than its values summed, so the count
    includes the field names, separators and braces the row carries on the
    wire and not only its payload. Those are what make the difference between
    a measurement and an estimate for a row of many small columns, where the
    keys outweigh the values.

    Serializing is also what makes the count a byte count. Character counts
    under-measure a prompt in a non-Latin script by its bytes-per-character
    factor, and even an all-ASCII prompt grows when JSON escapes its quotes,
    backslashes and newlines (about 18% for a realistic stored prompt, and up
    to double for escape-dense content). ``json.dumps`` escapes non-ASCII to
    ``\\uXXXX`` and defaults to ASCII output, so its length never under-states
    the wire size. ``default=str`` covers the datetimes and other scalars a
    row carries.

    A row the serializer refuses (a self-reference is the reachable case)
    counts as zero rather than raising: measuring a row must never be what
    loses spend data, since raising here would propagate out of the flush and
    drop every row queued behind it. Such a row is still written, it just does
    not contribute to the budget.
    """
    try:
        return len(json.dumps(row, default=str))
    except (TypeError, ValueError):
        return 0


def spend_log_write_batches(
    rows: Sequence[SpendLogRow],
    max_bytes: int,
) -> Iterator[Sequence[SpendLogRow]]:
    """Yield consecutive slices of ``rows`` whose payload fits ``max_bytes``.

    What is measured is the encoded slice, not the sum of its rows: rows become
    one collection on the wire, so the brackets around them and the separator
    between each pair count too. Summing rows alone under-states a slice by one
    separator per row, which is negligible for prompt-carrying rows and is not
    for a slice of many small ones, where the budget would be exceeded by the
    row count. The two framing constants are derived from the serializer rather
    than written down so they cannot drift from it.

    Slices preserve input order and together cover every row exactly once. A
    row larger than ``max_bytes`` on its own is yielded alone rather than
    dropped: the budget is a memory guardrail, not an admission filter, and
    losing spend data to protect RSS would be the worse failure.
    """
    sizes: Final = tuple(_row_payload_bytes(row) for row in rows)
    start = 0
    while start < len(rows):
        end = start + 1
        used = _STATEMENT_FRAMING_BYTES + sizes[start]
        while end < len(rows) and used + _ROW_SEPARATOR_BYTES + sizes[end] <= max_bytes:
            used += _ROW_SEPARATOR_BYTES + sizes[end]
            end += 1
        yield rows[start:end]
        start = end
