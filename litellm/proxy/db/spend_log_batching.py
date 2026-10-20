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

Bounding each statement caps that floor, and it takes two budgets because the
engine charges for both terms. A byte budget is what tracks a prompt-carrying
row, whose size swings by orders of magnitude, and a row budget is what tracks
the engine's per-row bookkeeping, which a byte budget cannot see: rows holding
attribution metadata only stay far under any useful byte budget, so it never
binds and every statement runs at the caller's row cap. Measured on such a
flush, the same 100,000 rows cost 151 MB of permanently resident engine RSS at
1000 rows per statement against 25 MB at 100, with no statement anywhere near
a 2 MB byte budget.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from itertools import accumulate
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


def spend_log_row_bytes(row: SpendLogRow) -> int:
    """Bytes this row costs, measured the same way the write budget measures it."""
    return _row_payload_bytes(row)


def spend_log_queue_within_budget(
    rows: Sequence[SpendLogRow],
    queued_bytes: int,
    max_bytes: int,
) -> tuple[Sequence[SpendLogRow], int]:
    """Drop the oldest rows until the queue costs at most ``max_bytes``.

    Returns the rows to keep and what they cost, so a caller tracking the total
    across calls does not have to re-measure the rows it kept. ``queued_bytes``
    is that running total for ``rows``; only the rows actually dropped are
    measured here, which is what keeps an append off an O(queue) path.

    A queue is bounded by bytes rather than by row count because a row's size
    swings by orders of magnitude with ``store_prompts_in_spend_logs``, so any
    row cap generous enough to ride out an outage of counter-only rows is an
    OOM once prompts are stored.

    The newest row is kept whatever it costs, for the same reason a statement
    over budget is still written: the budget is a memory guardrail, not an
    admission filter, and losing spend data to protect RSS is the worse failure.
    """
    if queued_bytes <= max_bytes or len(rows) <= 1:
        return rows, queued_bytes
    droppable: Final = rows[:-1]
    remaining_by_drops: Final = (
        queued_bytes - freed for freed in accumulate(_row_payload_bytes(row) for row in droppable)
    )
    fits: Final = next(
        ((drops, remaining) for drops, remaining in enumerate(remaining_by_drops, start=1) if remaining <= max_bytes),
        (len(droppable), _row_payload_bytes(rows[-1])),
    )
    return rows[fits[0] :], fits[1]


def spend_log_write_batches(
    rows: Sequence[SpendLogRow],
    max_bytes: int,
    max_rows: int,
) -> Iterator[Sequence[SpendLogRow]]:
    """Yield consecutive slices of ``rows`` within both ``max_bytes`` and ``max_rows``.

    What is measured for the byte budget is the encoded slice, not the sum of
    its rows: rows become one collection on the wire, so the brackets around
    them and the separator between each pair count too. Summing rows alone
    under-states a slice by one separator per row, which is negligible for
    prompt-carrying rows and is not for a slice of many small ones, where the
    budget would be exceeded by the row count. The two framing constants are
    derived from the serializer rather than written down so they cannot drift
    from it.

    Both budgets are needed because the engine's cost has two terms. Payload
    bytes dominate when prompts are stored, and per-row bookkeeping dominates
    when they are not: a slice of narrow rows costs the engine far more than
    its bytes suggest, so a byte budget alone never binds on a deployment whose
    rows carry no prompts and every statement stays at the caller's row cap.
    Measured on a spend-log flush of rows carrying attribution metadata only,
    writing the same 100,000 rows at 1000 rows per statement left 151 MB of
    engine RSS resident against 25 MB at 100, with neither reaching a 2 MB byte
    budget.

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
        while end < len(rows) and end - start < max_rows and used + _ROW_SEPARATOR_BYTES + sizes[end] <= max_bytes:
            used += _ROW_SEPARATOR_BYTES + sizes[end]
            end += 1
        yield rows[start:end]
        start = end
