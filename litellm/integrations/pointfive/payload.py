"""Turns buffered log records into the gzipped NDJSON objects that get uploaded."""

import gzip
from collections.abc import Iterator, Mapping, Sequence
from itertools import accumulate, groupby, islice
from typing import Final

from litellm.litellm_core_utils.asyncify import asyncify
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

_NEWLINE_BYTES: Final = 1


def serialize_records(records: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Serialize each record to one JSON line."""
    return tuple(safe_dumps(record) for record in records)


def _encoded_size(line: str) -> int:
    return len(line.encode("utf-8")) + _NEWLINE_BYTES


def _object_indices(sizes: Sequence[int], max_bytes: int) -> Iterator[int]:
    """Number each line with the object it belongs to, opening a new one on overflow."""

    def advance(state: tuple[int, int], size: int) -> tuple[int, int]:
        index, used = state
        return (index + 1, size) if used and used + size > max_bytes else (index, used + size)

    return (index for index, _ in islice(accumulate(sizes, advance, initial=(0, 0)), 1, None))


def chunk_lines(lines: Sequence[str], max_bytes: int) -> tuple[tuple[str, ...], ...]:
    """
    Group serialized lines into objects of at most ``max_bytes`` uncompressed.

    A line above the bound on its own still becomes its own object. A record cannot be
    split, and holding it back would stall every record queued behind it.
    """
    sizes: Final = tuple(_encoded_size(line) for line in lines)
    numbered: Final = zip(_object_indices(sizes, max_bytes), lines, strict=True)
    return tuple(tuple(line for _, line in group) for _, group in groupby(numbered, lambda pair: pair[0]))


async def encode_lines(lines: Sequence[str]) -> bytes:
    """
    Join lines as NDJSON and gzip them off the event loop.

    An object can be several megabytes, and compressing that inline would block the
    proxy for as long as it takes.
    """
    compress: Final = asyncify(gzip.compress)
    return await compress("\n".join(lines).encode("utf-8"))
