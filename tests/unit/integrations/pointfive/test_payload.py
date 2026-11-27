import gzip
import json

import pytest

from litellm.integrations.pointfive.payload import chunk_lines, encode_lines, serialize_records

UNBOUNDED = 10_000_000


def test_each_record_becomes_one_json_line():
    lines = serialize_records([{"id": "a"}, {"id": "b"}, {"id": "c"}])

    assert len(lines) == 3
    assert [json.loads(line)["id"] for line in lines] == ["a", "b", "c"]


def test_non_serializable_values_do_not_raise():
    """An odd payload must not kill the flush."""
    lines = serialize_records([{"id": "a", "when": object()}])

    assert json.loads(lines[0])["id"] == "a"


def test_records_that_fit_stay_in_one_object():
    lines = serialize_records([{"id": f"r{i}"} for i in range(50)])

    assert chunk_lines(lines, UNBOUNDED) == (lines,)


def test_objects_are_capped_by_uncompressed_size():
    lines = serialize_records([{"id": f"r{i}", "blob": "x" * 100} for i in range(10)])
    line_bytes = len(lines[0].encode("utf-8")) + 1

    chunks = chunk_lines(lines, line_bytes * 3)

    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 1]


def test_oversized_single_record_is_sent_alone_not_stalled():
    """A record too big for the cap must still go out, or it blocks everything behind it."""
    lines = serialize_records([{"id": "small"}, {"id": "huge", "blob": "x" * 5000}, {"id": "small2"}])

    chunks = chunk_lines(lines, 200)

    assert sum(len(chunk) for chunk in chunks) == 3
    huge = [chunk for chunk in chunks if any("huge" in line for line in chunk)]
    assert len(huge) == 1
    assert len(huge[0]) == 1


def test_no_records_produces_no_objects():
    assert chunk_lines((), UNBOUNDED) == ()


def test_every_record_appears_exactly_once():
    lines = serialize_records([{"id": f"r{i}"} for i in range(37)])

    chunks = chunk_lines(lines, len(lines[0]) * 4)

    assert [line for chunk in chunks for line in chunk] == list(lines)


@pytest.mark.asyncio
async def test_encode_lines_round_trips_through_gzip():
    lines = serialize_records([{"id": f"r{i}"} for i in range(5)])

    encoded = await encode_lines(lines)

    assert encoded[:2] == b"\x1f\x8b"
    assert gzip.decompress(encoded).decode("utf-8") == "\n".join(lines)


@pytest.mark.asyncio
async def test_encode_lines_compresses_repetitive_records():
    lines = serialize_records([{"id": f"r{i}", "model": "gpt-4o", "cost": 0.01} for i in range(200)])

    encoded = await encode_lines(lines)

    assert len(encoded) < len(gzip.decompress(encoded)) / 2
