"""
Tests for the OCI SSE event splitter.

Regression coverage for the streaming bug John Lathouwers reported: the old
``split_chunks`` helper split each individual HTTP read on ``\\n\\n``, so any
event that straddled a read boundary or any pair of events separated by a
single ``\\n`` would yield malformed chunks to ``OCIStreamWrapper.chunk_creator``
and crash ``json.loads``.
"""

import asyncio
from typing import AsyncIterator, Iterator, List

from litellm.llms.oci.chat.transformation import (
    _aiter_sse_events,
    _iter_sse_events,
)


def _collect_sync(stream: Iterator[str]) -> List[str]:
    return list(_iter_sse_events(iter(stream)))


def _collect_async(chunks: List[str]) -> List[str]:
    async def _src() -> AsyncIterator[str]:
        for c in chunks:
            yield c

    async def _run() -> List[str]:
        out: List[str] = []
        async for line in _aiter_sse_events(_src()):
            out.append(line)
        return out

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Sync splitter
# ---------------------------------------------------------------------------


class TestIterSseEventsSync:
    def test_well_formed_double_newline_separators(self):
        reads = ['data: {"a":1}\n\ndata: {"a":2}\n\n']
        assert _collect_sync(reads) == ['data: {"a":1}', 'data: {"a":2}']

    def test_event_split_across_two_reads(self):
        # The bug: read 1 ends mid-JSON, read 2 finishes it. Old code would
        # have yielded a truncated 'data: {"index":0,"text":"hel' and crashed
        # json.loads in chunk_creator.
        reads = [
            'data: {"index":0,"text":"hel',
            'lo"}\n\n',
        ]
        assert _collect_sync(reads) == ['data: {"index":0,"text":"hello"}']

    def test_event_split_into_many_tiny_reads(self):
        full = 'data: {"k":"value with spaces"}\n\n'
        reads = [full[i : i + 3] for i in range(0, len(full), 3)]
        assert _collect_sync(reads) == ['data: {"k":"value with spaces"}']

    def test_single_newline_separator(self):
        # The other shape John saw: events separated by just '\n'.
        reads = ['data: {"a":1}\ndata: {"a":2}\ndata: {"a":3}\n']
        assert _collect_sync(reads) == [
            'data: {"a":1}',
            'data: {"a":2}',
            'data: {"a":3}',
        ]

    def test_mixed_separators_in_one_read(self):
        reads = ['data: {"a":1}\ndata: {"a":2}\n\ndata: {"a":3}\n\n']
        assert _collect_sync(reads) == [
            'data: {"a":1}',
            'data: {"a":2}',
            'data: {"a":3}',
        ]

    def test_keepalive_and_comment_lines_dropped(self):
        # SSE keepalives ("\n") and comment lines (": ping") must not be
        # forwarded to chunk_creator, which would reject anything not
        # starting with 'data:'.
        reads = [
            "\n",
            ": ping\n",
            'data: {"a":1}\n\n',
            "\n",
            ": keepalive\n\n",
            'data: {"a":2}\n\n',
        ]
        assert _collect_sync(reads) == ['data: {"a":1}', 'data: {"a":2}']

    def test_trailing_partial_event_flushed_at_eof(self):
        # Final event arrives without a terminating newline. The splitter
        # must still emit it once the upstream iterator is exhausted.
        reads = ['data: {"a":1}\n\n', 'data: {"a":2}']
        assert _collect_sync(reads) == ['data: {"a":1}', 'data: {"a":2}']

    def test_trailing_non_data_line_dropped_at_eof(self):
        reads = ['data: {"a":1}\n\n: trailing-comment']
        assert _collect_sync(reads) == ['data: {"a":1}']

    def test_empty_stream(self):
        assert _collect_sync([]) == []

    def test_only_whitespace_and_keepalives(self):
        assert _collect_sync(["\n", "\n\n", ": ping\n"]) == []

    def test_boundary_between_data_keyword_and_payload(self):
        # The 'data:' marker itself straddles a read boundary.
        reads = ["dat", 'a: {"a":1}\n\n']
        assert _collect_sync(reads) == ['data: {"a":1}']

    def test_carriage_return_in_payload_preserved(self):
        # SSE-over-the-wire may use \r\n line endings. We split on \n; the
        # \r ends up on the previous line and strip() removes it.
        reads = ['data: {"a":1}\r\ndata: {"a":2}\r\n']
        assert _collect_sync(reads) == ['data: {"a":1}', 'data: {"a":2}']


# ---------------------------------------------------------------------------
# Async splitter — same scenarios, parallel coverage
# ---------------------------------------------------------------------------


class TestIterSseEventsAsync:
    def test_well_formed_double_newline_separators(self):
        assert _collect_async(['data: {"a":1}\n\ndata: {"a":2}\n\n']) == [
            'data: {"a":1}',
            'data: {"a":2}',
        ]

    def test_event_split_across_two_reads(self):
        assert _collect_async(['data: {"index":0,"text":"hel', 'lo"}\n\n']) == [
            'data: {"index":0,"text":"hello"}'
        ]

    def test_event_split_into_many_tiny_reads(self):
        full = 'data: {"k":"value with spaces"}\n\n'
        reads = [full[i : i + 3] for i in range(0, len(full), 3)]
        assert _collect_async(reads) == ['data: {"k":"value with spaces"}']

    def test_single_newline_separator(self):
        assert _collect_async(['data: {"a":1}\ndata: {"a":2}\ndata: {"a":3}\n']) == [
            'data: {"a":1}',
            'data: {"a":2}',
            'data: {"a":3}',
        ]

    def test_mixed_separators_in_one_read(self):
        assert _collect_async(
            ['data: {"a":1}\ndata: {"a":2}\n\ndata: {"a":3}\n\n']
        ) == ['data: {"a":1}', 'data: {"a":2}', 'data: {"a":3}']

    def test_keepalive_and_comment_lines_dropped(self):
        reads = [
            "\n",
            ": ping\n",
            'data: {"a":1}\n\n',
            "\n",
            ": keepalive\n\n",
            'data: {"a":2}\n\n',
        ]
        assert _collect_async(reads) == ['data: {"a":1}', 'data: {"a":2}']

    def test_trailing_partial_event_flushed_at_eof(self):
        assert _collect_async(['data: {"a":1}\n\n', 'data: {"a":2}']) == [
            'data: {"a":1}',
            'data: {"a":2}',
        ]

    def test_trailing_non_data_line_dropped_at_eof(self):
        assert _collect_async(['data: {"a":1}\n\n: trailing-comment']) == [
            'data: {"a":1}'
        ]

    def test_empty_stream(self):
        assert _collect_async([]) == []

    def test_only_whitespace_and_keepalives(self):
        assert _collect_async(["\n", "\n\n", ": ping\n"]) == []

    def test_boundary_between_data_keyword_and_payload(self):
        assert _collect_async(["dat", 'a: {"a":1}\n\n']) == ['data: {"a":1}']

    def test_carriage_return_in_payload_preserved(self):
        assert _collect_async(['data: {"a":1}\r\ndata: {"a":2}\r\n']) == [
            'data: {"a":1}',
            'data: {"a":2}',
        ]


# ---------------------------------------------------------------------------
# End-to-end: feed an awkwardly-chunked stream into OCIStreamWrapper and
# verify chunk_creator still parses each yielded line. This is the smoke
# test that proves the integration with the downstream consumer holds.
# ---------------------------------------------------------------------------


class TestSseSplitterFeedsChunkCreator:
    def test_split_event_parses_cleanly(self):
        # Build a realistic GENERIC OCI streaming payload, then chop it into
        # awkward reads. The splitter must reassemble exactly one event so
        # json.loads inside chunk_creator does not raise.
        import json
        from unittest.mock import MagicMock

        from litellm.llms.oci.chat.transformation import OCIStreamWrapper

        payload = {
            "apiFormat": "GENERIC",
            "message": {"content": [{"text": "hello"}]},
            "finishReason": None,
        }
        wire = f"data: {json.dumps(payload)}\n\n"
        # Split the wire string at an awkward point inside the JSON body.
        cut = wire.index('"hello"') + 3
        reads = [wire[:cut], wire[cut:]]

        # Drive the splitter directly and confirm we get exactly one event.
        events = list(_iter_sse_events(iter(reads)))
        assert len(events) == 1
        assert events[0].startswith("data: ")
        # chunk_creator should now parse this without raising.
        wrapper = OCIStreamWrapper(
            completion_stream=MagicMock(),
            model="xai.grok-4",
            logging_obj=MagicMock(),
        )
        # Must not raise.
        wrapper.chunk_creator(events[0])
