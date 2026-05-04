"""
Streaming buffers for passthrough spend/logging flush.

``RawBytesStreamBuffer`` retains socket chunks for non–AWS-binary streams.

``BedrockEventStreamBuffer`` buffers raw bytes in botocore ``EventStreamBuffer``;
``iter_chunks_for_logging`` parses and UTF-8-decodes on first consumption only (see class doc).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig


class StreamBuffer(ABC):
    """Collect streamed bytes until passthrough flush."""

    @abstractmethod
    def feed(self, chunk: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def should_flush(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def iter_chunks_for_logging(
        self, provider_config: BasePassthroughConfig
    ) -> Iterator[str]:
        """Yield inner payload strings for ``handle_logging_collected_chunks``.

        Passthrough flush calls this once per ``StreamBuffer`` lifetime. Implementations may
        return a consume-once iterator (draining internal parsers); callers must not iterate
        again without feeding new data unless the concrete buffer documents repeatable reads.
        """

    def all_chunks_for_logging(self, provider_config: BasePassthroughConfig) -> List[str]:
        """Materialize logging chunks (calls ``iter_chunks_for_logging`` once)."""
        return list(self.iter_chunks_for_logging(provider_config))


class RawBytesStreamBuffer(StreamBuffer):
    """Retain chunks as ``List[bytes]`` (SSE / plain streaming)."""

    __slots__ = ("_chunks",)

    def __init__(self) -> None:
        self._chunks: List[bytes] = []

    def feed(self, chunk: bytes) -> None:
        self._chunks.append(chunk)

    def should_flush(self) -> bool:
        return len(self._chunks) > 0

    def iter_chunks_for_logging(
        self, provider_config: BasePassthroughConfig
    ) -> Iterator[str]:
        """Yield logging lines; repeatable (``_chunks`` are not drained by iteration)."""
        yield from provider_config._convert_raw_bytes_to_str_lines(self._chunks)

    @property
    def raw_chunks(self) -> List[bytes]:
        """Read-only view for tests."""
        return self._chunks


class BedrockEventStreamBuffer(StreamBuffer):
    """Bedrock ``application/vnd.amazon.eventstream``.

    Iterating ``EventStreamBuffer`` consumes framed messages from botocore's internal buffer.
    ``iter_chunks_for_logging`` is therefore consume-once: a second pass yields nothing useful
    unless more bytes were fed after the first iteration.
    """

    __slots__ = ("_event_stream_buffer", "_received_any")

    def __init__(self) -> None:
        from botocore.eventstream import EventStreamBuffer

        self._event_stream_buffer = EventStreamBuffer()
        self._received_any = False

    def feed(self, chunk: bytes) -> None:
        self._received_any = True
        self._event_stream_buffer.add_data(chunk)

    def should_flush(self) -> bool:
        return self._received_any

    def iter_chunks_for_logging(
        self, provider_config: BasePassthroughConfig
    ) -> Iterator[str]:
        # Consume-once: botocore removes each parsed message from _event_stream_buffer._data.
        for event in self._event_stream_buffer:
            payload = event.payload
            if payload:
                yield payload.decode("utf-8")
