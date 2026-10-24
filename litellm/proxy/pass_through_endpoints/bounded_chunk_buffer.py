from collections import deque
from collections.abc import Iterator, Sequence
from typing import Final, overload

_FRAME_SEPARATORS: Final = (b"\r\n\r\n", b"\n\n")


def _end_of_first_frame(data: bytes) -> int:
    """Index just past the first complete frame in ``data``; 0 when it holds none."""
    ends: Final = [i + len(sep) for sep in _FRAME_SEPARATORS if (i := data.find(sep)) != -1]
    if ends:
        return min(ends)
    newline: Final = data.find(b"\n")
    return 0 if newline == -1 else newline + 1


def _end_of_last_frame(data: bytes) -> int:
    """Index just past the last complete frame in ``data``; 0 when it holds none."""
    ends: Final = [i + len(sep) for sep in _FRAME_SEPARATORS if (i := data.rfind(sep)) != -1]
    if ends:
        return max(ends)
    newline: Final = data.rfind(b"\n")
    return 0 if newline == -1 else newline + 1


class BoundedChunkBuffer(Sequence[bytes]):
    """
    Collects the chunks of a streamed response for end-of-stream logging without letting
    an unbounded stream exhaust memory.

    Half of ``max_bytes`` keeps the start of the stream and half keeps the end, so the
    first event (input usage, model) and the terminal event (final usage) both survive.
    When the stream fits, every chunk is kept untouched. When it does not, the bytes in
    between are dropped and both ends are trimmed back to a frame boundary, so the
    retained bytes never contain a half-cut event.

    ``max_bytes=None`` keeps everything.
    """

    def __init__(self, max_bytes: int | None = None) -> None:
        self._head_limit: Final = None if max_bytes is None else max_bytes // 2
        self._tail_limit: Final = None if max_bytes is None else max_bytes - (max_bytes // 2)
        self._head: Final[list[bytes]] = []  # mutable-ok: append-only buffer owned by this class
        self._tail: Final[deque[bytes]] = deque()  # mutable-ok: append-only buffer owned by this class
        self._head_size = 0
        self._tail_size = 0
        self.total_bytes = 0
        self._head_closed = False

    @property
    def truncated(self) -> bool:
        return self.total_bytes > self._head_size + self._tail_size

    @property
    def retained_bytes(self) -> int:
        return sum(len(chunk) for chunk in self)

    def append(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if self._head_limit is None or self._tail_limit is None:
            self._head.append(chunk)
            self._head_size += len(chunk)
            return
        chunk_rest: Final = chunk if self._head_closed else self._fill_head(chunk, self._head_limit)
        if chunk_rest is None:
            return
        self._tail.append(chunk_rest[-self._tail_limit :] if self._tail_limit else b"")
        self._tail_size += len(self._tail[-1])
        while self._tail_size > self._tail_limit:
            overflow = self._tail_size - self._tail_limit
            oldest = self._tail.popleft()
            if len(oldest) > overflow:
                self._tail.appendleft(oldest[overflow:])
                self._tail_size -= overflow
                break
            self._tail_size -= len(oldest)

    def _fill_head(self, chunk: bytes, head_limit: int) -> bytes | None:
        """Keep what fits in the head; the rest of the chunk, or None when all of it fit."""
        room: Final = head_limit - self._head_size
        if len(chunk) <= room:
            self._head.append(chunk)
            self._head_size += len(chunk)
            return None
        if room > 0:
            self._head.append(chunk[:room])
            self._head_size += room
        self._head_closed = True
        return chunk[room:]

    def _frames(self) -> Sequence[bytes]:
        if not self.truncated:
            return [*self._head, *self._tail]
        head: Final = b"".join(self._head)
        tail: Final = b"".join(self._tail)
        head_end: Final = _end_of_last_frame(head)
        tail_start: Final = 0 if not self._tail_size else self._tail_start(tail)
        return [part for part in (head[:head_end], tail[tail_start:]) if part]

    def _tail_start(self, tail: bytes) -> int:
        # the tail begins wherever the eviction cut it, which is mid-frame unless nothing before it was dropped
        return _end_of_first_frame(tail) if self._head_closed else 0

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._frames())

    def __len__(self) -> int:
        return len(self._frames())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return False
        other_items: Final[Sequence[object]] = other
        return list(self) == list(other_items)

    def __bool__(self) -> bool:
        return self.total_bytes > 0

    @overload
    def __getitem__(self, index: int) -> bytes: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[bytes]: ...

    def __getitem__(self, index: int | slice) -> bytes | Sequence[bytes]:
        return self._frames()[index]
