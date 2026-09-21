import json
from typing import Final, cast  # noqa: TID251  # raw_decode returns tuple[Any, int]; no cast-free unpack


class JSONFragmentAccumulator:
    """
    Buffers a JSON value that arrives piecemeal over a stream (SSE data split
    across TCP packets, one shard per network read, etc) without the O(n^2)
    cost of repeated `buffer += fragment` string concatenation, and without
    the O(n^2) cost of re-copying the unconsumed remainder on every peeled
    value when one payload holds many concatenated JSON values.

    Fragments are appended to a list in O(1). The buffer is only rebuilt into
    a single string, and only decoded, when a caller asks for a value via
    `pop_next_value`, and `could_close_json` lets callers skip that rebuild
    entirely for fragments that plainly cannot close a JSON value yet. Once
    rebuilt, consumed values are dropped by advancing a cursor rather than
    slicing a new string, so draining N concatenated values already sitting
    in the buffer costs O(n) total, not O(n^2).
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []  # mutable-ok: O(1) append; string concat would copy the buffer each time
        self._buffer: str = (
            ""  # mutable-ok: lazily materialized join of _chunks, rebuilt only when _chunks is non-empty
        )
        self._offset: int = 0  # mutable-ok: cursor past already-consumed values; avoids re-slicing on every pop
        self._could_close: bool = False  # mutable-ok: cached heuristic; rescanning past fragments was itself O(n^2)

    def __bool__(self) -> bool:
        return bool(self._chunks) or self._offset < len(self._buffer)

    def append(self, fragment: str) -> None:
        self._chunks.append(fragment)  # mutable-ok: see __init__
        stripped: Final = fragment.rstrip()
        if stripped:
            self._could_close = stripped[-1] in ("}", "]")  # mutable-ok: see __init__

    def could_close_json(self) -> bool:
        """
        Whether the buffer's logical last non-whitespace byte is "}" or "]",
        i.e. whether a JSON value could plausibly be complete. Tracked
        incrementally in `append` rather than rescanned here, so a run of
        blank keepalive fragments (e.g. from a malformed upstream stream)
        can't make this, or the join+parse it gates, cost O(n^2).
        """
        return self._could_close

    def _materialize(self) -> None:
        if not self._chunks:
            return
        unconsumed: Final = self._buffer[self._offset :]
        self._buffer = unconsumed + "".join(self._chunks)  # mutable-ok: merge pending fragments, once per append batch
        self._offset = 0  # mutable-ok: see __init__
        self._chunks = []  # mutable-ok: see __init__

    def pop_next_value(self) -> tuple[bool, object]:
        """
        Attempt to decode one complete JSON value from the front of the
        buffer. On success, advances a cursor past that value (keeping any
        unconsumed tail, e.g. a second concatenated value, in place rather
        than copying it) and returns (True, value). If the buffer is empty
        or holds no complete value yet, it is left untouched and this
        returns (False, None).
        """
        self._materialize()
        length: Final = len(self._buffer)
        start = self._offset
        while start < length and self._buffer[start].isspace():
            start += 1
        if start >= length:
            self._offset = start  # mutable-ok: see __init__
            return False, None
        decoder: Final = json.JSONDecoder()
        try:
            raw_value: Final = decoder.raw_decode(self._buffer, start)
        except json.JSONDecodeError:
            return False, None
        decoded, end_index = cast("tuple[object, int]", raw_value)  # cast-ok: raw_decode returns tuple[Any, int]
        self._offset = end_index  # mutable-ok: see __init__
        if self._offset >= len(self._buffer):
            self._buffer = ""  # mutable-ok: see __init__
            self._offset = 0  # mutable-ok: see __init__
            self._could_close = False  # mutable-ok: buffer is empty, nothing can close
        return True, decoded

    def snapshot(self) -> str:
        self._materialize()
        return self._buffer[self._offset :]

    def set(self, value: str) -> None:
        """Replace the buffer's contents with a single fragment."""
        self._chunks = []  # mutable-ok: see __init__
        self._buffer = value  # mutable-ok: see __init__
        self._offset = 0  # mutable-ok: see __init__
        stripped: Final = value.rstrip()
        self._could_close = bool(stripped) and stripped[-1] in ("}", "]")  # mutable-ok: see __init__
