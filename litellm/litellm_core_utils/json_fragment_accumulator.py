import json
from typing import Final, cast  # noqa: TID251  # raw_decode returns tuple[Any, int]; no cast-free unpack


class JSONFragmentAccumulator:
    """
    Buffers a JSON value that arrives piecemeal over a stream (SSE data split
    across TCP packets, one shard per network read, etc) without the O(n^2)
    cost of repeated `buffer += fragment` string concatenation.

    Fragments are appended to a list in O(1). The buffer is only rebuilt into
    a single string, and only decoded, when a caller asks for a value via
    `pop_next_value`, and `could_close_json` lets callers skip that rebuild
    entirely for fragments that plainly cannot close a JSON value yet.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []  # mutable-ok: O(1) append; string concat would copy the buffer each time

    def __bool__(self) -> bool:
        return bool(self._chunks)

    def append(self, fragment: str) -> None:
        self._chunks.append(fragment)  # mutable-ok: see __init__

    def could_close_json(self) -> bool:
        """
        Whether the buffer's logical last non-whitespace byte is "}" or "]",
        i.e. whether a JSON value could plausibly be complete. Scans
        fragments from the end and stops at the first non-blank one, so this
        is O(1) in the common case where the newest fragment is non-blank.
        """
        for stripped in (f.rstrip() for f in reversed(self._chunks) if f.rstrip()):
            return stripped[-1] in ("}", "]")
        return False

    def pop_next_value(self) -> tuple[bool, object]:
        """
        Attempt to decode one complete JSON value from the front of the
        buffer. On success, advances the buffer past that value (keeping any
        unconsumed tail, e.g. a second concatenated value) and returns
        (True, value). If the buffer is empty or holds no complete value
        yet, it is left untouched and this returns (False, None).
        """
        full: Final = "".join(self._chunks).strip()
        if not full:
            return False, None
        decoder: Final = json.JSONDecoder()
        try:
            raw_value: Final = decoder.raw_decode(full)
        except json.JSONDecodeError:
            return False, None
        decoded, end_index = cast("tuple[object, int]", raw_value)  # cast-ok: raw_decode returns tuple[Any, int]
        remainder: Final = full[end_index:].strip()
        self._chunks = [remainder] if remainder else []  # mutable-ok: replace buffer with the unconsumed tail
        return True, decoded

    def snapshot(self) -> str:
        return "".join(self._chunks)

    def set(self, value: str) -> None:
        """Replace the buffer's contents with a single fragment."""
        self._chunks = [value] if value else []  # mutable-ok: see __init__
