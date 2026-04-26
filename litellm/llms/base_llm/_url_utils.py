"""URL-encoding helpers for provider transformations."""

from typing import Optional
from urllib.parse import quote


def _reject_dot_segment(segment: str, full: str) -> None:
    if segment in ("..", "."):
        raise ValueError(f"Illegal path segment in identifier: {full!r}")


def encode_path_segment(segment: Optional[str], safe: str = "") -> str:
    """Percent-encode a single path segment. Raises on empty / ``None`` / ``.`` / ``..``.

    ``safe`` is forwarded to ``urllib.parse.quote`` for callers that need to
    preserve specific characters (e.g. ``:`` in Bedrock model IDs).
    """
    if segment is None or segment == "":
        raise ValueError("identifier is required, got empty or None")
    str_segment = str(segment)
    _reject_dot_segment(str_segment, str_segment)
    return quote(str_segment, safe=safe)


def encode_url_path(path: Optional[str]) -> str:
    """Percent-encode a multi-segment path; preserves ``/`` and ``@``.

    ``None`` / ``""`` → ``""``. Rejects ``.``, ``..``, or empty segments.
    """
    if path is None or path == "":
        return ""
    str_path = str(path)
    for segment in str_path.split("/"):
        if segment == "":
            raise ValueError(f"Empty path segment in identifier: {str_path!r}")
        _reject_dot_segment(segment, str_path)
    return quote(str_path, safe="/@")
