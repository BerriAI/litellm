"""
Cursor-based pagination helpers shared across /v2/agents, /v2/sessions,
/v2/sessions/{sid}/runs.

The cursor is an opaque base64 string carrying ``(created_at, id)`` of the
last item from the previous page. The server decodes the cursor, fetches
``LIMIT + 1`` rows ordered by ``(created_at desc, id desc)`` starting
strictly before that point, and returns ``{items, next_cursor}``.

Why ``(created_at, id)`` and not just ``id``: ``created_at`` is the visible
sort key the SDK presents; ``id`` is the tiebreaker that guarantees a
total order across rows that share a millisecond. This matches Cursor's
SDK pagination shape and keeps the page boundary deterministic.

Cursors are NOT signed/encrypted — they're just an opaque encoding so the
SDK doesn't accidentally start treating them as parseable. A caller that
forges a cursor can only land themselves on a different page of their own
data; ownership is still enforced by the row filter on every query.
"""

import base64
import binascii
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException

# Default page size when ``limit`` is not specified. Keep modest so SDK
# default behavior doesn't drag a huge response over the wire.
DEFAULT_PAGE_LIMIT = 50
# Hard cap to protect the proxy from unbounded fetches.
MAX_PAGE_LIMIT = 500


def encode_cursor(row: Any) -> str:
    """Encode the (created_at, id) of a row as an opaque base64 cursor.

    Uses base64-urlsafe so the cursor survives being passed as a
    query-string parameter without escaping.
    """
    created_at = getattr(row, "created_at", None)
    row_id = getattr(row, "id", None)
    if created_at is None or row_id is None:
        # Should never happen for a real Prisma row — every table here
        # has both columns. If it does, callers should treat "no
        # cursor" as "no more pages".
        raise ValueError("Cannot encode cursor for row without created_at + id")
    if isinstance(created_at, datetime):
        # Always emit UTC ISO so the round-trip is timezone-stable.
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        created_str = created_at.isoformat()
    else:
        created_str = str(created_at)
    raw = f"{created_str}|{row_id}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_cursor(cursor: str) -> Tuple[datetime, str]:
    """Decode an opaque cursor into ``(created_at, id)``.

    Raises ``HTTPException(400)`` on malformed input — callers should let
    this propagate so the SDK gets a 400 with a clear error envelope.
    """
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        decoded = raw.decode("utf-8")
        ts_str, _, row_id = decoded.partition("|")
        if not ts_str or not row_id:
            raise ValueError("cursor missing created_at or id")
        # ``fromisoformat`` handles offset-aware ISO strings.
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts, row_id
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"invalid cursor: {exc}",
        ) from exc


def cursor_where_clause(
    base_where: Optional[Dict[str, Any]],
    cursor: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Compose the Prisma ``where`` clause for cursor pagination.

    The cursor selects rows STRICTLY OLDER than the previous page's last
    row, since list endpoints sort newest-first. Tie-breaker on ``id``
    keeps ordering deterministic across rows sharing a created_at
    timestamp.

    Returns the merged where dict (or ``None`` if both base and cursor
    are empty). Prisma's ``OR`` operator is used to express the
    ``(created_at < X) OR (created_at = X AND id < Y)`` predicate that
    cursor pagination needs.
    """
    if not cursor:
        return base_where if base_where else None

    ts, row_id = decode_cursor(cursor)
    cursor_predicate: Dict[str, Any] = {
        "OR": [
            {"created_at": {"lt": ts}},
            {"created_at": ts, "id": {"lt": row_id}},
        ]
    }
    if not base_where:
        return cursor_predicate
    # Combine with the existing owner/agent filters. AND-merge: every
    # base filter must hold AND the cursor predicate must hold.
    return {"AND": [base_where, cursor_predicate]}


def normalize_limit(limit: Optional[int]) -> int:
    """Clamp the user-supplied limit to ``[1, MAX_PAGE_LIMIT]``."""
    if limit is None:
        return DEFAULT_PAGE_LIMIT
    if limit < 1:
        return 1
    if limit > MAX_PAGE_LIMIT:
        return MAX_PAGE_LIMIT
    return limit


def build_page_response(
    rows: List[Any],
    limit: int,
    serializer: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    """Slice ``rows`` to ``limit`` and emit ``{items, next_cursor}``.

    Callers fetch ``limit + 1`` rows. If we got back more than ``limit``,
    there's a next page and the LAST row of the trimmed page becomes
    the next cursor. ``serializer`` runs per-row to produce the wire dict.
    """
    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor = encode_cursor(page[-1]) if has_more and page else None
    return {
        "items": [serializer(r) for r in page],
        "next_cursor": next_cursor,
    }
