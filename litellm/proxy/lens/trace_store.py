import json
import sqlite3
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Final

from pydantic import TypeAdapter

from .models import Evidence, TracePart

_ROW: Final = TypeAdapter(tuple[str])
_OPTIONAL_ROW: Final = TypeAdapter(tuple[str] | None)
_COUNT: Final = TypeAdapter(tuple[int])


class TraceStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection: Final = connection
        connection.execute("CREATE TABLE spans (span_id TEXT PRIMARY KEY, body TEXT NOT NULL)")
        connection.execute("CREATE TABLE reads (span_id TEXT, body TEXT, UNIQUE(span_id, body))")

    def add(self, parts: tuple[TracePart, ...]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO spans VALUES (?, ?)",
            ((part.span_id, part.model_dump_json()) for part in parts),
        )

    def add_reads(self, parts: tuple[TracePart, ...]) -> None:
        self.connection.executemany(
            "INSERT OR IGNORE INTO reads VALUES (?, ?)",
            ((part.span_id, part.model_dump_json()) for part in parts),
        )

    def evidence(self, evidence: Evidence) -> TracePart | None:
        rows: Final = self.connection.execute(
            "SELECT body FROM spans WHERE span_id=? UNION ALL SELECT body FROM reads WHERE span_id=?",
            (evidence.span_id, evidence.span_id),
        )
        for row in map(_ROW.validate_python, rows):
            part = TracePart.model_validate_json(row[0])
            if part.execution_id == evidence.execution_id and any(
                evidence.quote in segment for segment in part.content.split("\n[... content omitted ...]\n")
            ):
                return part
        return None

    def parts(self) -> Iterator[TracePart]:
        for row in map(_ROW.validate_python, self.connection.execute("SELECT body FROM spans ORDER BY span_id")):
            yield TracePart.model_validate_json(row[0])

    def get(self, span_id: str) -> TracePart | None:
        row: Final = _OPTIONAL_ROW.validate_python(
            self.connection.execute("SELECT body FROM spans WHERE span_id=?", (span_id,)).fetchone()
        )
        return TracePart.model_validate_json(row[0]) if row else None

    def previous(self, span_id: str) -> str:
        row: Final = _OPTIONAL_ROW.validate_python(
            self.connection.execute(
                "SELECT span_id FROM spans WHERE span_id < ? ORDER BY span_id DESC LIMIT 1", (span_id,)
            ).fetchone()
        )
        return row[0] if row else ""

    def count(self) -> int:
        return _COUNT.validate_python(self.connection.execute("SELECT count(*) FROM spans").fetchone())[0]

    def catalogs(self, root_count: int) -> Iterator[tuple[tuple[str, str, str, str, str, str, str], ...]]:
        rows: list[tuple[str, str, str, str, str, str, str]] = []  # mutable-ok: one bounded catalog window
        size = 0  # rebind-ok: track the current window's serialized size
        for part in self.parts():
            row = (
                part.span_id,
                part.parent_span_id,
                part.name,
                part.kind,
                overview_content(part, root_count),
                part.start_time,
                part.end_time,
            )
            width = len(json.dumps(row))
            if rows and size + width > 24000:
                yield tuple(rows)
                rows.clear()
                size = 0
            rows.append(row)
            size += width
        if rows:
            yield tuple(rows)


def overview_content(part: TracePart, root_count: int) -> str:
    limit: Final = max(160, min(2000, 12000 // max(root_count, 1))) if not part.parent_span_id else 160
    if len(part.content) <= limit:
        return part.content
    return (
        part.content[: limit // 3]
        + "\n[... preview omitted; read this span for evidence ...]\n"
        + part.content[-(limit * 2 // 3) :]
    )


@contextmanager
def trace_store() -> Generator[TraceStore]:
    with TemporaryDirectory(prefix="lens-trace-") as directory:
        connection: Final = sqlite3.connect(f"{directory}/trace.sqlite")
        try:
            yield TraceStore(connection)
        finally:
            connection.close()
