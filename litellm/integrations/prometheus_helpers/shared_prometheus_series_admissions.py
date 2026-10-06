from __future__ import annotations

import os
from threading import RLock
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.constants import PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX

_LABEL_VALUES: Final = TypeAdapter(tuple[str, ...])


def _parse_admission(line: bytes) -> tuple[str, ...] | None:
    try:
        return _LABEL_VALUES.validate_json(line)
    except ValidationError:
        return None


class _MetricAdmissions:
    __slots__ = ("_label_sets", "_max_series", "_path", "_read_offset")

    def __init__(self, path: str, max_series: int) -> None:
        self._path = path
        self._max_series = max_series
        self._label_sets: set[tuple[str, ...]] = (  # mutable-ok: a frozenset copy per admission is quadratic in the cap
            set()
        )
        self._read_offset = 0

    def admit(self, label_values: tuple[str, ...]) -> bool:
        if label_values in self._label_sets:
            return True
        if self._is_full():
            return False
        self._read_new_admissions()
        if label_values not in self._label_sets and not self._is_full():
            self._append(label_values)
            self._read_new_admissions()
        return label_values in self._label_sets

    def _is_full(self) -> bool:
        return len(self._label_sets) >= self._max_series

    def _append(self, label_values: tuple[str, ...]) -> None:
        descriptor: Final = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, b"\n" + _LABEL_VALUES.dump_json(label_values) + b"\n")
        finally:
            os.close(descriptor)

    def _read_new_admissions(self) -> None:
        try:
            with open(self._path, "rb") as admissions_file:
                admissions_file.seek(self._read_offset)
                unread: Final = admissions_file.read()
        except FileNotFoundError:
            return
        complete_lines, newline, _ = unread.rpartition(b"\n")
        if not newline:
            return
        self._read_offset += len(complete_lines) + len(newline)
        for label_values in map(_parse_admission, complete_lines.split(b"\n")):
            if self._is_full():
                return
            if label_values is not None:
                self._label_sets.add(label_values)


class SharedPrometheusSeriesAdmissions:
    """Picks which label sets get a series when several worker processes write to one
    ``PROMETHEUS_MULTIPROC_DIR``. Each metric has one append-only file there, and its first ``max_series``
    distinct lines are the admitted label sets. Every worker reads the same lines in the same order, so all of
    them, including a worker that replaces an exited one, admit the same label sets and a scrape that merges
    the workers stays at the cap. Each record sits between two newlines, so a record a worker could only write
    part of (the directory ran out of space) is a line of its own that admits nothing for every worker, and
    it neither hides the records after it nor runs into the next worker's record."""

    def __init__(self, directory: str) -> None:
        self._directory = directory
        self._admissions: dict[str, _MetricAdmissions] = {}  # mutable-ok: one entry per metric, added on first use
        self.lock = RLock()

    def admit_series(self, metric_name: str, label_values: tuple[str, ...], max_series: int) -> bool:
        with self.lock:
            if metric_name not in self._admissions:
                self._admissions[metric_name] = _MetricAdmissions(
                    path=os.path.join(self._directory, f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}{metric_name}"),
                    max_series=max_series,
                )
            return self._admissions[metric_name].admit(label_values)
