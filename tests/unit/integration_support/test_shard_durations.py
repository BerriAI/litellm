from pathlib import Path
from typing import Final

from tests.integration._support.shard_durations import recorded_durations, refreshed


def _junit(path: Path, cases: tuple[tuple[str, float], ...]) -> Path:
    body: Final = "".join(f'<testcase classname="c" name="t{index}" file="{file}" time="{seconds}" />' for index, (file, seconds) in enumerate(cases))
    path.write_text(f'<?xml version="1.0"?><testsuites><testsuite name="pytest">{body}</testsuite></testsuites>')
    return path


def test_a_file_records_the_sum_of_its_cases_and_the_slowest_report_wins(tmp_path: Path) -> None:
    standard: Final = _junit(tmp_path / "standard.xml", (("a.py", 1.5), ("a.py", 2.5), ("b.py", 3.0)))
    replica: Final = _junit(tmp_path / "replica.xml", (("a.py", 3.0), ("b.py", 7.0)))
    assert recorded_durations((standard, replica)) == {"a.py": 4.0, "b.py": 7.0}


def test_a_refresh_keeps_unrecorded_files_and_drops_deleted_ones(tmp_path: Path) -> None:
    for name in ("kept.py", "updated.py", "instant.py"):
        (tmp_path / name).write_text("")
    existing: Final = {"kept.py": 40.0, "updated.py": 5.0, "deleted.py": 9.0}
    recorded: Final = {"updated.py": 12.4, "instant.py": 0.01}
    assert refreshed(existing, recorded, tmp_path) == {"instant.py": 1, "kept.py": 40, "updated.py": 12}
