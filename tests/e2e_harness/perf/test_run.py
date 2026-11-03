from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import pytest
from setup.run import run_until
from setup.wire import WireRecord
from test_upgrade import record


class StopAfterRecord:
    def __init__(self, *, partial: Path, resume: Path, fail: bool = False) -> None:
        self.partial: Final = partial
        self.resume: Final = resume
        self.fail: Final = fail
        self.received: tuple[WireRecord, ...] = ()

    def __call__(self, records: tuple[WireRecord, ...]) -> bool:
        self.received += records
        if self.partial.exists():
            self.resume.touch()
        assert not (self.fail and records), "Database failed"
        return bool(records)


@pytest.mark.parametrize("fail", [False, True])
def test_controlled_process_preserves_partial_lines_and_drains_on_stop_or_failure(tmp_path: Path, fail: bool) -> None:
    records: Final = tmp_path / "records.jsonl"
    stop: Final = tmp_path / "stop"
    partial: Final = tmp_path / "partial"
    resume: Final = tmp_path / "resume"
    records.touch()
    encoded: Final = record().model_dump_json() + "\n"
    program: Final = f"""
import time
from pathlib import Path
with Path({str(records)!r}).open("w") as f:
    text = {encoded!r}
    f.write(text[:10])
    f.flush()
    Path({str(partial)!r}).touch()
    while not Path({str(resume)!r}).exists():
        time.sleep(0.01)
    f.write(text[10:])
    f.flush()
    while not Path({str(stop)!r}).exists():
        time.sleep(0.01)
    f.write({record(1).model_dump_json()!r} + "\\n")
    f.flush()
"""
    callback: Final = StopAfterRecord(partial=partial, resume=resume, fail=fail)
    command: Final = [sys.executable, "-c", program]
    if fail:
        with pytest.raises(AssertionError, match="Database failed"):
            _ = run_until(command, dict(os.environ), records, stop, callback, timeout=10)
    else:
        result: Final = run_until(command, dict(os.environ), records, stop, callback, timeout=10)
        assert result.returncode == 0, result.stdout
    assert callback.received == (record(),)
    assert tuple(WireRecord.model_validate_json(line) for line in records.read_text().splitlines()) == (
        record(),
        record(1),
    )
