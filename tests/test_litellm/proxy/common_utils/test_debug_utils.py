import os
import socket
from pathlib import Path

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.debug_utils import (
    PSUTIL_MISSING_ERROR,
    _ProcFilesystemProcess,
    _summary_process_memory,
    get_memory_summary,
)

PAGE_SIZE = 4096
STATM_SIZE_PAGES = 100_000
STATM_RESIDENT_PAGES = 30_000
MEMINFO_TOTAL_KB = 1_000_000


@pytest.fixture
def proc_process(tmp_path: Path) -> _ProcFilesystemProcess:
    statm = tmp_path / "statm"
    statm.write_text(f"{STATM_SIZE_PAGES} {STATM_RESIDENT_PAGES} 5000 1 0 20000 0\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        f"MemTotal:       {MEMINFO_TOTAL_KB} kB\nMemFree:         400000 kB\nMemAvailable:    600000 kB\n"
    )
    return _ProcFilesystemProcess(statm_path=str(statm), meminfo_path=str(meminfo), page_size=PAGE_SIZE)


def test_proc_filesystem_process_reads_resident_and_virtual_bytes_from_statm(
    proc_process: _ProcFilesystemProcess,
) -> None:
    memory_info = proc_process.memory_info()

    assert memory_info.rss == STATM_RESIDENT_PAGES * PAGE_SIZE
    assert memory_info.vms == STATM_SIZE_PAGES * PAGE_SIZE


def test_proc_filesystem_process_reports_share_of_meminfo_total(proc_process: _ProcFilesystemProcess) -> None:
    expected_percent = STATM_RESIDENT_PAGES * PAGE_SIZE / (MEMINFO_TOTAL_KB * 1024) * 100

    assert proc_process.memory_percent() == pytest.approx(expected_percent)


def test_summary_reports_rss_from_the_proc_filesystem(proc_process: _ProcFilesystemProcess) -> None:
    memory, health_status = _summary_process_memory(proc_process)

    assert memory["ram_usage_mb"] == round(STATM_RESIDENT_PAGES * PAGE_SIZE / (1024 * 1024), 2)
    assert memory["system_memory_percent"] == pytest.approx(12.0)
    assert health_status == "healthy"
    assert "error" not in memory


def test_summary_without_any_memory_source_names_psutil_and_reports_no_rss() -> None:
    memory, health_status = _summary_process_memory(None)

    assert memory == {"error": PSUTIL_MISSING_ERROR}
    assert health_status == "healthy"


@pytest.mark.asyncio
async def test_memory_summary_names_the_host_and_worker_that_answered() -> None:
    summary = await get_memory_summary(UserAPIKeyAuth())

    assert summary["hostname"] == socket.gethostname()
    assert summary["worker_pid"] == os.getpid()
    assert summary["memory"]["ram_usage_mb"] > 0
