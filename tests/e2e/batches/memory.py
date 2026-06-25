"""Peak-memory sampling for the batch-upload memory guard.

The OOM regression shows up as the proxy's resident memory growing with the
uploaded file size. To catch it, sample the proxy's memory while an upload runs
and keep the peak. The sampler is a seam (Protocol) so the environment decides how
memory is read: the local docker-compose stack reads the container's cgroup, an
EKS run would read the gateway pod (e.g. via `kubectl exec`/metrics) with the same
interface.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PeakMemory:
    baseline_bytes: int
    peak_bytes: int

    @property
    def growth_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)


class MemorySampler(Protocol):
    def measure(self, during: Callable[[], T]) -> tuple[T, PeakMemory]:
        """Run `during`, sampling memory throughout; return its result and the peak
        memory observed against the pre-run baseline."""
        ...


@dataclass(frozen=True, slots=True)
class DockerCgroupSampler:
    """Reads the litellm container's anonymous (RSS) memory from its cgroup
    (`anon` in `/sys/fs/cgroup/memory.stat`, cgroup v2) on a background thread.
    Anonymous memory is what a buffered-in-memory copy of the upload shows up as;
    `memory.current` is avoided because it also counts reclaimable page cache, which
    a streamed-to-disk upload fills without ever risking an OOM."""

    container: str
    interval_seconds: float = 0.1

    def _read_bytes(self) -> int:
        try:
            result = subprocess.run(
                ["docker", "exec", self.container, "cat", "/sys/fs/cgroup/memory.stat"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return -1
        for line in result.stdout.splitlines():
            field, _, value = line.partition(" ")
            if field == "anon" and value.strip().isdigit():
                return int(value.strip())
        return -1

    def measure(self, during: Callable[[], T]) -> tuple[T, PeakMemory]:
        baseline = self._read_bytes()
        if baseline < 0:
            raise RuntimeError(
                f"could not read anon memory from container {self.container!r}; the "
                "memory guard cannot run, so refusing to pass vacuously (needs cgroup "
                "v2 and docker exec access)"
            )
        peak = baseline
        stop = threading.Event()

        def sample() -> None:
            nonlocal peak
            while not stop.is_set():
                current = self._read_bytes()
                if current > peak:
                    peak = current
                time.sleep(self.interval_seconds)

        sampler_thread = threading.Thread(target=sample)
        sampler_thread.start()
        try:
            result = during()
        finally:
            stop.set()
            sampler_thread.join()
        return result, PeakMemory(baseline_bytes=baseline, peak_bytes=peak)
