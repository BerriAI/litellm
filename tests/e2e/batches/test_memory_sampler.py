"""The memory sampler must fail loudly, never vacuously pass.

If the cgroup read fails (no cgroup v2, docker exec unavailable, a different
memory.stat layout) _read_bytes returns -1; without a guard, baseline and peak are
both -1, growth is 0, and the OOM assertion passes as if the proxy used no memory -
hiding the LIT-3382 regression. measure() must raise instead.
"""

from __future__ import annotations

import pytest

from memory import DockerCgroupSampler


def test_measure_raises_when_cgroup_unreadable() -> None:
    sampler = DockerCgroupSampler(container="e2e-nonexistent-container-xyz")
    with pytest.raises(RuntimeError):
        sampler.measure(lambda: None)
