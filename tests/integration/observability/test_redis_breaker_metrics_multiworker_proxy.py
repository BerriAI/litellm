"""A two-worker proxy sharing one ``PROMETHEUS_MULTIPROC_DIR``: the Redis circuit-breaker state gauge on /metrics is
one series per state summed over the workers, never one per worker pid, and it is there even though the proxy
builds its Redis cache before it turns the prometheus callback on."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from integration._support.prometheus_series import Sample, scrape, series_cap_rig

_STATE_GAUGE: Final = "litellm_redis_circuit_breaker_state"
_WORKERS: Final = 2


def _breaker_state_samples(samples: tuple[Sample, ...]) -> tuple[Sample, ...]:
    return tuple(sample for sample in samples if sample.name == _STATE_GAUGE)


@pytest.mark.timeout(300)
def test_two_proxy_workers_scrape_one_closed_breaker_series(tmp_path: Path) -> None:
    with series_cap_rig(tmp_path, {}, workers=_WORKERS, warm_keys=1, multiproc_dir=tmp_path / "prom") as rig:
        breaker: Final = _breaker_state_samples(scrape(rig.gateway))
    assert tuple(dict(sample.labels) for sample in breaker) == ({"state": "closed"},), breaker
    assert breaker[0].value >= _WORKERS and breaker[0].value % _WORKERS == 0, breaker
