from typing import Final

import pytest

from perf.database_observe import IndexSample
from perf.database_queries import latency_regressed
from perf.database_upgrade import RunSpec, UpgradeWindow, background_overlap


@pytest.mark.parametrize(
    ("pending", "active", "persisted", "expected"),
    (
        (("new-index",), 1, True, "passed"),
        (("new-index",), 1, False, "failed"),
        (("new-index",), 0, True, "failed"),
        ((), 0, True, "failed"),
    ),
)
def test_background_build_requires_target_spend_write_before_index_completion(
    pending: tuple[str, ...],
    active: int,
    persisted: bool,
    expected: str,
) -> None:
    before: Final = IndexSample(observed=0, pending=("new-index",), active_builds=0, target_probe_persisted=False)
    during: Final = IndexSample(observed=1, pending=pending, active_builds=active, target_probe_persisted=persisted)
    window: Final = UpgradeWindow(
        started=0,
        finished=2,
        exit_code=0,
        traffic=(),
        database=(),
        indexes_before=before,
        indexes=(during,),
    )
    assert background_overlap(window) == expected


def test_existing_indexes_do_not_claim_background_build_coverage() -> None:
    sample: Final = IndexSample(observed=0, pending=(), active_builds=0, target_probe_persisted=True)
    window: Final = UpgradeWindow(
        started=0,
        finished=2,
        exit_code=0,
        traffic=(),
        database=(),
        indexes_before=sample,
        indexes=(sample,),
    )
    assert background_overlap(window) == "no-pending-index-work"


@pytest.mark.parametrize(
    ("baseline", "target", "regressed"),
    ((0.01, 0.5, False), (0.01, 0.501, True), (0.8, 1.6, False), (0.8, 1.601, True)),
)
def test_default_query_threshold_uses_two_times_baseline_with_a_noise_floor(
    baseline: float, target: float, regressed: bool
) -> None:
    spec: Final = RunSpec(baseline_version="v0.0.0", target_version="v0.0.1", model="test", target_indexes=("index",))
    assert latency_regressed(baseline, target, spec.query_regression_ratio) is regressed
