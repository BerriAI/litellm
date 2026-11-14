import re
from pathlib import Path
from typing import Final

import pytest
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest

import litellm
from litellm.constants import PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX
from litellm.integrations.prometheus_helpers.shared_prometheus_series_admissions import (
    SharedPrometheusSeriesAdmissions,
)
from litellm.proxy.prometheus_cleanup import wipe_directory

SERIES_SETTINGS: Final = (
    "prometheus_metrics_max_series_per_metric",
    "prometheus_metrics_ttl_seconds",
    "prometheus_metrics_cleanup_interval_seconds",
    "prometheus_exclude_labels",
    "prometheus_metrics_config",
    "enable_end_user_cost_tracking_prometheus_only",
    "prometheus_end_user_metrics_max_series_per_metric",
    "prometheus_end_user_metrics_ttl_seconds",
)


def _unregister_everything() -> None:
    for collector in list(REGISTRY._collector_to_names):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture(autouse=True)
def isolated_registry_and_settings(monkeypatch):
    collectors_before: Final = tuple(REGISTRY._collector_to_names)
    _unregister_everything()
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    for setting in SERIES_SETTINGS:
        monkeypatch.setattr(litellm, setting, getattr(litellm, setting))
    yield
    _unregister_everything()
    for collector in collectors_before:
        REGISTRY.register(collector)




def _scraped_series(sample_name: str, registry: CollectorRegistry = REGISTRY) -> frozenset[str]:
    exposition: Final = generate_latest(registry).decode()
    return frozenset(line for line in exposition.splitlines() if line.startswith(f"{sample_name}{{"))


def _label_values(series: frozenset[str], label: str) -> frozenset[str]:
    pattern: Final = re.compile(rf'[{{,]{label}="([^"]*)"')
    return frozenset(match.group(1) for match in map(pattern.search, series) if match is not None)


def _sample_value(series: frozenset[str], label: str, value: str) -> float:
    (line,) = (line for line in series if f'{label}="{value}"' in line)
    return float(line.rsplit(" ", 1)[1])














def test_workers_sharing_a_multiprocess_dir_admit_the_same_label_sets(tmp_path: Path):
    first_worker: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    second_worker: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))

    assert first_worker.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
    assert second_worker.admit_series("litellm_requests_metric", ("user-b",), max_series=2)

    replacement_worker: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    for worker in (first_worker, second_worker, replacement_worker):
        assert worker.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
        assert worker.admit_series("litellm_requests_metric", ("user-b",), max_series=2)
        assert not worker.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert first_worker.admit_series("litellm_spend_metric", ("user-c",), max_series=2)


def test_workers_agree_when_racing_appends_overfill_the_admissions_file(tmp_path: Path):
    racing_workers: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    for user in ("user-a", "user-b", "user-c"):
        assert racing_workers.admit_series("litellm_requests_metric", (user,), max_series=3)

    worker: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))

    assert not worker.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert worker.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
    assert worker.admit_series("litellm_requests_metric", ("user-b",), max_series=2)


def test_a_line_another_worker_is_still_writing_is_read_once_it_is_complete(tmp_path: Path):
    admissions_file: Final = tmp_path / f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric"
    assert SharedPrometheusSeriesAdmissions(directory=str(tmp_path)).admit_series(
        "litellm_requests_metric", ("user-a",), max_series=2
    )
    reader: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))

    with admissions_file.open("ab") as write_in_progress:
        write_in_progress.write(b'["user')
        write_in_progress.flush()
        assert reader.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
        write_in_progress.write(b'-b"]\n')

    assert not reader.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert reader.admit_series("litellm_requests_metric", ("user-b",), max_series=2)


def test_a_record_cut_short_by_a_full_disk_admits_nothing_and_hides_no_other_record(tmp_path: Path):
    admissions_file: Final = tmp_path / f"{PROMETHEUS_ADMITTED_SERIES_FILE_PREFIX}litellm_requests_metric"
    admissions_file.write_bytes(b'\n["user-a')
    writer: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    reader: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))

    assert writer.admit_series("litellm_requests_metric", ("user-b",), max_series=2)
    assert reader.admit_series("litellm_requests_metric", ("user-b",), max_series=2)
    assert reader.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert writer.admit_series("litellm_requests_metric", ("user-c",), max_series=2)
    assert not writer.admit_series("litellm_requests_metric", ("user-a",), max_series=2)
    assert not reader.admit_series("litellm_requests_metric", ("user-a",), max_series=2)


def test_wiping_the_multiprocess_dir_frees_every_admitted_slot(tmp_path: Path):
    before_restart: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    assert before_restart.admit_series("litellm_requests_metric", ("user-a",), max_series=1)

    wipe_directory(str(tmp_path))

    after_restart: Final = SharedPrometheusSeriesAdmissions(directory=str(tmp_path))
    assert after_restart.admit_series("litellm_requests_metric", ("user-b",), max_series=1)
    assert not after_restart.admit_series("litellm_requests_metric", ("user-a",), max_series=1)




















