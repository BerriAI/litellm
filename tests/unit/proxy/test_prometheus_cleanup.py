"""
Tests for litellm.proxy.prometheus_cleanup.wipe_directory and
ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Final
from unittest.mock import patch

import pytest
from prometheus_client import CollectorRegistry, multiprocess

from litellm.proxy.prometheus_cleanup import mark_dead_workers, mark_worker_exit, wipe_directory
from litellm.proxy.proxy_cli import ProxyInitializationHelpers

_WORKER: Final = """
import sys, time
from prometheus_client import Gauge
Gauge("litellm_in_flight", "", multiprocess_mode="livesum").set(float(sys.argv[1]))
print("ready", flush=True)
if sys.argv[2] == "stay":
    time.sleep(120)
"""


def _spawn_worker(directory: Path, in_flight: str, lifetime: str) -> subprocess.Popen[str]:
    env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": str(directory)}
    worker = subprocess.Popen(
        [sys.executable, "-c", _WORKER, in_flight, lifetime], env=env, stdout=subprocess.PIPE, text=True
    )
    assert worker.stdout is not None and worker.stdout.readline() == "ready\n"
    return worker


def _livesum(directory: Path) -> float:
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(directory))
    value = registry.get_sample_value("litellm_in_flight")
    return 0.0 if value is None else value


class TestMarkDeadWorkers:
    def test_drops_live_gauges_of_exited_workers_and_keeps_running_ones(self, tmp_path: Path) -> None:
        """A worker that died mid-request leaves its livesum file behind; the replacement worker's startup prune
        must remove exactly that file so the aggregate stops counting requests nobody is serving."""
        dead = _spawn_worker(tmp_path, "3", "exit")
        assert dead.wait(timeout=30) == 0
        alive = _spawn_worker(tmp_path, "2", "stay")
        try:
            assert (tmp_path / f"gauge_livesum_{dead.pid}.db").exists()
            assert _livesum(tmp_path) == 5.0

            assert mark_dead_workers(str(tmp_path)) == (dead.pid,)

            assert not (tmp_path / f"gauge_livesum_{dead.pid}.db").exists()
            assert (tmp_path / f"gauge_livesum_{alive.pid}.db").exists()
            assert _livesum(tmp_path) == 2.0
            assert mark_dead_workers(str(tmp_path)) == ()
        finally:
            alive.kill()
            alive.wait(timeout=30)

    def test_leaves_counters_of_exited_workers_alone(self, tmp_path: Path) -> None:
        (tmp_path / "counter_424242.db").touch()
        (tmp_path / "histogram_424242.db").touch()
        assert mark_dead_workers(str(tmp_path)) == ()
        assert sorted(p.name for p in tmp_path.glob("*.db")) == ["counter_424242.db", "histogram_424242.db"]

    def test_keeps_live_gauges_of_workers_it_may_not_signal(self, tmp_path: Path) -> None:
        """Signal 0 to pid 1 raises PermissionError for an unprivileged proxy; that pid is alive, not dead."""
        (tmp_path / "gauge_livesum_1.db").touch()
        assert mark_dead_workers(str(tmp_path)) == ()
        assert (tmp_path / "gauge_livesum_1.db").exists()


class TestWipeDirectory:
    def test_deletes_all_db_files(self, tmp_path):
        (tmp_path / "counter_1234.db").touch()
        (tmp_path / "histogram_5678.db").touch()
        (tmp_path / "gauge_livesum_9999.db").touch()
        wipe_directory(str(tmp_path))
        assert not list(tmp_path.glob("*.db"))


class TestMarkWorkerExit:
    def test_calls_mark_process_dead_when_env_set(self, tmp_path):
        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch("prometheus_client.multiprocess.mark_process_dead") as mock_mark:
                mark_worker_exit(12345)
                mock_mark.assert_called_once_with(12345)

    def test_noop_when_env_not_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            with patch("prometheus_client.multiprocess.mark_process_dead") as mock_mark:
                mark_worker_exit(12345)
                mock_mark.assert_not_called()

    def test_exception_is_caught_and_logged(self, tmp_path):
        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch(
                "prometheus_client.multiprocess.mark_process_dead",
                side_effect=FileNotFoundError("gone"),
            ) as mock_mark:
                # Should not raise
                mark_worker_exit(99)
                mock_mark.assert_called_once_with(99)


class TestMaybeSetupPrometheusMultiprocDir:
    def test_respects_existing_env_var(self, tmp_path):
        """When PROMETHEUS_MULTIPROC_DIR is already set, don't override it."""
        custom_dir = str(tmp_path / "custom_prom")
        litellm_settings = {"callbacks": ["prometheus"]}

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": custom_dir}):
            ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir(
                num_workers=4,
                litellm_settings=litellm_settings,
            )

            assert os.environ["PROMETHEUS_MULTIPROC_DIR"] == custom_dir
            assert os.path.isdir(custom_dir)

    @pytest.mark.parametrize(
        "litellm_settings",
        [
            {"callbacks": "prometheus"},
            {"success_callback": "prometheus"},
            {"failure_callback": "prometheus"},
            {"callbacks": "custom_callback"},  # string but not prometheus
        ],
    )
    def test_handles_string_callbacks(self, litellm_settings):
        """When callbacks are specified as a string instead of a list, should not crash."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            os.environ.pop("prometheus_multiproc_dir", None)

            # Should not raise TypeError
            ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir(
                num_workers=4,
                litellm_settings=litellm_settings,
            )

            # Cleanup
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)

    @pytest.mark.parametrize(
        "num_workers, litellm_settings",
        [
            (1, {"callbacks": ["prometheus"]}),
            (4, {"callbacks": ["langfuse"]}),
            (4, None),
        ],
    )
    def test_noop_when_setup_not_needed(self, num_workers, litellm_settings):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            os.environ.pop("prometheus_multiproc_dir", None)

            ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir(
                num_workers=num_workers,
                litellm_settings=litellm_settings,
            )

            assert os.environ.get("PROMETHEUS_MULTIPROC_DIR") is None

    @pytest.mark.parametrize(
        "litellm_settings",
        [
            {"callbacks": ["prometheus"]},
            {"success_callback": ["prometheus"]},
        ],
    )
    def test_auto_creates_dir_when_prometheus_configured(self, litellm_settings):
        """When multiple workers + prometheus callback, auto-creates temp dir."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            os.environ.pop("prometheus_multiproc_dir", None)

            ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir(
                num_workers=4,
                litellm_settings=litellm_settings,
            )

            result_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
            assert result_dir is not None
            assert os.path.isdir(result_dir)

            # Cleanup
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)

    @pytest.mark.parametrize(
        "litellm_settings",
        [
            {"callbacks": ["prometheus"]},
            {"callbacks": ["langfuse"]},
            None,
        ],
    )
    def test_separate_metrics_port_forces_dir_for_single_worker(self, litellm_settings):
        """The separate metrics process reads the samples, so one worker still needs the shared dir, even when
        prometheus is not in config.yaml (callbacks can be turned on from the DB after startup)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            os.environ.pop("prometheus_multiproc_dir", None)

            result_dir = ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir(
                num_workers=1,
                litellm_settings=litellm_settings,
                prometheus_metrics_port=4001,
            )

            assert result_dir is not None
            assert os.environ.get("PROMETHEUS_MULTIPROC_DIR") == result_dir
            assert os.path.isdir(result_dir)

            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)

    def test_lowercase_env_var_is_reused_and_exported_uppercase(self, tmp_path):
        """prometheus_client honours both spellings; the metrics server only reads the uppercase one."""
        with patch.dict(os.environ, {"prometheus_multiproc_dir": str(tmp_path)}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)

            result_dir = ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir(
                num_workers=4,
                litellm_settings={"callbacks": "prometheus"},
            )

            assert result_dir == str(tmp_path)
            assert os.environ["PROMETHEUS_MULTIPROC_DIR"] == str(tmp_path)
