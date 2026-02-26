"""
Tests for litellm.proxy.prometheus_cleanup module and
ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from litellm.proxy.prometheus_cleanup import (
    _extract_pids_from_dir,
    _is_pid_alive,
    cleanup_own_pid_files,
    mark_dead_pids,
    wipe_directory,
)
from litellm.proxy.proxy_cli import ProxyInitializationHelpers


class TestExtractPidsFromDir:
    def test_counter_files(self, tmp_path):
        (tmp_path / "counter_1234.db").touch()
        (tmp_path / "counter_5678.db").touch()
        assert _extract_pids_from_dir(str(tmp_path)) == {1234, 5678}

    def test_histogram_files(self, tmp_path):
        (tmp_path / "histogram_1234.db").touch()
        assert _extract_pids_from_dir(str(tmp_path)) == {1234}

    def test_gauge_files(self, tmp_path):
        (tmp_path / "gauge_livesum_1234.db").touch()
        (tmp_path / "gauge_liveall_5678.db").touch()
        assert _extract_pids_from_dir(str(tmp_path)) == {1234, 5678}

    def test_multiple_pids(self, tmp_path):
        (tmp_path / "counter_100.db").touch()
        (tmp_path / "histogram_200.db").touch()
        (tmp_path / "gauge_livesum_300.db").touch()
        assert _extract_pids_from_dir(str(tmp_path)) == {100, 200, 300}

    def test_non_db_files_ignored(self, tmp_path):
        (tmp_path / "counter_1234.db").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "data.json").touch()
        assert _extract_pids_from_dir(str(tmp_path)) == {1234}

    def test_empty_directory(self, tmp_path):
        assert _extract_pids_from_dir(str(tmp_path)) == set()

    def test_nonexistent_directory(self):
        assert _extract_pids_from_dir("/nonexistent/path/abc123") == set()

    def test_malformed_filenames(self, tmp_path):
        (tmp_path / "counter_.db").touch()  # no PID
        (tmp_path / "random.db").touch()  # no underscore+PID pattern
        (tmp_path / "counter_abc.db").touch()  # non-numeric PID
        assert _extract_pids_from_dir(str(tmp_path)) == set()


class TestIsPidAlive:
    def test_own_pid_is_alive(self):
        assert _is_pid_alive(os.getpid()) is True

    def test_dead_pid(self):
        # A very high PID is almost certainly dead
        assert _is_pid_alive(4_000_000) is False

    def test_permission_error_treated_as_alive(self):
        with patch("os.kill", side_effect=PermissionError):
            assert _is_pid_alive(99999) is True


class TestWipeDirectory:
    def test_deletes_all_db_files(self, tmp_path):
        (tmp_path / "counter_1234.db").touch()
        (tmp_path / "histogram_5678.db").touch()
        (tmp_path / "gauge_livesum_9999.db").touch()
        wipe_directory(str(tmp_path))
        assert not list(tmp_path.glob("*.db"))

    def test_preserves_non_db_files(self, tmp_path):
        (tmp_path / "counter_1234.db").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "config.json").touch()
        wipe_directory(str(tmp_path))
        assert not list(tmp_path.glob("*.db"))
        assert (tmp_path / "readme.txt").exists()
        assert (tmp_path / "config.json").exists()

    def test_empty_directory(self, tmp_path):
        wipe_directory(str(tmp_path))
        assert not list(tmp_path.glob("*.db"))


class TestCleanupOwnPidFiles:
    def test_calls_mark_process_dead_for_own_pid(self, tmp_path):
        """Should call mark_process_dead with current PID on shutdown."""
        pid = os.getpid()

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch(
                "prometheus_client.multiprocess.mark_process_dead"
            ) as mock_mark_dead:
                cleanup_own_pid_files()
                mock_mark_dead.assert_called_once_with(pid)

    def test_noop_when_not_configured(self, tmp_path):
        """Should not call mark_process_dead when env var is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            os.environ.pop("prometheus_multiproc_dir", None)
            with patch(
                "prometheus_client.multiprocess.mark_process_dead"
            ) as mock_mark_dead:
                cleanup_own_pid_files()
                mock_mark_dead.assert_not_called()


class TestMarkDeadPids:
    def test_calls_mark_process_dead_for_dead_pids(self, tmp_path):
        """mark_dead_pids should call mark_process_dead() for dead PIDs."""
        dead_pid = 4_000_000
        (tmp_path / f"counter_{dead_pid}.db").touch()
        (tmp_path / f"gauge_livesum_{dead_pid}.db").touch()

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch(
                "prometheus_client.multiprocess.mark_process_dead"
            ) as mock_mark_dead:
                mark_dead_pids()
                mock_mark_dead.assert_called_once_with(dead_pid)

    def test_skips_own_pid(self, tmp_path):
        """Should not call mark_process_dead for the current process."""
        own_pid = os.getpid()
        (tmp_path / f"counter_{own_pid}.db").touch()

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch(
                "prometheus_client.multiprocess.mark_process_dead"
            ) as mock_mark_dead:
                mark_dead_pids()
                mock_mark_dead.assert_not_called()

    def test_skips_alive_pids(self, tmp_path):
        """Should not call mark_process_dead for alive PIDs."""
        alive_pid = 99999
        (tmp_path / f"counter_{alive_pid}.db").touch()

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch(
                "litellm.proxy.prometheus_cleanup._is_pid_alive",
                return_value=True,
            ):
                with patch(
                    "prometheus_client.multiprocess.mark_process_dead"
                ) as mock_mark_dead:
                    mark_dead_pids()
                    mock_mark_dead.assert_not_called()

    def test_handles_mixed_alive_and_dead(self, tmp_path):
        """Should only call mark_process_dead for dead PIDs."""
        dead_pid = 4_000_000
        own_pid = os.getpid()
        (tmp_path / f"counter_{dead_pid}.db").touch()
        (tmp_path / f"counter_{own_pid}.db").touch()

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch(
                "prometheus_client.multiprocess.mark_process_dead"
            ) as mock_mark_dead:
                mark_dead_pids()
                mock_mark_dead.assert_called_once_with(dead_pid)

    def test_noop_when_not_configured(self, tmp_path):
        """Should do nothing when PROMETHEUS_MULTIPROC_DIR is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            os.environ.pop("prometheus_multiproc_dir", None)
            with patch(
                "prometheus_client.multiprocess.mark_process_dead"
            ) as mock_mark_dead:
                mark_dead_pids()
                mock_mark_dead.assert_not_called()


class TestMaybeSetupPrometheusMultiprocDir:
    def test_auto_creates_dir_when_prometheus_configured(self):
        """When multiple workers + prometheus callback, auto-creates temp dir."""
        litellm_settings = {"callbacks": ["prometheus"]}

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
            expected = os.path.join(
                tempfile.gettempdir(), "litellm_prometheus_multiproc"
            )
            assert result_dir == expected

            # Cleanup
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)

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

    def test_wipes_stale_files_on_setup(self, tmp_path):
        """Should wipe existing .db files from a previous run."""
        custom_dir = str(tmp_path / "prom_dir")
        os.makedirs(custom_dir)
        # Simulate stale files from a previous run
        for name in ["counter_9999.db", "histogram_9999.db", "gauge_livesum_9999.db"]:
            with open(os.path.join(custom_dir, name), "w") as f:
                f.write("stale")

        litellm_settings = {"callbacks": ["prometheus"]}

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": custom_dir}):
            ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir(
                num_workers=4,
                litellm_settings=litellm_settings,
            )

            # All .db files should be wiped
            import glob
            remaining = glob.glob(os.path.join(custom_dir, "*.db"))
            assert remaining == []

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

    def test_prometheus_in_success_callback(self):
        """Prometheus in success_callback should also trigger setup."""
        litellm_settings = {"success_callback": ["prometheus"]}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            os.environ.pop("prometheus_multiproc_dir", None)

            ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir(
                num_workers=2,
                litellm_settings=litellm_settings,
            )

            result_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
            assert result_dir is not None
            assert os.path.isdir(result_dir)

            # Cleanup
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
