"""
Tests for litellm.proxy.prometheus_cleanup.wipe_directory,
cleanup_stale_worker_files, and
ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from litellm.proxy.prometheus_cleanup import (
    _is_pid_alive,
    cleanup_stale_worker_files,
    mark_worker_exit,
    wipe_directory,
)
from litellm.proxy.proxy_cli import ProxyInitializationHelpers


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

    # ------------------------------------------------------------------
    # Orphaned-file cleanup (LIT-1969)
    # prometheus_client.mark_process_dead() only removes live gauge files;
    # counters, histograms, and non-live gauges must be deleted explicitly.
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "filename",
        [
            "counter_42.db",
            "histogram_42.db",
            "summary_42.db",
            "gauge_all_42.db",
            "gauge_min_42.db",
            "gauge_max_42.db",
            "gauge_mostrecent_42.db",
            "gauge_livesum_42.db",  # also deleted even if mark_process_dead already did it
        ],
    )
    def test_deletes_all_db_files_for_dead_pid(self, tmp_path, filename):
        """All *_{pid}.db file types are removed after worker exit."""
        dead_file = tmp_path / filename
        dead_file.touch()

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch("prometheus_client.multiprocess.mark_process_dead"):
                mark_worker_exit(42)

        assert not dead_file.exists(), f"{filename} should have been deleted"

    def test_does_not_delete_files_for_other_pids(self, tmp_path):
        """Files belonging to still-alive workers must not be touched."""
        (tmp_path / "counter_42.db").touch()
        (tmp_path / "counter_99.db").touch()  # different (live) worker
        (tmp_path / "histogram_99.db").touch()

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch("prometheus_client.multiprocess.mark_process_dead"):
                mark_worker_exit(42)

        assert not (tmp_path / "counter_42.db").exists()
        assert (tmp_path / "counter_99.db").exists()
        assert (tmp_path / "histogram_99.db").exists()

    def test_handles_already_deleted_files_gracefully(self, tmp_path):
        """If mark_process_dead already removed a file, no exception is raised."""
        # Create a live-gauge file, then simulate mark_process_dead deleting it
        # before our glob runs (race condition / already cleaned up).
        live_file = tmp_path / "gauge_livesum_77.db"
        live_file.touch()

        def side_effect_delete(pid):
            live_file.unlink()  # simulate prometheus_client removing it

        with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": str(tmp_path)}):
            with patch(
                "prometheus_client.multiprocess.mark_process_dead",
                side_effect=side_effect_delete,
            ):
                # Should not raise even though the file is already gone
                mark_worker_exit(77)

        assert not live_file.exists()


class TestIsPidAlive:
    def test_current_process_is_alive(self):
        assert _is_pid_alive(os.getpid()) is True

    def test_nonexistent_pid_is_dead(self):
        with patch("os.kill", side_effect=ProcessLookupError):
            assert _is_pid_alive(999999) is False

    def test_permission_error_means_alive(self):
        # EPERM means process exists but we can't signal it.
        with patch("os.kill", side_effect=PermissionError):
            assert _is_pid_alive(1) is True


class TestCleanupStaleWorkerFiles:
    """
    Test matrix for cleanup_stale_worker_files():

    | Filename pattern            | PID state | Expected action  |
    |-----------------------------|-----------|------------------|
    | counter_{pid}.db            | dead      | deleted          |
    | histogram_{pid}.db          | dead      | deleted          |
    | summary_{pid}.db            | dead      | deleted          |
    | gauge_all_{pid}.db          | dead      | deleted          |
    | gauge_livesum_{pid}.db      | dead      | deleted          |
    | counter_{pid}.db            | alive     | kept             |
    | gauge_livesum_{pid}.db      | alive     | kept             |
    | no_pid_here.db              | n/a       | kept (no match)  |
    | mixed alive+dead in dir     | both      | only dead removed|
    """

    DEAD_PID = 99999899  # very unlikely to be a live PID

    def _make_file(self, tmp_path, name: str):
        f = tmp_path / name
        f.touch()
        return f

    def test_counter_file_for_dead_pid_is_deleted(self, tmp_path):
        f = self._make_file(tmp_path, f"counter_{self.DEAD_PID}.db")
        with patch(
            "litellm.proxy.prometheus_cleanup._is_pid_alive", return_value=False
        ):
            count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 1
        assert not f.exists()

    def test_histogram_file_for_dead_pid_is_deleted(self, tmp_path):
        f = self._make_file(tmp_path, f"histogram_{self.DEAD_PID}.db")
        with patch(
            "litellm.proxy.prometheus_cleanup._is_pid_alive", return_value=False
        ):
            count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 1
        assert not f.exists()

    def test_summary_file_for_dead_pid_is_deleted(self, tmp_path):
        f = self._make_file(tmp_path, f"summary_{self.DEAD_PID}.db")
        with patch(
            "litellm.proxy.prometheus_cleanup._is_pid_alive", return_value=False
        ):
            count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 1
        assert not f.exists()

    def test_gauge_all_file_for_dead_pid_is_deleted(self, tmp_path):
        f = self._make_file(tmp_path, f"gauge_all_{self.DEAD_PID}.db")
        with patch(
            "litellm.proxy.prometheus_cleanup._is_pid_alive", return_value=False
        ):
            count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 1
        assert not f.exists()

    def test_gauge_livesum_file_for_dead_pid_is_deleted(self, tmp_path):
        f = self._make_file(tmp_path, f"gauge_livesum_{self.DEAD_PID}.db")
        with patch(
            "litellm.proxy.prometheus_cleanup._is_pid_alive", return_value=False
        ):
            count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 1
        assert not f.exists()

    def test_file_for_alive_pid_is_kept(self, tmp_path):
        alive_pid = os.getpid()
        f = self._make_file(tmp_path, f"counter_{alive_pid}.db")
        count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 0
        assert f.exists()

    def test_gauge_livesum_file_for_alive_pid_is_kept(self, tmp_path):
        alive_pid = os.getpid()
        f = self._make_file(tmp_path, f"gauge_livesum_{alive_pid}.db")
        count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 0
        assert f.exists()

    def test_file_without_pid_pattern_is_kept(self, tmp_path):
        f = self._make_file(tmp_path, "no_pid_here.db")
        with patch(
            "litellm.proxy.prometheus_cleanup._is_pid_alive", return_value=False
        ):
            count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 0
        assert f.exists()

    def test_mixed_alive_and_dead_pids(self, tmp_path):
        alive_pid = os.getpid()
        dead_file = self._make_file(tmp_path, f"counter_{self.DEAD_PID}.db")
        alive_file = self._make_file(tmp_path, f"counter_{alive_pid}.db")
        with patch(
            "litellm.proxy.prometheus_cleanup._is_pid_alive",
            side_effect=lambda p: p == alive_pid,
        ):
            count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 1
        assert not dead_file.exists()
        assert alive_file.exists()

    def test_multiple_dead_files_all_deleted(self, tmp_path):
        files = [
            self._make_file(tmp_path, f"counter_{self.DEAD_PID}.db"),
            self._make_file(tmp_path, f"histogram_{self.DEAD_PID}.db"),
            self._make_file(tmp_path, f"gauge_livesum_{self.DEAD_PID}.db"),
            self._make_file(tmp_path, f"gauge_all_{self.DEAD_PID}.db"),
            self._make_file(tmp_path, f"summary_{self.DEAD_PID}.db"),
        ]
        with patch(
            "litellm.proxy.prometheus_cleanup._is_pid_alive", return_value=False
        ):
            count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 5
        for f in files:
            assert not f.exists()

    def test_oserror_on_remove_is_caught(self, tmp_path):
        self._make_file(tmp_path, f"counter_{self.DEAD_PID}.db")
        with patch(
            "litellm.proxy.prometheus_cleanup._is_pid_alive", return_value=False
        ):
            with patch("os.remove", side_effect=OSError("disk full")):
                count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 0

    def test_empty_directory_returns_zero(self, tmp_path):
        count = cleanup_stale_worker_files(str(tmp_path))
        assert count == 0


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
