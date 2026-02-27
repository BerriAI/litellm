"""
Tests for litellm.proxy.prometheus_cleanup.wipe_directory and
ProxyInitializationHelpers._maybe_setup_prometheus_multiproc_dir.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from litellm.proxy.prometheus_cleanup import wipe_directory
from litellm.proxy.proxy_cli import ProxyInitializationHelpers


class TestWipeDirectory:
    def test_deletes_all_db_files(self, tmp_path):
        (tmp_path / "counter_1234.db").touch()
        (tmp_path / "histogram_5678.db").touch()
        (tmp_path / "gauge_livesum_9999.db").touch()
        wipe_directory(str(tmp_path))
        assert not list(tmp_path.glob("*.db"))


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
