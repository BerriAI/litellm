"""
Tests for LITELLM_LOG env var handling in debug_utils.init_verbose_loggers().

Verifies that WARNING, ERROR, and CRITICAL levels are respected in addition
to the previously-supported DEBUG and INFO levels.
"""

import json
import logging
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm._logging import (
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)
from litellm.proxy.common_utils.debug_utils import init_verbose_loggers


@pytest.fixture(autouse=True)
def _restore_log_levels():
    """Save and restore logger levels around each test."""
    orig = (
        verbose_logger.level,
        verbose_proxy_logger.level,
        verbose_router_logger.level,
    )
    yield
    verbose_logger.setLevel(orig[0])
    verbose_proxy_logger.setLevel(orig[1])
    verbose_router_logger.setLevel(orig[2])


def _worker_config_json(debug=False, detailed_debug=False):
    """Return a WORKER_CONFIG JSON string with debug flags set to False."""
    return json.dumps({"debug": debug, "detailed_debug": detailed_debug})


@pytest.mark.parametrize(
    "log_level_str, expected_level",
    [
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
    ],
)
def test_litellm_log_env_sets_logger_levels(
    monkeypatch, log_level_str, expected_level
):
    """Test that LITELLM_LOG env var correctly sets logger levels."""
    monkeypatch.setenv("LITELLM_LOG", log_level_str)

    with patch(
        "litellm.proxy.common_utils.debug_utils.get_secret_str",
        return_value=_worker_config_json(),
    ):
        init_verbose_loggers()

    assert verbose_router_logger.level == expected_level
    assert verbose_proxy_logger.level == expected_level
