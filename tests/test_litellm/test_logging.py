import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path
import io
import logging
import sys
import unittest
from contextlib import redirect_stdout

import litellm
from litellm._logging import (
    ALL_LOGGERS,
    _initialize_loggers_with_handler,
    verbose_logger,
    verbose_proxy_logger,
    verbose_router_logger,
)


def test_json_mode_emits_one_record_per_logger(capfd):
    # Turn on JSON logging
    litellm._logging._turn_on_json()
    # Make sure our loggers will emit INFO-level records
    for lg in (verbose_logger, verbose_router_logger, verbose_proxy_logger):
        lg.setLevel(logging.INFO)

    # Log one message from each logger at different levels
    verbose_logger.info("first info")
    verbose_router_logger.info("second info from router")
    verbose_proxy_logger.info("third info from proxy")

    # Capture stdout
    out, err = capfd.readouterr()
    print("out", out)
    print("err", err)
    lines = [l for l in err.splitlines() if l.strip()]

    # Expect exactly three JSON lines
    assert len(lines) == 3, f"got {len(lines)} lines, want 3: {lines!r}"

    # Each line must be valid JSON with the required fields
    for line in lines:
        obj = json.loads(line)
        assert "message" in obj, "`message` key missing"
        assert "level" in obj, "`level` key missing"
        assert "timestamp" in obj, "`timestamp` key missing"


def test_initialize_loggers_with_handler_sets_propagate_false():
    """
    Test that the initialize_loggers_with_handler function sets propagate to False for all loggers
    """
    # Initialize loggers with the test handler
    _initialize_loggers_with_handler(logging.StreamHandler())

    # Check that propagate is set to False for all loggers
    for logger in ALL_LOGGERS:
        assert (
            logger.propagate is False
        ), f"Logger {logger.name} has propagate set to {logger.propagate}, expected False"
