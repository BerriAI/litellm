"""Unit tests for Mavvrik Logger marker class."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.logger import Logger
from litellm.integrations.custom_logger import CustomLogger


class TestLogger:
    def test_is_custom_logger_subclass(self):
        assert issubclass(Logger, CustomLogger)

    def test_can_be_instantiated(self):
        logger = Logger()
        assert logger is not None

    def test_registered_as_mavvrik_callback(self):
        """Logger is registered as the 'mavvrik' callback in custom_logger_registry."""
        assert Logger is not None
