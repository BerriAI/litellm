"""Asserts that computer use support is correctly identified for relevant models."""

import io
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import litellm
import pytest


def test_supports_computer_use():
    from litellm.utils import supports_computer_use

    # 'anthropic/claude-3-7-sonnet-20250219' is a model that should support computer_use
    supports_cu = supports_computer_use(model="anthropic/claude-3-7-sonnet-20250219")

    assert supports_cu is True
