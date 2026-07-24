# SPDX-License-Identifier: MIT
# Tests for validate_environment fix: zai provider was missing from the
# elif chain in litellm/utils.py, causing a false all-clear when ZAI_API_KEY
# is unset (fixes #32962).

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import litellm


def test_zai_missing_key_reports_missing(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    result = litellm.validate_environment(model="zai/glm-5.1")
    assert result["keys_in_environment"] is False
    assert "ZAI_API_KEY" in result["missing_keys"]


def test_zai_key_present_reports_ok(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-key-123")
    result = litellm.validate_environment(model="zai/glm-5.1")
    assert result["keys_in_environment"] is True
    assert "ZAI_API_KEY" not in result["missing_keys"]


def test_moonshot_missing_key_still_reports_missing(monkeypatch):
    """Regression: moonshot neighbour branch still works after the zai addition."""
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    result = litellm.validate_environment(model="moonshot/kimi-k2")
    assert result["keys_in_environment"] is False
    assert "MOONSHOT_API_KEY" in result["missing_keys"]
