"""Tests for the v2 MCP egress transport scaffolding (the flag)."""

import pytest

from litellm.proxy._experimental.mcp_server.v2_egress import v2_egress_enabled

FLAG = "LITELLM_USE_V2_MCP_EGRESS"


def test_egress_flag_off_by_default(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert v2_egress_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on"])
def test_egress_flag_truthy_values(monkeypatch, value):
    monkeypatch.setenv(FLAG, value)
    assert v2_egress_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
def test_egress_flag_falsey_values(monkeypatch, value):
    monkeypatch.setenv(FLAG, value)
    assert v2_egress_enabled() is False
