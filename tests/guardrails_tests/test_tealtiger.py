"""
Unit tests for the TealTiger guardrail.

Run with: pytest tests/guardrails_tests/test_tealtiger.py -v

These import from the vendored litellm/proxy/guardrails/guardrail_hooks/
tealtiger/ package. Placed inside an actual litellm checkout, no path
changes should be needed since the tests import via the standard package path.
"""
import pytest

from litellm.proxy.guardrails.guardrail_hooks.tealtiger.engine import (
    Action,
    PolicyMode,
    TealEngine,
)
from litellm.proxy.guardrails.guardrail_hooks.tealtiger.tealtiger import (
    DEFAULT_POLICIES,
    TealTigerGuardrail,
)


def make_engine(mode=PolicyMode.ENFORCE):
    policies = [
        {"type": "pii", "action": "REDACT", "patterns": "all"},
        {"type": "cost", "action": "ENFORCE", "daily_limit_usd": 1.0},
        {"type": "tool_auth", "action": "ENFORCE", "allowlist": ["search", "calculator"]},
    ]
    return TealEngine(policies=policies, mode=mode)


def test_engine_allows_clean_text():
    engine = make_engine()
    decision = engine.evaluate_text("What's the weather like today?")
    assert decision.action == Action.ALLOW.value


def test_engine_redacts_ssn():
    engine = make_engine()
    decision = engine.evaluate_text("My SSN is 123-45-6789")
    assert decision.action == Action.REDACT.value
    assert "123-45-6789" not in decision.redacted_text


def test_engine_monitor_mode_never_redacts():
    engine = make_engine(mode=PolicyMode.MONITOR)
    decision = engine.evaluate_text("My SSN is 123-45-6789")
    assert decision.action == Action.ALLOW.value


def test_engine_tool_allowlist():
    engine = make_engine()
    assert engine.check_tool("calculator") is True
    assert engine.check_tool("shell_exec") is False


def test_engine_budget_enforcement():
    engine = make_engine()
    engine.track_cost(tokens=1_000_000)  # pushes spend over $1 limit
    over_budget, spent, limit = engine.check_budget()
    assert over_budget is True


@pytest.mark.asyncio
async def test_apply_guardrail_redacts_request_text():
    guardrail = TealTigerGuardrail(policies=DEFAULT_POLICIES, policy_mode="ENFORCE")
    inputs = {"texts": ["Contact me at jane@example.com"], "images": [], "tool_calls": []}

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={"model": "gpt-4", "user": "test-user"},
        input_type="request",
    )
    assert "jane@example.com" not in result["texts"][0]


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_disallowed_tool():
    policies = [
        {"type": "pii", "action": "REDACT", "patterns": "all"},
        {"type": "tool_auth", "action": "ENFORCE", "allowlist": ["search"]},
    ]
    guardrail = TealTigerGuardrail(policies=policies, policy_mode="ENFORCE")
    inputs = {
        "texts": ["run it"],
        "images": [],
        "tool_calls": [{"name": "shell_exec"}],
    }

    with pytest.raises(ValueError, match="TOOL_NOT_ALLOWLISTED"):
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4"},
            input_type="request",
        )
