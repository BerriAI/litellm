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


def test_engine_malformed_blocklist_fails_safe():
    """A non-list/tuple blocklist value shouldn't crash — treated as empty."""
    policies = [
        {"type": "tool_auth", "action": "ENFORCE", "blocklist": "not-a-list"},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    # malformed blocklist -> treated as empty -> nothing gets blocked via it
    assert engine.check_tool("anything") is True


def test_engine_malformed_allowlist_denies_by_default():
    """A non-list/tuple allowlist value shouldn't crash — fails closed (deny)."""
    policies = [
        {"type": "tool_auth", "action": "ENFORCE", "allowlist": "not-a-list"},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    # malformed allowlist -> treated as empty -> nothing is in it -> denied
    assert engine.check_tool("anything") is False


def test_engine_malformed_budget_limit_treated_as_no_limit():
    """A non-numeric daily_limit_usd shouldn't crash — treated as no budget cap."""
    policies = [
        {"type": "cost", "action": "ENFORCE", "daily_limit_usd": "fifty dollars"},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    engine.track_cost(tokens=999_999_999)  # would blow any real limit
    over_budget, spent, limit = engine.check_budget()
    assert over_budget is False


@pytest.mark.asyncio
async def test_apply_guardrail_session_id_falls_back_when_user_not_a_string():
    """A non-string 'user' in request_data shouldn't crash session_id handling."""
    policies = [
        {"type": "cost", "action": "ENFORCE", "daily_limit_usd": 1.0},
    ]
    guardrail = TealTigerGuardrail(policies=policies, policy_mode="ENFORCE")
    inputs = {"texts": ["hello"], "images": [], "tool_calls": []}
    # user is an int here, not a str -- should fall back to "default" cleanly
    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={"model": "gpt-4", "user": 12345},
        input_type="request",
    )
    assert result["texts"] == ["hello"]


# ---------- coverage for _to_event_hook and _tool_call_name helpers ----------

from litellm.proxy.guardrails.guardrail_hooks.tealtiger.tealtiger import (
    _to_event_hook,
    _tool_call_name,
)
from litellm.types.guardrails import GuardrailEventHooks


def test_to_event_hook_passes_through_none():
    assert _to_event_hook(None) is None


def test_to_event_hook_passes_through_real_enum():
    result = _to_event_hook(GuardrailEventHooks.pre_call)
    assert result == GuardrailEventHooks.pre_call


def test_to_event_hook_converts_bare_string():
    result = _to_event_hook("pre_call")
    assert result == GuardrailEventHooks.pre_call


def test_to_event_hook_converts_list_of_strings():
    result = _to_event_hook(["pre_call", "post_call"])
    assert result == [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]


def test_to_event_hook_converts_mixed_list():
    result = _to_event_hook([GuardrailEventHooks.pre_call, "post_call"])
    assert result == [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]


def test_tool_call_name_from_dict_shape():
    """ChatCompletionToolCallChunk-style: a plain dict."""
    tool_call = {"id": "1", "type": "function", "function": {"name": "shell_exec"}, "index": 0}
    assert _tool_call_name(tool_call) == "shell_exec"


def test_tool_call_name_from_dict_shape_missing_function():
    tool_call = {"id": "1", "type": "function", "index": 0}
    assert _tool_call_name(tool_call) is None


class _FakeFunction:
    def __init__(self, name):
        self.name = name


class _FakeToolCallObject:
    """Mimics ChatCompletionMessageToolCall's real shape: attribute access."""

    def __init__(self, name):
        self.function = _FakeFunction(name)
        self.id = "1"
        self.type = "function"


def test_tool_call_name_from_object_shape():
    """ChatCompletionMessageToolCall-style: a pydantic-like object."""
    assert _tool_call_name(_FakeToolCallObject("calculator")) == "calculator"


def test_tool_call_name_from_object_shape_missing_function():
    class _Empty:
        pass

    assert _tool_call_name(_Empty()) is None


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_object_shaped_tool_call():
    """The object-shaped (pydantic) tool_call variant should be checked too,
    not just the dict-shaped one."""
    policies = [
        {"type": "pii", "action": "REDACT", "patterns": "all"},
        {"type": "tool_auth", "action": "ENFORCE", "allowlist": ["search"]},
    ]
    guardrail = TealTigerGuardrail(policies=policies, policy_mode="ENFORCE")
    inputs = {
        "texts": ["run it"],
        "images": [],
        "tool_calls": [_FakeToolCallObject("shell_exec")],
    }
    with pytest.raises(ValueError, match="TOOL_NOT_ALLOWLISTED"):
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4"},
            input_type="request",
        )
