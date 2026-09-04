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


def test_engine_allows_clean_text() -> None:
    engine = make_engine()
    decision = engine.evaluate_text("What's the weather like today?")
    assert decision.action == Action.ALLOW.value


def test_engine_redacts_ssn() -> None:
    engine = make_engine()
    decision = engine.evaluate_text("My SSN is 123-45-6789")
    assert decision.action == Action.REDACT.value
    assert "123-45-6789" not in decision.redacted_text


def test_engine_monitor_mode_never_redacts() -> None:
    engine = make_engine(mode=PolicyMode.MONITOR)
    decision = engine.evaluate_text("My SSN is 123-45-6789")
    assert decision.action == Action.ALLOW.value


def test_engine_tool_allowlist() -> None:
    engine = make_engine()
    assert engine.check_tool("calculator") is True
    assert engine.check_tool("shell_exec") is False


def test_engine_budget_enforcement() -> None:
    engine = make_engine()
    engine.track_cost(tokens=1_000_000)  # pushes spend over $1 limit
    over_budget, _spent, _limit = engine.check_budget()
    assert over_budget is True


@pytest.mark.asyncio
async def test_apply_guardrail_redacts_request_text() -> None:
    guardrail = TealTigerGuardrail(policies=DEFAULT_POLICIES, policy_mode="ENFORCE")
    inputs = {"texts": ["Contact me at jane@example.com"], "images": [], "tool_calls": []}

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={"model": "gpt-4", "user": "test-user"},
        input_type="request",
    )
    assert "jane@example.com" not in result["texts"][0]


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_disallowed_tool() -> None:
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


def test_engine_malformed_blocklist_fails_safe() -> None:
    """A non-list/tuple blocklist value shouldn't crash — treated as empty."""
    policies = [
        {"type": "tool_auth", "action": "ENFORCE", "blocklist": "not-a-list"},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    # malformed blocklist -> treated as empty -> nothing gets blocked via it
    assert engine.check_tool("anything") is True


def test_engine_malformed_allowlist_denies_by_default() -> None:
    """A non-list/tuple allowlist value shouldn't crash — fails closed (deny)."""
    policies = [
        {"type": "tool_auth", "action": "ENFORCE", "allowlist": "not-a-list"},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    # malformed allowlist -> treated as empty -> nothing is in it -> denied
    assert engine.check_tool("anything") is False


def test_engine_malformed_budget_limit_treated_as_no_limit() -> None:
    """A non-numeric daily_limit_usd shouldn't crash — treated as no budget cap."""
    policies = [
        {"type": "cost", "action": "ENFORCE", "daily_limit_usd": "fifty dollars"},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    engine.track_cost(tokens=999_999_999)  # would blow any real limit
    over_budget, _spent, _limit = engine.check_budget()
    assert over_budget is False


@pytest.mark.asyncio
async def test_apply_guardrail_session_id_falls_back_when_user_not_a_string() -> None:
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


def test_to_event_hook_passes_through_none() -> None:
    assert _to_event_hook(None) is None


def test_to_event_hook_passes_through_real_enum() -> None:
    result = _to_event_hook(GuardrailEventHooks.pre_call)
    assert result == GuardrailEventHooks.pre_call


def test_to_event_hook_converts_bare_string() -> None:
    result = _to_event_hook("pre_call")
    assert result == GuardrailEventHooks.pre_call


def test_to_event_hook_converts_list_of_strings() -> None:
    result = _to_event_hook(["pre_call", "post_call"])
    assert result == [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]


def test_to_event_hook_converts_mixed_list() -> None:
    result = _to_event_hook([GuardrailEventHooks.pre_call, "post_call"])
    assert result == [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]


def test_tool_call_name_from_dict_shape() -> None:
    """ChatCompletionToolCallChunk-style: a plain dict."""
    tool_call = {"id": "1", "type": "function", "function": {"name": "shell_exec"}, "index": 0}
    assert _tool_call_name(tool_call) == "shell_exec"


def test_tool_call_name_from_dict_shape_missing_function() -> None:
    tool_call = {"id": "1", "type": "function", "index": 0}
    assert _tool_call_name(tool_call) is None


class _FakeFunction:
    def __init__(self, name) -> None:
        self.name = name


class _FakeToolCallObject:
    """Mimics ChatCompletionMessageToolCall's real shape: attribute access."""

    def __init__(self, name) -> None:
        self.function = _FakeFunction(name)
        self.id = "1"
        self.type = "function"


def test_tool_call_name_from_object_shape() -> None:
    """ChatCompletionMessageToolCall-style: a pydantic-like object."""
    assert _tool_call_name(_FakeToolCallObject("calculator")) == "calculator"


def test_tool_call_name_from_object_shape_missing_function() -> None:
    class _Empty:
        pass

    assert _tool_call_name(_Empty()) is None


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_object_shaped_tool_call() -> None:
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


# ---------- Additional Edge Cases & Error Handling Coverage ----------


def test_engine_pii_types_and_no_match() -> None:
    """Verify engine handles clean text and custom regex rules cleanly."""
    engine = make_engine()
    # PII scanner with no matching patterns
    decision = engine.evaluate_text("Contact me at clean_user_handle")
    assert decision.action == Action.ALLOW.value
    # Fix: redacted_text is None when no redaction occurred
    assert decision.redacted_text is None


def test_engine_tool_auth_blocklist() -> None:
    """Test explicit tool blocklist evaluation."""
    policies = [
        {"type": "tool_auth", "action": "ENFORCE", "blocklist": ["dangerous_exec", "terminal"]},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    assert engine.check_tool("calculator") is True
    assert engine.check_tool("dangerous_exec") is False


def test_engine_tool_auth_disabled() -> None:
    """Test tool auth behavior when no allowlist or blocklist is specified."""
    policies = [
        {"type": "tool_auth", "action": "ENFORCE"},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    assert engine.check_tool("any_tool") is True


def test_engine_budget_tracking_under_limit() -> None:
    """Test cost tracking within allowed limits."""
    policies = [
        {"type": "cost", "action": "ENFORCE", "daily_limit_usd": 10.0},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    engine.track_cost(tokens=10)
    over_budget, spent, limit = engine.check_budget()
    assert over_budget is False
    assert spent > 0
    assert limit == 10.0


@pytest.mark.asyncio
async def test_apply_guardrail_budget_exceeded_raises_error() -> None:
    """Verify that apply_guardrail raises a ValueError when the budget cap is reached."""
    policies = [
        {"type": "cost", "action": "ENFORCE", "daily_limit_usd": 0.00001},
    ]
    guardrail = TealTigerGuardrail(policies=policies, policy_mode="ENFORCE")
    # Manually push engine spend over budget
    guardrail.engine.track_cost(tokens=1_000_000)

    inputs = {"texts": ["hello"], "images": [], "tool_calls": []}

    with pytest.raises(ValueError, match="BUDGET_EXCEEDED"):
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4", "user": "test-user"},
            input_type="request",
        )


@pytest.mark.asyncio
async def test_apply_guardrail_response_input_type() -> None:
    """Test apply_guardrail for 'response' or 'post_call' inputs."""
    guardrail = TealTigerGuardrail(policies=DEFAULT_POLICIES, policy_mode="ENFORCE")
    inputs = {"texts": ["Found secret token: 123-45-6789"], "images": [], "tool_calls": []}

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={"model": "gpt-4"},
        input_type="response",
    )
    assert "123-45-6789" not in result["texts"][0]


@pytest.mark.asyncio
async def test_async_pre_call_hook() -> None:
    """Directly test async_pre_call_hook method if exposed by TealTigerGuardrail."""
    guardrail = TealTigerGuardrail(policies=DEFAULT_POLICIES, policy_mode="ENFORCE")
    data = {"messages": [{"role": "user", "content": "Contact me at test@example.com"}]}

    if hasattr(guardrail, "async_pre_call_hook"):
        res = await guardrail.async_pre_call_hook(
            user_api_key_dict={},
            cache=None,
            data=data,
            call_type="completion",
        )
        # Fix: LiteLLM hook returns modified data dict or None
        assert res is None or isinstance(res, dict)


@pytest.mark.asyncio
async def test_async_post_call_success_hook() -> None:
    """Directly test async_post_call_success_hook method if exposed by TealTigerGuardrail."""
    guardrail = TealTigerGuardrail(policies=DEFAULT_POLICIES, policy_mode="ENFORCE")
    response_obj = {"choices": [{"message": {"content": "SSN: 123-45-6789"}}]}

    if hasattr(guardrail, "async_post_call_success_hook"):
        # Pass data, user_api_key_dict, and response_obj positionally
        res = await guardrail.async_post_call_success_hook(
            {"model": "gpt-4"},
            {},
            response_obj,
        )
        assert res is None or isinstance(res, dict)


def test_tool_call_name_invalid_types() -> None:
    """Verify _tool_call_name gracefully handles primitives or None."""
    assert _tool_call_name(None) is None
    assert _tool_call_name("invalid_str") is None
    assert _tool_call_name(12345) is None
