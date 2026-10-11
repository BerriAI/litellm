"""
Unit tests for the TealTiger guardrail.

Run with: pytest tests/unit/proxy/guardrails/guardrail_hooks/test_tealtiger.py -v

These import from the vendored litellm/proxy/guardrails/guardrail_hooks/
tealtiger/ package. Placed inside an actual litellm checkout, no path
changes should be needed since the tests import via the standard package path.
"""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.tealtiger.engine import (
    Action,
    PolicyMode,
    TealEngine,
    estimate_tokens,
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

from litellm.types.guardrails import GuardrailEventHooks

from litellm.proxy.guardrails.guardrail_hooks.tealtiger.tealtiger import (
    _to_event_hook,
    _tool_call_name,
)


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


# ---------- additional edge-case coverage ----------


def test_engine_no_pii_match_leaves_redacted_text_none():
    engine = make_engine()
    decision = engine.evaluate_text("Contact me at clean_user_handle")
    assert decision.action == Action.ALLOW.value
    assert decision.redacted_text is None


def test_engine_tool_auth_blocklist_only():
    policies = [
        {"type": "tool_auth", "action": "ENFORCE", "blocklist": ["dangerous_exec", "terminal"]},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    assert engine.check_tool("calculator") is True
    assert engine.check_tool("dangerous_exec") is False


def test_engine_tool_auth_no_allowlist_or_blocklist_allows_everything():
    policies = [
        {"type": "tool_auth", "action": "ENFORCE"},
    ]
    engine = TealEngine(policies=policies, mode=PolicyMode.ENFORCE)
    assert engine.check_tool("any_tool") is True


def test_engine_budget_tracking_stays_under_limit():
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
async def test_apply_guardrail_budget_exceeded_raises_error():
    policies = [
        {"type": "cost", "action": "ENFORCE", "daily_limit_usd": 0.00001},
    ]
    guardrail = TealTigerGuardrail(policies=policies, policy_mode="ENFORCE")
    guardrail.engine.track_cost(tokens=1_000_000)

    inputs = {"texts": ["hello"], "images": [], "tool_calls": []}

    with pytest.raises(ValueError, match="BUDGET_EXCEEDED"):
        await guardrail.apply_guardrail(
            inputs=inputs,
            request_data={"model": "gpt-4", "user": "test-user"},
            input_type="request",
        )


@pytest.mark.asyncio
async def test_apply_guardrail_redacts_response_text():
    guardrail = TealTigerGuardrail(policies=DEFAULT_POLICIES, policy_mode="ENFORCE")
    inputs = {"texts": ["Found secret token: 123-45-6789"], "images": [], "tool_calls": []}

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={"model": "gpt-4"},
        input_type="response",
    )
    assert "123-45-6789" not in result["texts"][0]


def test_tool_call_name_handles_invalid_types():
    assert _tool_call_name(None) is None
    assert _tool_call_name("invalid_str") is None
    assert _tool_call_name(12345) is None


_TOOL_POLICIES = [
    {"type": "tool_auth", "action": "ENFORCE", "allowlist": ["search"]},
]


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_disallowed_tool_definition_openai_shape():
    guardrail = TealTigerGuardrail(policies=_TOOL_POLICIES, policy_mode="ENFORCE")
    inputs = {"texts": ["hi"], "tools": [{"type": "function", "function": {"name": "shell_exec"}}]}

    with pytest.raises(ValueError, match="TOOL_NOT_ALLOWLISTED"):
        await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_disallowed_tool_definition_anthropic_shape():
    guardrail = TealTigerGuardrail(policies=_TOOL_POLICIES, policy_mode="ENFORCE")
    inputs = {"texts": ["hi"], "tools": [{"name": "shell_exec"}]}

    with pytest.raises(ValueError, match="TOOL_NOT_ALLOWLISTED"):
        await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")


@pytest.mark.asyncio
async def test_apply_guardrail_allows_allowlisted_tool_definition():
    guardrail = TealTigerGuardrail(policies=_TOOL_POLICIES, policy_mode="ENFORCE")
    inputs = {"texts": ["hi"], "tools": [{"type": "function", "function": {"name": "search"}}]}

    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")
    assert result["texts"] == ["hi"]


@pytest.mark.asyncio
async def test_apply_guardrail_blocks_disallowed_tool_call_in_response():
    guardrail = TealTigerGuardrail(policies=_TOOL_POLICIES, policy_mode="ENFORCE")
    inputs = {"texts": [], "tool_calls": [{"name": "shell_exec"}]}

    with pytest.raises(ValueError, match="TOOL_NOT_ALLOWLISTED"):
        await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="response")


@pytest.mark.asyncio
async def test_monitor_mode_does_not_block_disallowed_tool():
    guardrail = TealTigerGuardrail(policies=_TOOL_POLICIES, policy_mode="MONITOR")
    inputs = {"texts": ["hi"], "tools": [{"name": "shell_exec"}], "tool_calls": [{"name": "shell_exec"}]}

    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")
    assert result["texts"] == ["hi"]


@pytest.mark.asyncio
async def test_monitor_mode_does_not_block_exhausted_budget():
    policies = [{"type": "cost", "action": "ENFORCE", "daily_limit_usd": 0.00001}]
    guardrail = TealTigerGuardrail(policies=policies, policy_mode="MONITOR")
    guardrail.engine.track_cost(tokens=1_000_000)

    result = await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data={}, input_type="request")
    assert result["texts"] == ["hi"]


@pytest.mark.asyncio
async def test_budget_accrues_from_traffic_and_eventually_blocks():
    policies = [{"type": "cost", "action": "ENFORCE", "daily_limit_usd": 0.01, "cost_per_1k_tokens": 1.0}]
    guardrail = TealTigerGuardrail(policies=policies, policy_mode="ENFORCE")
    big_text = "x" * 40_000  # about 10k tokens, about $10 at $1 per 1k tokens

    await guardrail.apply_guardrail(inputs={"texts": [big_text]}, request_data={}, input_type="request")
    over_budget, spent, _ = guardrail.engine.check_budget()
    assert over_budget is True
    assert spent > 0

    with pytest.raises(ValueError, match="DAILY_BUDGET_EXCEEDED"):
        await guardrail.apply_guardrail(inputs={"texts": ["next"]}, request_data={}, input_type="request")


@pytest.mark.asyncio
async def test_budget_accrues_on_response_path():
    policies = [{"type": "cost", "action": "ENFORCE", "daily_limit_usd": 100.0, "cost_per_1k_tokens": 1.0}]
    guardrail = TealTigerGuardrail(policies=policies, policy_mode="ENFORCE")

    await guardrail.apply_guardrail(inputs={"texts": ["y" * 4000]}, request_data={}, input_type="response")
    _, spent, _ = guardrail.engine.check_budget()
    assert spent == pytest.approx(1.0)


def test_estimate_tokens():
    assert estimate_tokens([]) == 0
    assert estimate_tokens(["abcd"]) == 1
    assert estimate_tokens(["abcde", "fg"]) == 2


def test_pattern_scope_list_only_scans_listed_patterns():
    engine = TealEngine(policies=[{"type": "pii", "action": "REDACT", "patterns": ["email"]}])

    ssn = engine.evaluate_text("SSN 123-45-6789")
    assert ssn.action == Action.ALLOW.value

    email = engine.evaluate_text("mail jane@example.com")
    assert email.action == Action.REDACT.value
    assert "jane@example.com" not in (email.redacted_text or "")


def test_pattern_scope_accepts_category_names():
    engine = TealEngine(policies=[{"type": "pii", "action": "REDACT", "patterns": ["credentials"]}])

    assert engine.evaluate_text("mail jane@example.com").action == Action.ALLOW.value
    assert engine.evaluate_text("key AKIAIOSFODNN7EXAMPLE").action == Action.REDACT.value


def test_pattern_scope_unknown_name_fails_at_startup():
    with pytest.raises(ValueError, match="unknown pii pattern"):
        TealEngine(policies=[{"type": "pii", "action": "REDACT", "patterns": ["not_a_pattern"]}])


def test_pattern_scope_invalid_type_fails_at_startup():
    with pytest.raises(TypeError, match="must be 'all' or a list"):
        TealEngine(policies=[{"type": "pii", "action": "REDACT", "patterns": 5}])


def test_cost_rate_read_from_policy_and_defaults():
    custom = TealEngine(policies=[{"type": "cost", "daily_limit_usd": 1.0, "cost_per_1k_tokens": 0.5}])
    assert custom.cost_per_1k_tokens == 0.5
    default = TealEngine(policies=[{"type": "cost", "daily_limit_usd": 1.0}])
    assert default.cost_per_1k_tokens == 0.002


def test_registries_are_plain_dicts_so_litellm_discovers_them():
    from litellm.proxy.guardrails.guardrail_hooks import tealtiger as hook

    assert isinstance(hook.guardrail_initializer_registry, dict)
    assert isinstance(hook.guardrail_class_registry, dict)
    assert "tealtiger" in hook.guardrail_initializer_registry


def test_litellm_registry_discovers_tealtiger():
    from litellm.proxy.guardrails.guardrail_registry import get_guardrail_initializer_from_hooks

    assert "tealtiger" in get_guardrail_initializer_from_hooks()
