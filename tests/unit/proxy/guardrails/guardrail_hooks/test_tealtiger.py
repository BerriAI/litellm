"""Unit tests for the TealTiger guardrail. Nothing on the other side of the call: no network, no API keys."""

import json
import re
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from litellm.types.guardrails import GuardrailEventHooks
from pydantic import ValidationError

from litellm.proxy.guardrails.guardrail_hooks.tealtiger import budget as budget_module
from litellm.proxy.guardrails.guardrail_hooks.tealtiger.budget import BudgetLedger
from litellm.proxy.guardrails.guardrail_hooks.tealtiger.config import parse_config
from litellm.proxy.guardrails.guardrail_hooks.tealtiger.decision import Action, RiskLevel
from litellm.proxy.guardrails.guardrail_hooks.tealtiger.engine import TealEngine, estimate_tokens
from litellm.proxy.guardrails.guardrail_hooks.tealtiger.request import (
    ANONYMOUS,
    Identity,
    identity_of,
    tool_name_of,
)
from litellm.proxy.guardrails.guardrail_hooks.tealtiger.tealtiger import TealTigerGuardrail, _to_event_hook
from litellm.types.proxy.guardrails.guardrail_hooks.tealtiger import (
    DEFAULT_POLICIES,
    CostPolicy,
    PiiPolicy,
    PolicyMode,
    TealTigerGuardrailConfigModel,
    ToolAuthPolicy,
)

_BLOCKED_TOOL = re.compile(r"TealTiger: blocked: TOOL_NOT_ALLOWLISTED \(correlation_id=[0-9a-f-]{36}\)")
_ALICE = Identity(user="alice", session="s1")


def _engine(*policies, mode: PolicyMode = PolicyMode.ENFORCE) -> TealEngine:
    return TealEngine(policies=policies, mode=mode)


# ---------- PII ----------


def test_default_policies_redact_an_ssn_and_mark_it_critical():
    decision = _engine(*DEFAULT_POLICIES).check_text("My SSN is 123-45-6789")

    assert (decision.action, decision.reason_code, decision.risk_level, decision.redacted_text) == (
        Action.REDACT,
        "PII_DETECTED:ssn",
        RiskLevel.CRITICAL,
        "My SSN is [REDACTED:SSN]",
    )


def test_email_is_redacted_at_medium_risk():
    decision = _engine(*DEFAULT_POLICIES).check_text("mail jane@example.com")

    assert (decision.reason_code, decision.risk_level, decision.redacted_text) == (
        "PII_DETECTED:email",
        RiskLevel.MEDIUM,
        "mail [REDACTED:EMAIL]",
    )


def test_clean_text_is_allowed_without_a_redaction():
    decision = _engine(*DEFAULT_POLICIES).check_text("What's the weather like today?")

    assert (decision.action, decision.reason_code, decision.risk_level, decision.redacted_text) == (
        Action.ALLOW,
        "NO_VIOLATIONS",
        RiskLevel.NONE,
        None,
    )


@pytest.mark.parametrize(
    "text",
    ["Enter your PASSWORD", "Order 1234567890 shipped", "upgrade to 1.2.3.4", "due 12/25/2024", "INTERNATIONAL"],
)
def test_ordinary_text_is_not_flagged_by_the_default_scope(text: str):
    decision = _engine(*DEFAULT_POLICIES).check_text(text)

    assert (decision.action, decision.redacted_text) == (Action.ALLOW, None), f"flagged {text!r}: {decision}"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call (555) 123-4567 now", "call [REDACTED:PHONE_US] now"),
        ("call 555-123-4567 now", "call [REDACTED:PHONE_US] now"),
        ("card 4111 1111 1111 1111", "card [REDACTED:CREDIT_CARD_VISA]"),
        ("key AKIAIOSFODNN7EXAMPLE", "key [REDACTED:AWS_ACCESS_KEY]"),
    ],
)
def test_real_pii_is_still_redacted_by_the_default_scope(text: str, expected: str):
    assert _engine(*DEFAULT_POLICIES).check_text(text).redacted_text == expected


def test_block_action_blocks_instead_of_redacting():
    decision = _engine(PiiPolicy(action="BLOCK")).check_text("My SSN is 123-45-6789")

    assert (decision.action, decision.reason_code, decision.redacted_text) == (Action.BLOCK, "PII_DETECTED:ssn", None)


def test_pattern_list_scans_only_the_listed_patterns():
    engine = _engine(PiiPolicy(patterns=("email",)))

    assert engine.check_text("SSN 123-45-6789").action == Action.ALLOW
    assert engine.check_text("mail jane@example.com").redacted_text == "mail [REDACTED:EMAIL]"


def test_category_name_scans_every_pattern_in_that_category():
    engine = _engine(PiiPolicy(patterns=("credentials",)))

    assert engine.check_text("mail jane@example.com").action == Action.ALLOW
    assert engine.check_text("key AKIAIOSFODNN7EXAMPLE").redacted_text == "key [REDACTED:AWS_ACCESS_KEY]"


def test_loose_pattern_is_available_when_named_explicitly():
    decision = _engine(PiiPolicy(patterns=("swift_bic",))).check_text("Enter your PASSWORD")

    assert decision.redacted_text == "Enter your [REDACTED:SWIFT_BIC]"


def test_monitor_mode_keeps_the_text_and_records_what_enforce_would_have_done():
    decision = _engine(*DEFAULT_POLICIES, mode=PolicyMode.MONITOR).check_text("My SSN is 123-45-6789")

    assert (decision.action, decision.reason_code, decision.redacted_text, dict(decision.details)) == (
        Action.ALLOW,
        "PII_DETECTED:ssn",
        None,
        {"types": ("ssn",), "monitor_would": "REDACT"},
    )


def test_receipt_is_typed_json_with_risk_level_reason_and_correlation_id():
    decision = _engine(*DEFAULT_POLICIES).check_text("My SSN is 123-45-6789")

    assert json.loads(decision.receipt_json()) == {
        "correlation_id": decision.correlation_id,
        "action": "REDACT",
        "reason_code": "PII_DETECTED:ssn",
        "risk_level": "CRITICAL",
        "details": {"types": ["ssn"]},
    }


# ---------- tools ----------


def test_tool_allowlist_allows_listed_and_blocks_unlisted_tools():
    engine = _engine(ToolAuthPolicy(allowlist=("search",)))

    assert engine.check_tool("search").action == Action.ALLOW
    blocked = engine.check_tool("shell_exec")
    assert (blocked.action, blocked.reason_code, blocked.risk_level, dict(blocked.details)) == (
        Action.BLOCK,
        "TOOL_NOT_ALLOWLISTED",
        RiskLevel.HIGH,
        {"tool_name": "shell_exec"},
    )


def test_tool_blocklist_wins_over_a_missing_allowlist():
    engine = _engine(ToolAuthPolicy(blocklist=("shell_exec",)))

    assert [engine.check_tool(name).action for name in ("search", "shell_exec")] == [Action.ALLOW, Action.BLOCK]


def test_no_tool_policy_allows_every_tool():
    assert _engine(*DEFAULT_POLICIES).check_tool("anything").action == Action.ALLOW


# ---------- budgets ----------


def _cost_engine(**limits: float) -> TealEngine:
    return _engine(CostPolicy(cost_per_1k_tokens=1.0, **limits))


def test_daily_budget_is_tracked_per_user():
    engine = _cost_engine(daily_limit_usd=1.0)
    engine.record_usage(_ALICE, ["x" * 4000])  # 1000 estimated tokens at $1 per 1k

    alice = engine.check_budget(_ALICE, ["hi"])
    bob = engine.check_budget(Identity(user="bob", session=None), ["hi"])

    assert (alice.action, alice.reason_code, bob.action) == (Action.BLOCK, "DAILY_BUDGET_EXCEEDED", Action.ALLOW)
    assert dict(alice.details) == {"spent_usd": 1.0, "limit_usd": 1.0, "user": "alice"}


def test_session_budget_is_tracked_per_session():
    engine = _cost_engine(session_limit_usd=1.0)
    engine.record_usage(_ALICE, ["x" * 4000])

    same_session = engine.check_budget(Identity(user="bob", session="s1"), ["hi"])
    other_session = engine.check_budget(Identity(user="alice", session="s2"), ["hi"])

    assert (same_session.reason_code, other_session.action) == ("SESSION_BUDGET_EXCEEDED", Action.ALLOW)


def test_request_budget_blocks_one_oversized_request():
    engine = _cost_engine(request_limit_usd=1.0)

    small = engine.check_budget(_ALICE, ["x" * 3996])  # 999 tokens: $0.999
    large = engine.check_budget(_ALICE, ["x" * 4000])  # 1000 tokens: $1.00

    assert (small.action, large.reason_code) == (Action.ALLOW, "REQUEST_BUDGET_EXCEEDED")


def test_no_cost_policy_means_no_budget():
    engine = _engine(*DEFAULT_POLICIES)
    engine.record_usage(_ALICE, ["x" * 4_000_000])

    assert engine.check_budget(_ALICE, ["hi"]).action == Action.ALLOW


def test_monitor_mode_allows_an_exhausted_budget_and_records_it():
    engine = _engine(CostPolicy(daily_limit_usd=1.0, cost_per_1k_tokens=1.0), mode=PolicyMode.MONITOR)
    engine.record_usage(_ALICE, ["x" * 4000])

    decision = engine.check_budget(_ALICE, ["hi"])

    assert (decision.action, decision.reason_code, decision.details["monitor_would"]) == (
        Action.ALLOW,
        "DAILY_BUDGET_EXCEEDED",
        "BLOCK",
    )


def test_ledger_resets_when_the_local_day_rolls_over():
    ledger = BudgetLedger()
    ledger.record("alice", "s1", 2.0)

    with patch.object(budget_module, "_today", return_value="2999-01-01"):
        assert ledger.spent("alice", "s1") == (0.0, 0.0)


def test_ledger_accumulates_within_a_day():
    ledger = BudgetLedger()
    ledger.record("alice", "s1", 0.25)
    ledger.record("alice", "s1", 0.5)

    assert ledger.spent("alice", "s1") == (0.75, 0.75)


def test_estimate_tokens_is_about_four_characters_per_token():
    assert [estimate_tokens(texts) for texts in ([], ["abcd"], ["abcde", "fg"])] == [0, 1, 2]


# ---------- config ----------


def test_config_defaults_to_pii_redaction_in_enforce_mode_with_no_budget():
    config = parse_config(None, None)

    assert (config.policies, config.policy_mode) == ((PiiPolicy(),), PolicyMode.ENFORCE)


def test_config_parses_a_yaml_style_policy_list_into_typed_policies():
    config = parse_config(
        [
            {"type": "pii", "action": "BLOCK", "patterns": ["email"]},
            {"type": "cost", "daily_limit_usd": 5},
            {"type": "tool_auth", "allowlist": ["search"]},
        ],
        "MONITOR",
    )

    assert config.policies == (
        PiiPolicy(action="BLOCK", patterns=("email",)),
        CostPolicy(daily_limit_usd=5.0),
        ToolAuthPolicy(allowlist=("search",)),
    )
    assert config.policy_mode == PolicyMode.MONITOR


@pytest.mark.parametrize(
    "policies",
    [
        [{"type": "nope"}],
        [{"type": "cost", "daily_limit": 5}],
        [{"type": "cost", "daily_limit_usd": -1}],
        [{"type": "pii", "patterns": "email"}],
    ],
)
def test_config_rejects_invalid_policies_at_startup(policies: list[dict[str, object]]):
    with pytest.raises(ValidationError):
        parse_config(policies, None)


def test_config_rejects_an_unknown_pattern_name_at_startup():
    with pytest.raises(ValueError, match="unknown pii pattern or category name"):
        parse_config([{"type": "pii", "patterns": ["email", "not_a_pattern"]}], None)


def test_config_rejects_an_unknown_policy_mode():
    with pytest.raises(ValidationError):
        parse_config(None, "WATCH")


# ---------- request shapes ----------


class _Function:
    name = "search"


class _ToolCallObject:
    function = _Function()


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ({"name": "search"}, "search"),
        ({"type": "function", "function": {"name": "search"}}, "search"),
        (_ToolCallObject(), "search"),
        ({"function": "not-a-mapping"}, None),
        ({}, None),
        (None, None),
        ("invalid", None),
        (12345, None),
    ],
)
def test_tool_name_of_reads_every_tool_shape(tool: object, expected: str | None):
    assert tool_name_of(tool) == expected


@pytest.mark.parametrize(
    ("request_data", "expected"),
    [
        ({"user": "alice", "litellm_session_id": "s1"}, Identity(user="alice", session="s1")),
        ({"metadata": {"user_api_key_user_id": "key-user"}}, Identity(user="key-user", session=None)),
        ({"user": "alice", "metadata": {"user_api_key_user_id": "key-user"}}, Identity(user="alice", session=None)),
        ({}, ANONYMOUS),
        ({"user": 42}, ANONYMOUS),
    ],
)
def test_identity_of_prefers_the_request_user_and_falls_back_to_anonymous(request_data: dict, expected: Identity):
    assert identity_of(request_data) == expected


# ---------- guardrail ----------


def _guardrail(*policies, mode: PolicyMode = PolicyMode.ENFORCE) -> TealTigerGuardrail:
    return TealTigerGuardrail(policies=policies or DEFAULT_POLICIES, policy_mode=mode)


@pytest.mark.asyncio
async def test_request_text_is_redacted_and_other_inputs_are_preserved():
    inputs = {"texts": ["Contact me at jane@example.com"], "images": ["img"], "model": "gpt-4"}

    result = await _guardrail().apply_guardrail(inputs=inputs, request_data={}, input_type="request")

    assert result == {"texts": ["Contact me at [REDACTED:EMAIL]"], "images": ["img"], "model": "gpt-4"}
    assert inputs["texts"] == ["Contact me at jane@example.com"], "the caller's inputs must not be mutated"


@pytest.mark.asyncio
async def test_response_text_is_redacted_too():
    result = await _guardrail().apply_guardrail(
        inputs={"texts": ["token: 123-45-6789"]}, request_data={}, input_type="response"
    )

    assert result == {"texts": ["token: [REDACTED:SSN]"]}


@pytest.mark.asyncio
async def test_block_policy_raises_a_400_with_the_reason_and_correlation_id():
    guardrail = _guardrail(PiiPolicy(action="BLOCK"))

    with pytest.raises(HTTPException) as blocked:
        await guardrail.apply_guardrail(inputs={"texts": ["SSN 123-45-6789"]}, request_data={}, input_type="request")

    assert blocked.value.status_code == 400
    assert re.fullmatch(r"TealTiger: blocked: PII_DETECTED:ssn \(correlation_id=[0-9a-f-]{36}\)", blocked.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inputs",
    [
        {"texts": ["hi"], "tools": [{"type": "function", "function": {"name": "shell_exec"}}]},
        {"texts": ["hi"], "tools": [{"name": "shell_exec"}]},
        {"texts": ["hi"], "tool_calls": [{"name": "shell_exec"}]},
        {"texts": ["hi"], "tool_calls": [_ToolCallObject()]},
    ],
)
async def test_unlisted_tool_in_a_request_is_blocked(inputs: dict):
    guardrail = _guardrail(ToolAuthPolicy(allowlist=("other",)))

    with pytest.raises(HTTPException) as blocked:
        await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")

    assert (blocked.value.status_code, bool(_BLOCKED_TOOL.fullmatch(blocked.value.detail))) == (400, True)


@pytest.mark.asyncio
async def test_unlisted_tool_call_in_a_response_is_blocked():
    guardrail = _guardrail(ToolAuthPolicy(allowlist=("search",)))

    with pytest.raises(HTTPException) as blocked:
        await guardrail.apply_guardrail(
            inputs={"texts": [], "tool_calls": [{"name": "shell_exec"}]}, request_data={}, input_type="response"
        )

    assert _BLOCKED_TOOL.fullmatch(blocked.value.detail)


@pytest.mark.asyncio
async def test_allowlisted_tool_definition_passes():
    guardrail = _guardrail(ToolAuthPolicy(allowlist=("search",)))
    inputs = {"texts": ["hi"], "tools": [{"type": "function", "function": {"name": "search"}}]}

    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")

    assert result == {"texts": ["hi"], "tools": [{"type": "function", "function": {"name": "search"}}]}


@pytest.mark.asyncio
async def test_monitor_mode_never_raises_and_leaves_text_untouched():
    guardrail = _guardrail(PiiPolicy(action="BLOCK"), ToolAuthPolicy(allowlist=("other",)), mode=PolicyMode.MONITOR)
    inputs = {"texts": ["SSN 123-45-6789"], "tools": [{"name": "shell_exec"}], "tool_calls": [{"name": "shell_exec"}]}

    result = await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request")

    assert result == inputs


@pytest.mark.asyncio
async def test_traffic_accrues_spend_per_user_until_the_daily_budget_blocks_that_user_only():
    guardrail = _guardrail(CostPolicy(daily_limit_usd=0.01, cost_per_1k_tokens=1.0))
    big_request = {"texts": ["x" * 40_000]}  # about 10k tokens: $10 against a $0.01 limit

    await guardrail.apply_guardrail(inputs=big_request, request_data={"user": "alice"}, input_type="request")

    with pytest.raises(HTTPException) as blocked:
        await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data={"user": "alice"}, input_type="request")
    bob = await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data={"user": "bob"}, input_type="request")

    assert blocked.value.detail.startswith("TealTiger: blocked: DAILY_BUDGET_EXCEEDED")
    assert bob == {"texts": ["hi"]}


@pytest.mark.asyncio
async def test_response_text_also_accrues_spend():
    guardrail = _guardrail(CostPolicy(daily_limit_usd=1.0, cost_per_1k_tokens=1.0))

    await guardrail.apply_guardrail(
        inputs={"texts": ["y" * 4000]}, request_data={"user": "alice"}, input_type="response"
    )

    with pytest.raises(HTTPException) as blocked:
        await guardrail.apply_guardrail(inputs={"texts": ["hi"]}, request_data={"user": "alice"}, input_type="request")
    assert blocked.value.detail.startswith("TealTiger: blocked: DAILY_BUDGET_EXCEEDED")


@pytest.mark.asyncio
async def test_every_non_clean_decision_is_logged_as_a_receipt():
    guardrail = _guardrail(PiiPolicy())

    with patch("litellm.proxy.guardrails.guardrail_hooks.tealtiger.tealtiger.verbose_proxy_logger") as logger:
        await guardrail.apply_guardrail(
            inputs={"texts": ["SSN 123-45-6789", "hello"]}, request_data={}, input_type="request"
        )

    (info_call,) = logger.info.call_args_list
    template, receipt_json = info_call.args
    receipt = json.loads(receipt_json)
    assert (template, receipt["action"], receipt["reason_code"], receipt["risk_level"]) == (
        "tealtiger.decision_receipt=%s",
        "REDACT",
        "PII_DETECTED:ssn",
        "CRITICAL",
    )
    clean_reasons = [json.loads(call.args[1])["reason_code"] for call in logger.debug.call_args_list]
    assert clean_reasons == ["NO_VIOLATIONS", "NO_VIOLATIONS"], (
        "the budget check and the clean text each get a debug receipt"
    )


@pytest.mark.asyncio
async def test_monitor_mode_logs_the_pii_finding_it_does_not_enforce():
    guardrail = _guardrail(PiiPolicy(), mode=PolicyMode.MONITOR)

    with patch("litellm.proxy.guardrails.guardrail_hooks.tealtiger.tealtiger.verbose_proxy_logger") as logger:
        await guardrail.apply_guardrail(inputs={"texts": ["SSN 123-45-6789"]}, request_data={}, input_type="request")

    receipt = json.loads(logger.info.call_args.args[1])
    assert (receipt["action"], receipt["reason_code"], receipt["details"]["monitor_would"]) == (
        "ALLOW",
        "PII_DETECTED:ssn",
        "REDACT",
    )


# ---------- wiring ----------


def test_get_config_model_returns_the_tealtiger_config_model():
    assert TealTigerGuardrail.get_config_model() is TealTigerGuardrailConfigModel
    assert TealTigerGuardrailConfigModel.ui_friendly_name() == "TealTiger"


def test_registries_are_plain_dicts_so_litellm_discovers_them():
    from litellm.proxy.guardrails.guardrail_hooks import tealtiger as hook

    assert (type(hook.guardrail_initializer_registry), type(hook.guardrail_class_registry)) == (dict, dict)
    assert list(hook.guardrail_initializer_registry) == ["tealtiger"]


def test_litellm_registry_discovers_tealtiger():
    from litellm.proxy.guardrails.guardrail_registry import get_guardrail_initializer_from_hooks

    assert "tealtiger" in get_guardrail_initializer_from_hooks()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (None, None),
        (GuardrailEventHooks.pre_call, GuardrailEventHooks.pre_call),
        ("pre_call", GuardrailEventHooks.pre_call),
        (["pre_call", GuardrailEventHooks.post_call], [GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call]),
    ],
)
def test_to_event_hook_converts_litellm_params_mode(mode, expected):
    assert _to_event_hook(mode) == expected
