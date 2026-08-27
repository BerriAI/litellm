"""Pin ``ProxyLogging.pre_call_hook`` and ``process_pre_call_hook_response``."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import RejectedRequestError
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks


def _load(module: str, name: str):
    """The enterprise package is optional; a missing one is not an unclassified hook."""
    import importlib

    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ProxyLogging._callback_capabilities_cache.clear()
    yield
    ProxyLogging._callback_capabilities_cache.clear()


# ---------------------------------------------------------------------------
# process_pre_call_hook_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_dict_returns_response(proxy_logging):
    out = await proxy_logging.process_pre_call_hook_response(
        response={"messages": [{"x": 1}], "model": "m", "temperature": 0.5},
        data={"original": True},
        call_type="completion",
    )
    assert out == {"messages": [{"x": 1}], "model": "m", "temperature": 0.5}


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_string_completion_raises_rejected(proxy_logging):
    with pytest.raises(RejectedRequestError):
        await proxy_logging.process_pre_call_hook_response(
            response="rejected",
            data={"model": "m"},
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_string_other_call_type_raises_http(proxy_logging):
    with pytest.raises(HTTPException) as info:
        await proxy_logging.process_pre_call_hook_response(
            response="bad",
            data={},
            call_type="embeddings",
        )
    assert info.value.status_code == 400


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_exception_reraises(proxy_logging):
    err = RuntimeError("hook said no")
    with pytest.raises(RuntimeError, match="hook said no"):
        await proxy_logging.process_pre_call_hook_response(
            response=err, data={}, call_type="completion"
        )


@pytest.mark.asyncio
async def test_process_pre_call_hook_response_other_type_returns_data(proxy_logging):
    out = await proxy_logging.process_pre_call_hook_response(
        response=12345, data={"a": 1, "b": 2, "c": 3}, call_type="completion"
    )
    assert out == {"a": 1, "b": 2, "c": 3}


# ---------------------------------------------------------------------------
# pre_call_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_hook_returns_data_when_no_callbacks(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    data = {"messages": [{"role": "user", "content": "hi"}], "model": "m", "temperature": 0.7}
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=data,
        call_type="completion",
    )
    assert out is data


@pytest.mark.asyncio
async def test_pre_call_hook_returns_none_for_none_data(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=None,
        call_type="completion",
    )
    assert out is None


@pytest.mark.asyncio
async def test_pre_call_hook_invokes_pre_call_override(proxy_logging, make_user_api_key_auth, monkeypatch):
    captured: Dict[str, Any] = {}

    class _Cb(CustomLogger):
        async def async_pre_call_hook(self, **kwargs):  # type: ignore[override]
            captured.update(kwargs)
            return {"messages": [{"x": "modified"}], "model": "m", "temperature": 0.1}

    monkeypatch.setattr(litellm, "callbacks", [_Cb()])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"x": "input"}], "model": "m", "temperature": 0.1},
        call_type="completion",
    )
    snapshot = {
        "out_messages": out["messages"],
        "out_model": out["model"],
        "out_temp": out["temperature"],
        "cb_received_call_type": captured.get("call_type"),
    }
    assert snapshot == {
        "out_messages": [{"x": "modified"}],
        "out_model": "m",
        "out_temp": 0.1,
        "cb_received_call_type": "completion",
    }


@pytest.mark.asyncio
async def test_pre_call_hook_propagates_callback_error_raises(proxy_logging, make_user_api_key_auth, monkeypatch):
    class _BadCb(CustomLogger):
        async def async_pre_call_hook(self, **kwargs):  # type: ignore[override]
            raise RuntimeError("rejected")

    monkeypatch.setattr(litellm, "callbacks", [_BadCb()])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    with pytest.raises(RuntimeError, match="rejected"):
        await proxy_logging.pre_call_hook(
            user_api_key_dict=make_user_api_key_auth(),
            data={"model": "m"},
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_pre_call_hook_processes_guardrail_metadata_when_no_overrides(proxy_logging, make_user_api_key_auth, mock_callbacks_disabled):
    """Even when no callback overrides exist, ``_process_guardrail_metadata`` runs."""
    data = {"messages": [{"role": "user"}], "model": "m", "metadata": {"guardrails": ["g1"]}}
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    invoked = {}

    def fake_process(d):
        invoked["data"] = d

    proxy_logging._process_guardrail_metadata = fake_process  # type: ignore[assignment]
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=data,
        call_type="completion",
    )
    assert out is data
    assert invoked["data"] is data


@pytest.mark.asyncio
async def test_guardrails_only_skips_non_guardrail_pre_call_callbacks(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """Rate limiters and budget hooks ride this same loop; a content scan must not trip them."""
    calls: list[str] = []

    class _RateLimiterLike(CustomLogger):
        async def async_pre_call_hook(self, **kwargs):  # type: ignore[override]
            calls.append("ran")
            return None

    monkeypatch.setattr(litellm, "callbacks", [_RateLimiterLike()])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)

    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"x": "input"}], "model": "m"},
        call_type="completion",
        guardrails_only=True,
    )
    assert calls == []

    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"x": "input"}], "model": "m"},
        call_type="completion",
    )
    assert calls == ["ran"], "the default path must still run non-guardrail pre-call callbacks"


@pytest.mark.asyncio
async def test_guardrails_only_skips_the_hanging_request_alert(proxy_logging, make_user_api_key_auth, monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    alerting = MagicMock(alerting=True)
    proxy_logging.slack_alerting_instance = alerting

    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"x": "input"}], "model": "m"},
        call_type="completion",
        guardrails_only=True,
    )
    alerting.response_taking_too_long.assert_not_called()


@pytest.mark.asyncio
async def test_guardrails_only_skips_prompt_template_rewriting(proxy_logging, make_user_api_key_auth, monkeypatch):
    """A prompt template would rewrite messages, which a per-record content diff would misread."""
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    process = AsyncMock()
    monkeypatch.setattr(proxy_logging, "_process_prompt_template", process)

    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"x": "input"}], "model": "m", "prompt_id": "p1", "litellm_logging_obj": MagicMock()},
        call_type="acompletion",
        guardrails_only=True,
    )
    process.assert_not_awaited()


@pytest.mark.parametrize(
    "event_hook, expected",
    [("pre_call", True), ("post_call", False), ("during_call", False)],
)
def test_has_pre_call_guardrails_follows_the_guardrail_event_hook(proxy_logging, monkeypatch, event_hook, expected):
    """A post-call-only guardrail must not make callers pay for pre-call work."""
    from litellm.integrations.custom_guardrail import CustomGuardrail

    guardrail = CustomGuardrail(guardrail_name="g", event_hook=event_hook, default_on=True)
    monkeypatch.setattr(litellm, "callbacks", [guardrail])

    assert proxy_logging.has_pre_call_guardrails({}) is expected


def test_has_pre_call_guardrails_is_false_without_callbacks(proxy_logging, monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])

    assert proxy_logging.has_pre_call_guardrails({}) is False


def test_has_pre_call_guardrails_is_true_for_a_configured_pipeline(proxy_logging, monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])

    assert proxy_logging.has_pre_call_guardrails({"_guardrail_pipelines": ["p1"]}) is True


@pytest.mark.asyncio
async def test_default_path_still_arms_the_hanging_request_alert(proxy_logging, make_user_api_key_auth, monkeypatch):
    """Pins the other side of the gate: without the flag, the alert must still fire."""
    monkeypatch.setattr(litellm, "callbacks", [])
    alerting = MagicMock(alerting=True)
    alerting.response_taking_too_long = AsyncMock()
    proxy_logging.slack_alerting_instance = alerting

    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"x": "input"}], "model": "m"},
        call_type="completion",
    )
    alerting.response_taking_too_long.assert_called_once()


@pytest.mark.asyncio
async def test_default_path_still_applies_prompt_templates(proxy_logging, make_user_api_key_auth, monkeypatch):
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    process = AsyncMock()
    monkeypatch.setattr(proxy_logging, "_process_prompt_template", process)

    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"x": "input"}], "model": "m", "prompt_id": "p1", "litellm_logging_obj": MagicMock()},
        call_type="acompletion",
    )
    process.assert_awaited_once()


@pytest.mark.asyncio
async def test_aresponses_call_type_applies_prompt_templates_before_routing(proxy_logging, make_user_api_key_auth, monkeypatch):
    """The responses surface must process registry prompts pre-routing so credentials follow the swapped model."""
    monkeypatch.setattr(litellm, "callbacks", [])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    process = AsyncMock()
    monkeypatch.setattr(proxy_logging, "_process_prompt_template", process)

    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"input": "hi", "model": "m", "prompt_id": "p1", "litellm_logging_obj": MagicMock()},
        call_type="aresponses",
    )
    process.assert_awaited_once()


# ---------------------------------------------------------------------------
# enforces_request_content: which CustomLoggers a guardrails-only walk reaches
# ---------------------------------------------------------------------------


class _Enforcer(CustomLogger):
    """Stands in for detect_prompt_injection: judges the payload, so batch records need it."""

    enforces_request_content = True

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.calls += 1
        return data


class _Accountant(CustomLogger):
    """Stands in for a rate limiter: counts a request, so it must not see records."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        self.calls += 1
        return data


@pytest.mark.asyncio
@pytest.mark.parametrize("guardrails_only", [False, True])
async def test_a_content_enforcer_runs_in_both_walks(proxy_logging, monkeypatch, guardrails_only):
    enforcer = _Enforcer()
    monkeypatch.setattr(litellm, "callbacks", [enforcer])

    await proxy_logging.pre_call_hook(
        user_api_key_dict=MagicMock(),
        data={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        call_type="acompletion",
        guardrails_only=guardrails_only,
    )

    assert enforcer.calls == 1


@pytest.mark.asyncio
async def test_an_accounting_hook_is_skipped_by_a_guardrails_only_walk(proxy_logging, monkeypatch):
    """Charging budget or taking a rate-limit slot once per batch record is the bug this prevents."""
    accountant = _Accountant()
    monkeypatch.setattr(litellm, "callbacks", [accountant])

    await proxy_logging.pre_call_hook(
        user_api_key_dict=MagicMock(),
        data={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        call_type="acompletion",
        guardrails_only=True,
    )
    assert accountant.calls == 0

    await proxy_logging.pre_call_hook(
        user_api_key_dict=MagicMock(),
        data={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        call_type="acompletion",
        guardrails_only=False,
    )
    assert accountant.calls == 1, "the online path must be untouched"


def test_has_pre_call_guardrails_counts_a_content_enforcer(proxy_logging, monkeypatch):
    """The batch scan is gated on this, so an enforcer-only proxy must still stream the file."""
    monkeypatch.setattr(litellm, "callbacks", [_Accountant()])
    assert proxy_logging.has_pre_call_guardrails({}) is False

    monkeypatch.setattr(litellm, "callbacks", [_Enforcer()])
    # required: the list keeps length one, so a reused object address could hit a stale entry
    ProxyLogging._callback_capabilities_cache.clear()
    assert proxy_logging.has_pre_call_guardrails({}) is True


def test_every_pre_call_customlogger_is_deliberately_classified():
    """
    A ledger, so a new hook cannot land unclassified.

    The flag has no forcing function on its own: an enforcement hook added later would simply
    default to False and silently skip batch records, which is the bug this fixes. Adding a
    pre-call CustomLogger now fails here until someone puts it on one side.
    """
    judges_content = {
        "_OPTIONAL_PromptInjectionDetection",
        "_PROXY_AzureContentSafety",
        "_ENTERPRISE_BannedKeywords",
        "_ENTERPRISE_BlockedUserList",
    }
    counts_or_shapes_the_request = {
        "_PROXY_MaxBudgetLimiter",
        "_PROXY_MaxParallelRequestsHandler_v3",
        "_PROXY_MaxIterationsHandler",
        "_PROXY_MaxBudgetPerSessionHandler",
        "_PROXY_CacheControlCheck",
        "_PROXY_BatchRedisRequests",
        "_PROXY_SensitiveDataRoutingHandler",
        "ResponsesIDSecurity",
        "SkillsInjectionHook",
        "_PROXY_LiteLLMManagedFiles",
        "_PROXY_LiteLLMManagedVectorStores",
    }

    from litellm.proxy.hooks import PROXY_HOOKS

    registered = dict(PROXY_HOOKS)
    for name, cls in (
        ("banned_keywords", _load("enterprise.enterprise_hooks.banned_keywords", "_ENTERPRISE_BannedKeywords")),
        ("blocked_user_check", _load("enterprise.enterprise_hooks.blocked_user_list", "_ENTERPRISE_BlockedUserList")),
        ("detect_prompt_injection", _load("litellm.proxy.hooks.prompt_injection_detection", "_OPTIONAL_PromptInjectionDetection")),
        ("azure_content_safety", _load("litellm.proxy.hooks.azure_content_safety", "_PROXY_AzureContentSafety")),
    ):
        if cls is not None:
            registered[name] = cls

    unclassified = []
    for cls in registered.values():
        if not (isinstance(cls, type) and issubclass(cls, CustomLogger)):
            continue
        if "async_pre_call_hook" not in cls.__dict__:
            continue
        name = cls.__name__
        if name in judges_content:
            assert cls.enforces_request_content is True, f"{name} judges content but is not marked"
        elif name in counts_or_shapes_the_request:
            assert cls.enforces_request_content is False, f"{name} must not run once per record"
        else:
            unclassified.append(name)

    assert not unclassified, (
        f"pre-call CustomLogger(s) with no recorded classification: {sorted(unclassified)}. "
        "Decide whether each judges the payload (mark it) or counts the request (leave it)."
    )
    assert CustomLogger.enforces_request_content is False


# ---------------------------------------------------------------------------
# scan_raw_request: a guardrail's block decision must not depend on YAML order
# ---------------------------------------------------------------------------


class _RedactingGuardrail(CustomGuardrail):
    """Mirrors a real masking guardrail (e.g. Lakera's advisory mode): mutates
    ``data`` in place and returns None, same as CustomGuardrail's documented
    contract for in-place mutation."""

    def __init__(self, **kwargs):
        kwargs.setdefault("default_on", True)
        kwargs.setdefault("event_hook", GuardrailEventHooks.pre_call)
        super().__init__(guardrail_name="redactor", **kwargs)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):  # type: ignore[override]
        for msg in data.get("messages", []):
            if "SECRET" in msg.get("content", ""):
                msg["content"] = msg["content"].replace("SECRET", "[REDACTED]")
        return None


class _BlockOnSecretGuardrail(CustomGuardrail):
    """Blocks the request if any message contains the literal string SECRET."""

    def __init__(self, **kwargs):
        kwargs.setdefault("default_on", True)
        kwargs.setdefault("event_hook", GuardrailEventHooks.pre_call)
        super().__init__(guardrail_name="blocker", **kwargs)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):  # type: ignore[override]
        if any("SECRET" in msg.get("content", "") for msg in data.get("messages", [])):
            raise HTTPException(status_code=400, detail="blocked: SECRET detected")
        return None


def _secret_request() -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": "here is my SECRET"}], "model": "m"}


@pytest.mark.asyncio
async def test_yaml_order_changes_enforcement_without_scan_raw_request(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """Baseline (the bug): declaring the redactor before the blocker lets a
    request through that would have been blocked in the opposite order,
    because the blocker only ever sees the already-redacted content."""
    monkeypatch.setattr(litellm, "callbacks", [_RedactingGuardrail(), _BlockOnSecretGuardrail()])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=_secret_request(),
        call_type="completion",
    )
    assert "[REDACTED]" in out["messages"][0]["content"]


@pytest.mark.asyncio
async def test_reversed_yaml_order_blocks_the_same_request(proxy_logging, make_user_api_key_auth, monkeypatch):
    """Same two guardrails, opposite declaration order: the blocker now runs
    first against the still-raw content and correctly rejects the request.
    Confirms the baseline test above is a real order-dependence, not a fluke."""
    monkeypatch.setattr(litellm, "callbacks", [_BlockOnSecretGuardrail(), _RedactingGuardrail()])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    with pytest.raises(HTTPException, match="blocked"):
        await proxy_logging.pre_call_hook(
            user_api_key_dict=make_user_api_key_auth(),
            data=_secret_request(),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_scan_raw_request_makes_blocking_order_independent(proxy_logging, make_user_api_key_auth, monkeypatch):
    """Maintainer finding on BerriAI/litellm#34940: with scan_raw_request=True
    on the blocker, declaring the redactor first no longer lets the request
    through -- the blocker evaluates the pre-loop snapshot regardless of its
    position in the guardrails list."""
    monkeypatch.setattr(
        litellm, "callbacks", [_RedactingGuardrail(), _BlockOnSecretGuardrail(scan_raw_request=True)]
    )
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    with pytest.raises(HTTPException, match="blocked"):
        await proxy_logging.pre_call_hook(
            user_api_key_dict=make_user_api_key_auth(),
            data=_secret_request(),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_scan_raw_request_guardrail_does_not_undo_later_masking(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """A scan_raw_request guardrail that passes (its own snapshot has no
    violation) must not affect what a later guardrail in the sequence does to
    the live request -- its own discarded view of the data must not corrupt
    or reset the shared ``data`` object for the rest of the loop. Uses a
    request with no SECRET at all, so the blocker passes cleanly, and a
    separate marker (PII_TOKEN) that only the redactor reacts to."""

    class _PiiRedactor(_RedactingGuardrail):
        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):  # type: ignore[override]
            for msg in data.get("messages", []):
                if "PII_TOKEN" in msg.get("content", ""):
                    msg["content"] = msg["content"].replace("PII_TOKEN", "[REDACTED]")
            return None

    monkeypatch.setattr(
        litellm, "callbacks", [_BlockOnSecretGuardrail(scan_raw_request=True), _PiiRedactor()]
    )
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"role": "user", "content": "my PII_TOKEN is here"}], "model": "m"},
        call_type="completion",
    )
    assert "[REDACTED]" in out["messages"][0]["content"]


class _Unpicklable:
    """Mirrors a real otel span: deepcopy always raises, matching what
    safe_deep_copy exists to handle (see litellm_core_utils/core_helpers.py)."""

    def __deepcopy__(self, memo):
        raise TypeError("cannot deepcopy this object")


@pytest.mark.asyncio
async def test_scan_raw_request_snapshot_survives_unpicklable_metadata(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    Bugbot finding on BerriAI/litellm#34940: the scan_raw_request snapshot
    used a bare copy.deepcopy, which raises on request payloads carrying
    unpicklable objects (e.g. metadata["litellm_parent_otel_span"] when
    tracing is enabled) -- failing every guarded request, not just ones
    that actually use scan_raw_request. Must use safe_deep_copy instead.
    """
    monkeypatch.setattr(litellm, "callbacks", [_BlockOnSecretGuardrail(scan_raw_request=True)])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    data = {
        "messages": [{"role": "user", "content": "hello, nothing flagged here"}],
        "model": "m",
        "metadata": {"litellm_parent_otel_span": _Unpicklable()},
    }
    out = await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=data,
        call_type="completion",
    )
    assert out is not None


@pytest.mark.asyncio
async def test_scan_raw_request_snapshot_taken_before_pipelines(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    veria-ai finding on BerriAI/litellm#34940: the raw snapshot was taken
    after _maybe_execute_pipelines ran, so a pipeline that masks content
    ahead of a non-pipelined scan_raw_request guardrail could still hide
    the violation from it. Simulates a pipeline-style rewrite by having
    _maybe_execute_pipelines itself return redacted data, and confirms the
    scan_raw_request blocker still sees the pre-pipeline raw content.
    """

    async def fake_pipelines(self, data, user_api_key_dict, call_type, event_hook):
        for msg in data.get("messages", []):
            if "SECRET" in msg.get("content", ""):
                msg["content"] = msg["content"].replace("SECRET", "[REDACTED]")
        return data

    monkeypatch.setattr(ProxyLogging, "_maybe_execute_pipelines", fake_pipelines)
    monkeypatch.setattr(litellm, "callbacks", [_BlockOnSecretGuardrail(scan_raw_request=True)])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    with pytest.raises(HTTPException, match="blocked"):
        await proxy_logging.pre_call_hook(
            user_api_key_dict=make_user_api_key_auth(),
            data=_secret_request(),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_scan_raw_request_warns_when_guardrail_mutation_discarded(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    veria-ai finding on BerriAI/litellm#34940: scan_raw_request is accepted
    even for a guardrail that mutates the request (e.g. a masking
    integration), silently discarding its redaction and forwarding raw
    content. Config-time rejection isn't generically possible (no marker
    exists for "this guardrail mutates"), so a loud runtime warning is the
    mitigation: confirm it fires when a scan_raw_request guardrail returns
    a modified payload.
    """

    class _MutatingScanner(_RedactingGuardrail):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scan_raw_request = True

        async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):  # type: ignore[override]
            for msg in data.get("messages", []):
                msg["content"] = msg["content"].replace("SECRET", "[REDACTED]")
            return data

    from litellm.proxy import utils as proxy_utils_module

    mock_logger = MagicMock()
    monkeypatch.setattr(proxy_utils_module, "verbose_proxy_logger", mock_logger)
    monkeypatch.setattr(litellm, "callbacks", [_MutatingScanner()])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=_secret_request(),
        call_type="completion",
    )
    mock_logger.warning.assert_called_once()
    assert "scan_raw_request" in str(mock_logger.warning.call_args)


@pytest.mark.asyncio
async def test_scan_raw_request_does_not_warn_when_guardrail_only_blocks(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    Bugbot finding on BerriAI/litellm#34940: _process_guardrail_callback always
    returns a dict once a guardrail actually runs (it only returns None when
    should_run_guardrail is False), so checking `result is not None` is true on
    every single request -- a correctly configured, non-mutating scan_raw_request
    blocker (like _BlockOnSecretGuardrail here) would warn on every call, not just
    when it actually mutates something.
    """
    from litellm.proxy import utils as proxy_utils_module

    mock_logger = MagicMock()
    monkeypatch.setattr(proxy_utils_module, "verbose_proxy_logger", mock_logger)
    monkeypatch.setattr(litellm, "callbacks", [_BlockOnSecretGuardrail(scan_raw_request=True)])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data={"messages": [{"role": "user", "content": "nothing flagged here"}], "model": "m"},
        call_type="completion",
    )
    mock_logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_scan_raw_request_warns_on_in_place_mutation_returning_none(
    proxy_logging, make_user_api_key_auth, monkeypatch
):
    """
    _RedactingGuardrail mirrors the common in-place-mutate-and-return-None
    guardrail contract (e.g. real masking integrations). Detecting this case
    correctly requires comparing dict *content*, not object identity: the
    mutated dict is still the exact same object reference the guardrail was
    given, so an identity check (`result is input_data`) would wrongly say
    nothing changed.
    """
    from litellm.proxy import utils as proxy_utils_module

    class _ScanningRedactor(_RedactingGuardrail):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scan_raw_request = True

    mock_logger = MagicMock()
    monkeypatch.setattr(proxy_utils_module, "verbose_proxy_logger", mock_logger)
    monkeypatch.setattr(litellm, "callbacks", [_ScanningRedactor()])
    proxy_logging.slack_alerting_instance = MagicMock(alerting=None)
    await proxy_logging.pre_call_hook(
        user_api_key_dict=make_user_api_key_auth(),
        data=_secret_request(),
        call_type="completion",
    )
    mock_logger.warning.assert_called_once()
    assert "scan_raw_request" in str(mock_logger.warning.call_args)
