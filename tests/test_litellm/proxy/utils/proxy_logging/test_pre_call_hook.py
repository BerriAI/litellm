"""Pin ``ProxyLogging.pre_call_hook`` and ``process_pre_call_hook_response``."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import RejectedRequestError
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.utils import ProxyLogging


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
