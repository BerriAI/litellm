"""
Unit tests for the treat_finish_reason_as_failure router knob.

A provider can report a terminal condition (context window exceeded, and
similar) as a stop reason on an HTTP 200. The knob maps such reasons to a
router-understood exception class, so the mapped reason engages allowed_fails,
cooldowns, and fallbacks like any failure. When the mapped reason is present
but no generic fallback can serve the retry, the response reaches the client
unchanged while the deployment still counts the failure.
"""

import json
from typing import Any

import httpx
import pytest
from pytest import MonkeyPatch

import litellm
from litellm import Router
from litellm.router_utils.cooldown_handlers import _get_cooldown_deployments

CONTEXT_WINDOW_RESPONSE: dict[str, Any] = {
    "id": "msg_context",
    "type": "message",
    "role": "assistant",
    "model": "claude-fable-5",
    "content": [],
    "stop_reason": "model_context_window_exceeded",
    "stop_sequence": None,
    "usage": {"input_tokens": 25, "output_tokens": 1},
}

OK_RESPONSE: dict[str, Any] = {
    "id": "msg_ok",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 25, "output_tokens": 2},
}


class FakeAnthropicUpstream:
    """Intercepts the third-party transport (httpx.AsyncClient.send): reports the
    context-window stop reason on fable models, answers on others. The router
    deliberately does not forward caller-injected clients, so the transport is the
    seam that exercises the real litellm pipeline end to end."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        model = body.get("model", "")
        self.calls.append(model)
        overrun = "fable" in model
        return httpx.Response(
            200,
            json=CONTEXT_WINDOW_RESPONSE if overrun else OK_RESPONSE,
            request=request,
        )

    def install(self, monkeypatch: MonkeyPatch) -> None:
        async def _send(_client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            return await self.send(request, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "send", _send)


FABLE_TIER = {
    "model_name": "fable-tier",
    "litellm_params": {"model": "anthropic/claude-fable-5", "api_key": "sk-test"},
}
OPUS_TARGET = {
    "model_name": "opus-target",
    "litellm_params": {"model": "anthropic/claude-opus-5", "api_key": "sk-test"},
}


def _knob() -> dict[str, str]:
    return {"model_context_window_exceeded": "RateLimitError"}


def _deployment_id(router: Router, index: int = 0) -> str:
    return router.model_list[index]["model_info"]["id"]


@pytest.mark.asyncio
async def test_chat_completion_mapped_reason_falls_back_and_cools_down(monkeypatch: MonkeyPatch):
    fake = FakeAnthropicUpstream()
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET],
        treat_finish_reason_as_failure=_knob(),
        default_fallbacks=["opus-target"],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=10,
    )
    fake.install(monkeypatch)

    response = await router.acompletion(model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}])

    assert response.model == "claude-opus-5"
    assert len(fake.calls) == 2
    assert "claude-fable-5" in fake.calls[0]
    assert "claude-opus-5" in fake.calls[1]
    assert router.fail_calls["anthropic/claude-fable-5"] == 1
    fable_id = _deployment_id(router, 0)
    assert fable_id in _get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None)


@pytest.mark.asyncio
async def test_anthropic_messages_mapped_reason_falls_back(monkeypatch: MonkeyPatch):
    fake = FakeAnthropicUpstream()
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET],
        treat_finish_reason_as_failure=_knob(),
        default_fallbacks=["opus-target"],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=10,
    )
    fake.install(monkeypatch)

    response = await router.aanthropic_messages(
        model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
    )

    assert response["id"] == "msg_ok"
    assert response["stop_reason"] == "end_turn"
    assert len(fake.calls) == 2
    assert "claude-opus-5" in fake.calls[1]


@pytest.mark.asyncio
async def test_mapped_reason_without_fallback_returns_response_and_counts_failure(monkeypatch: MonkeyPatch):
    fake = FakeAnthropicUpstream()
    router = Router(
        model_list=[FABLE_TIER],
        treat_finish_reason_as_failure=_knob(),
        num_retries=0,
        allowed_fails=0,
        cooldown_time=10,
    )
    fake.install(monkeypatch)

    response = await router.acompletion(model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}])

    assert response.model == "claude-fable-5"
    assert len(fake.calls) == 1
    fable_id = _deployment_id(router, 0)
    failures = router.cache.get_cache(local_only=True, key=f"{fable_id}:fails")
    assert failures == 1
    assert fable_id in _get_cooldown_deployments(litellm_router_instance=router, parent_otel_span=None)


@pytest.mark.asyncio
async def test_knob_unset_ignores_terminal_stop_reason(monkeypatch: MonkeyPatch):
    fake = FakeAnthropicUpstream()
    router = Router(model_list=[FABLE_TIER], num_retries=0)
    fake.install(monkeypatch)

    response = await router.acompletion(model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}])

    assert response.model == "claude-fable-5"
    assert len(fake.calls) == 1
    assert router.fail_calls["fable-tier"] == 0


def test_unknown_exception_name_raises_at_construction():
    with pytest.raises(ValueError, match="NotAnException"):
        Router(model_list=[], treat_finish_reason_as_failure={"x": "NotAnException"})


@pytest.mark.asyncio
async def test_mapped_finish_reason_helpers_direct(monkeypatch: MonkeyPatch):
    """Direct coverage of the knob helpers (the router code-coverage check matches by name)."""
    fake = FakeAnthropicUpstream()
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET],
        treat_finish_reason_as_failure=_knob(),
        default_fallbacks=["opus-target"],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=10,
    )
    fake.install(monkeypatch)

    ok = await router.acompletion(model="opus-target", max_tokens=16, messages=[{"role": "user", "content": "hi"}])
    assert router._get_mapped_finish_reason(ok) is None
    assert router._generic_fallback_available("fable-tier", {}) is True

    error = router._finish_reason_failure_error(model="fable-tier", reason="model_context_window_exceeded")
    assert error.status_code == 429

    deployment = router.model_list[0]
    accounted = router._account_mapped_finish_reason_failure(
        model="fable-tier",
        deployment=deployment,
        reason="model_context_window_exceeded",
        kwargs={},
    )
    assert accounted is not None

    with pytest.raises(litellm.RateLimitError):
        router._handle_mapped_finish_reason_failure(
            model="fable-tier",
            deployment=deployment,
            reason="model_context_window_exceeded",
            kwargs={},
        )
