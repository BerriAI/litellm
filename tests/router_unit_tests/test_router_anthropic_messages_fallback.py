"""
Unit tests for safeguard-refusal fallback on the /v1/messages router surface.

An Anthropic safeguard refusal is an HTTP 200 whose body carries
stop_reason "refusal" plus a stop_details object; the router converts it
into a ContentPolicyViolationError so the content-policy fallback chain
runs, but only when a matching fallback is configured. A plain refusal
without stop_details, or any refusal with nothing configured, must reach
the client byte-identical.

The upstream is faked at the HTTP boundary by intercepting the third-party
transport (httpx.AsyncClient.send), so requests run litellm's real
transformation, allowlist, and streaming pipeline end to end.
"""

import json
from typing import Any, AsyncIterator
from unittest.mock import patch

import httpx
import pytest

from litellm import Router
from litellm.router_utils.fallback_event_handlers import (
    PRE_ROUTING_SELECTED_MODEL_KEY,
    record_pre_routing_selection,
)

REFUSAL_RESPONSE: dict[str, Any] = {
    "id": "msg_refusal",
    "type": "message",
    "role": "assistant",
    "model": "claude-fable-5",
    "content": [],
    "stop_reason": "refusal",
    "stop_sequence": None,
    "stop_details": {"category": "cyber", "explanation": "flagged"},
    "usage": {"input_tokens": 25, "output_tokens": 1},
}

PLAIN_REFUSAL_RESPONSE: dict[str, Any] = {k: v for k, v in REFUSAL_RESPONSE.items() if k != "stop_details"}

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


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


REFUSAL_STREAM_FRAMES: tuple[bytes, ...] = (
    _sse("message_start", {"type": "message_start", "message": {**REFUSAL_RESPONSE, "stop_reason": None}}),
    _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "refusal", "stop_details": {"category": "cyber"}},
            "usage": {"output_tokens": 1},
        },
    ),
    _sse("message_stop", {"type": "message_stop"}),
)

OK_STREAM_FRAMES: tuple[bytes, ...] = (
    _sse("message_start", {"type": "message_start", "message": {**OK_RESPONSE, "stop_reason": None}}),
    _sse(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
    ),
    _sse("message_stop", {"type": "message_stop"}),
)


def _split_frames_mid_data_line(frames: tuple[bytes, ...]) -> tuple[bytes, ...]:
    """Split each frame's data line in half, modeling a transport chunk boundary."""
    return tuple(part for frame in frames for part in (frame[: len(frame) // 2], frame[len(frame) // 2 :]))


class _FrameStream(httpx.AsyncByteStream):
    def __init__(self, frames: tuple[bytes, ...]) -> None:
        self._frames = frames

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame

    async def aclose(self) -> None:
        return None


class FakeAnthropicUpstream:
    """Intercepts the third-party transport (httpx.AsyncClient.send): refuses on fable
    models, answers on others. The router deliberately does not forward caller-injected
    clients, so the transport is the seam that exercises the real litellm pipeline."""

    def __init__(
        self,
        refusal_body: dict[str, Any] = REFUSAL_RESPONSE,
        refusal_frames: tuple[bytes, ...] = REFUSAL_STREAM_FRAMES,
    ) -> None:
        self.refusal_body = refusal_body
        self.refusal_frames = refusal_frames
        self.calls: list[str] = []
        self.bodies: list[dict[str, Any]] = []

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        model = body.get("model", "")
        self.calls.append(model)
        self.bodies.append(body)
        refuses = "fable" in model
        if body.get("stream"):
            frames = self.refusal_frames if refuses else OK_STREAM_FRAMES
            return httpx.Response(
                200,
                stream=_FrameStream(frames),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx.Response(200, json=self.refusal_body if refuses else OK_RESPONSE, request=request)

    def install(self):
        async def _send(_client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
            return await self.send(request, **kwargs)

        return patch("httpx.AsyncClient.send", new=_send)


FABLE_TIER = {
    "model_name": "fable-tier",
    "litellm_params": {"model": "anthropic/claude-fable-5", "api_key": "sk-test"},
}
OPUS_TARGET = {
    "model_name": "opus-target",
    "litellm_params": {"model": "anthropic/claude-opus-5", "api_key": "sk-test"},
}


def _router(content_policy_fallbacks: list | None) -> Router:
    return Router(model_list=[FABLE_TIER, OPUS_TARGET], content_policy_fallbacks=content_policy_fallbacks)


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in stream])


@pytest.mark.asyncio
async def test_non_streaming_refusal_with_fallback_row_returns_fallback_response():
    fake = FakeAnthropicUpstream()
    router = _router(content_policy_fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )

    assert response["stop_reason"] == "end_turn"
    assert response["id"] == "msg_ok"
    assert len(fake.calls) == 2
    assert "claude-opus-5" in fake.calls[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_policy_fallbacks, upstream_body",
    [
        (None, REFUSAL_RESPONSE),
        ([{"unrelated-group": ["opus-target"]}], REFUSAL_RESPONSE),
        ([{"fable-tier": ["opus-target"]}], PLAIN_REFUSAL_RESPONSE),
    ],
    ids=["nothing-configured", "row-for-other-group", "refusal-without-stop-details"],
)
async def test_non_streaming_refusal_passes_through_untouched(content_policy_fallbacks, upstream_body):
    fake = FakeAnthropicUpstream(refusal_body=upstream_body)
    router = _router(content_policy_fallbacks=content_policy_fallbacks)

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )

    assert response["stop_reason"] == "refusal"
    assert response.get("stop_details") == upstream_body.get("stop_details")
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_streaming_refusal_with_fallback_row_streams_fallback_frames():
    fake = FakeAnthropicUpstream()
    router = _router(content_policy_fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        stream = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, stream=True, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(stream)

    assert b'"refusal"' not in body
    assert b"text_delta" in body
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_streaming_refusal_split_across_chunks_still_falls_back():
    fake = FakeAnthropicUpstream(refusal_frames=_split_frames_mid_data_line(REFUSAL_STREAM_FRAMES))
    router = _router(content_policy_fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        stream = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, stream=True, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(stream)

    assert b'"refusal"' not in body
    assert b"text_delta" in body
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_streaming_refusal_without_fallback_row_passes_frames_through():
    fake = FakeAnthropicUpstream()
    router = _router(content_policy_fallbacks=None)

    with fake.install():
        stream = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, stream=True, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(stream)

    assert b'"stop_reason": "refusal"' in body
    assert b"stop_details" in body
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_streaming_refusal_on_routed_tier_matches_tier_keyed_row_without_inbound_metadata():
    """The pre-routing hook's tier stamp must reach the mid-stream fallback lookup even when the
    request carries no metadata bucket at all (the snapshot is taken before the request runs)."""
    fake = FakeAnthropicUpstream()
    smart_router = {
        "model_name": "smart-router",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {
                "tiers": {"SIMPLE": "fable-tier", "MEDIUM": "fable-tier", "COMPLEX": "fable-tier"}
            },
            "complexity_router_default_model": "fable-tier",
        },
        "model_info": {"id": "router-1", "db_model": True},
    }
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET, smart_router],
        content_policy_fallbacks=[{"fable-tier": ["opus-target"]}],
        ignore_invalid_deployments=True,
    )

    with fake.install():
        stream = await router.aanthropic_messages(
            model="smart-router", max_tokens=16, stream=True, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(stream)

    assert b'"refusal"' not in body
    assert b"text_delta" in body
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_caller_forged_tier_stamp_cannot_pick_the_streaming_fallback_chain():
    fake = FakeAnthropicUpstream()
    router = _router(content_policy_fallbacks=[{"forged-tier": ["opus-target"]}])

    with fake.install():
        stream = await router.aanthropic_messages(
            model="fable-tier",
            max_tokens=16,
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
            litellm_metadata={PRE_ROUTING_SELECTED_MODEL_KEY: "forged-tier"},
        )
        body = await _collect(stream)

    assert b'"stop_reason": "refusal"' in body
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_tier_stamp_never_reaches_provider_bound_metadata():
    """On /v1/messages the top-level metadata dict is Anthropic's own request field, so the
    routed-tier stamp must never appear in any upstream body even when the client sends one."""
    fake = FakeAnthropicUpstream()
    smart_router = {
        "model_name": "smart-router",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {
                "tiers": {"SIMPLE": "fable-tier", "MEDIUM": "fable-tier", "COMPLEX": "fable-tier"}
            },
            "complexity_router_default_model": "fable-tier",
        },
        "model_info": {"id": "router-1", "db_model": True},
    }
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET, smart_router],
        content_policy_fallbacks=[{"fable-tier": ["opus-target"]}],
        ignore_invalid_deployments=True,
    )

    with fake.install():
        response = await router.aanthropic_messages(
            model="smart-router",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            metadata={"user_id": "u1"},
        )

    assert response["stop_reason"] == "end_turn"
    assert len(fake.bodies) == 2
    for body in fake.bodies:
        assert body.get("metadata") == {"user_id": "u1"}


def test_record_pre_routing_selection_writes_only_the_internal_bucket():
    """The Anthropic request's own metadata field must never carry the tier stamp."""
    kwargs = {"metadata": {"user_id": "u1"}, "litellm_metadata": {}}

    record_pre_routing_selection(kwargs, "tier-x")

    assert kwargs["litellm_metadata"] == {PRE_ROUTING_SELECTED_MODEL_KEY: "tier-x"}
    assert kwargs["metadata"] == {"user_id": "u1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_generic_only_row_recovers_safeguard_refusal(stream):
    """With no content-policy list configured, a generic fallback row covers safeguard refusals,
    so the dashboard's generic fallbacks work without config-only content_policy rows."""
    fake = FakeAnthropicUpstream()
    router = Router(model_list=[FABLE_TIER, OPUS_TARGET], fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, stream=stream, messages=[{"role": "user", "content": "hi"}]
        )
        body = await _collect(response) if stream else response

    if stream:
        assert b'"refusal"' not in body
        assert b"text_delta" in body
    else:
        assert body["stop_reason"] == "end_turn"
    assert len(fake.calls) == 2
    assert "claude-opus-5" in fake.calls[1]


@pytest.mark.asyncio
async def test_configured_content_policy_list_stays_authoritative_over_generic_rows():
    fake = FakeAnthropicUpstream()
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET],
        fallbacks=[{"fable-tier": ["opus-target"]}],
        content_policy_fallbacks=[{"unrelated-group": ["opus-target"]}],
    )

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )

    assert response["stop_reason"] == "refusal"
    assert len(fake.calls) == 1


def test_refusal_fallback_available_arms_on_generic_rows_only_without_content_policy():
    router = Router(model_list=[FABLE_TIER, OPUS_TARGET], fallbacks=[{"tier-group": ["opus-target"]}])
    stamped = {"litellm_metadata": {PRE_ROUTING_SELECTED_MODEL_KEY: "tier-group"}}

    assert router._refusal_fallback_available("router-group", stamped) is True
    assert router._refusal_fallback_available("router-group", {}) is False
    assert router._refusal_fallback_available("router-group", {"content_policy_fallbacks": [{"other": ["x"]}]}) is False


def test_chat_content_filter_gate_unchanged_by_generic_rows():
    """The generic-row arming is scoped to /v1/messages safeguard refusals; the chat surface's
    content_filter gate keeps its long-standing content-policy-only semantics."""
    from litellm.types.utils import Choices, ModelResponse

    router = Router(model_list=[FABLE_TIER, OPUS_TARGET], fallbacks=[{"fable-tier": ["opus-target"]}])
    response = ModelResponse(choices=[Choices(finish_reason="content_filter")])

    assert router._should_raise_content_policy_error(model="fable-tier", response=response, kwargs={}) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_disable_fallbacks_returns_the_refusal_instead_of_raising(stream):
    """A request that opted out of fallbacks must receive the provider's refusal response,
    never a ContentPolicyViolationError the dispatcher refuses to recover."""
    fake = FakeAnthropicUpstream()
    router = Router(model_list=[FABLE_TIER, OPUS_TARGET], fallbacks=[{"fable-tier": ["opus-target"]}])

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier",
            max_tokens=16,
            stream=stream,
            disable_fallbacks=True,
            messages=[{"role": "user", "content": "hi"}],
        )
        body = await _collect(response) if stream else response

    if stream:
        assert b'"stop_reason": "refusal"' in body
    else:
        assert body["stop_reason"] == "refusal"
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_disable_fallbacks_beats_a_content_policy_row_too():
    fake = FakeAnthropicUpstream()
    router = Router(
        model_list=[FABLE_TIER, OPUS_TARGET],
        content_policy_fallbacks=[{"fable-tier": ["opus-target"]}],
    )

    with fake.install():
        response = await router.aanthropic_messages(
            model="fable-tier",
            max_tokens=16,
            disable_fallbacks=True,
            messages=[{"role": "user", "content": "hi"}],
        )

    assert response["stop_reason"] == "refusal"
    assert len(fake.calls) == 1


def test_refusal_gate_keys_on_pre_routing_tier_stamp():
    router = _router(content_policy_fallbacks=[{"tier-group": ["opus-target"]}])

    def anthropic_messages(**kwargs: Any) -> None:
        return None

    refusal_kwargs = {"litellm_metadata": {PRE_ROUTING_SELECTED_MODEL_KEY: "tier-group"}}
    assert (
        router._should_raise_anthropic_refusal_error(
            model="router-group",
            original_generic_function=anthropic_messages,
            response=dict(REFUSAL_RESPONSE),
            kwargs=refusal_kwargs,
        )
        is True
    )
    assert (
        router._should_raise_anthropic_refusal_error(
            model="router-group",
            original_generic_function=anthropic_messages,
            response=dict(REFUSAL_RESPONSE),
            kwargs={},
        )
        is False
    )


def test_has_content_policy_fallback_default_fallbacks_arm():
    router = Router(model_list=[OPUS_TARGET], fallbacks=[{"*": ["opus-target"]}])

    assert router._has_content_policy_fallback("any-group", {}) is True
    assert router._has_content_policy_fallback("any-group", {"content_policy_fallbacks": [{"other": ["x"]}]}) is False


def test_get_fallback_model_group_for_lookup_groups_orders_tier_before_requested():
    router = _router(content_policy_fallbacks=None)
    fallbacks = [{"tier1": ["backup-a"]}, {"smart-router": ["backup-b"]}]

    assert router._get_fallback_model_group_for_lookup_groups(
        fallbacks=fallbacks, lookup_groups=("tier1", "smart-router")
    ) == ["backup-a"]
    assert router._get_fallback_model_group_for_lookup_groups(
        fallbacks=fallbacks, lookup_groups=("tier9", "smart-router")
    ) == ["backup-b"]
    assert router._get_fallback_model_group_for_lookup_groups(fallbacks=fallbacks, lookup_groups=()) is None


def test_refusal_gate_ignores_other_generic_call_types():
    router = _router(content_policy_fallbacks=[{"fable-tier": ["opus-target"]}])

    def aresponses(**kwargs: Any) -> None:
        return None

    assert (
        router._should_raise_anthropic_refusal_error(
            model="fable-tier",
            original_generic_function=aresponses,
            response=dict(REFUSAL_RESPONSE),
            kwargs={},
        )
        is False
    )
