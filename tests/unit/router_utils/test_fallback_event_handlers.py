import asyncio
import importlib
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import get_llm_provider_logic
from litellm.router_utils.fallback_event_handlers import (
    PRE_ROUTING_SELECTED_MODEL_KEY,
    AttemptedFallbackTargets,
    clear_pre_routing_selection,
    get_fallback_model_group,
    get_pre_routing_selection,
    log_failure_fallback_event,
    log_success_fallback_event,
    record_pre_routing_selection,
)
from litellm.types.router import FallbackSkipReason, SkippedFallbackTarget
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


def test_get_fallback_model_group_does_not_mutate_fallbacks():
    """A string fallback must be resolved without mutating the caller's
    fallbacks list, which is the live router config shared across requests."""
    fallbacks = [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="unmatched-model")

    assert fallback_model_group == ["gpt-4o-mini"]
    assert fallbacks == [{"gpt-3.5-turbo": ["claude-3-haiku"]}, "gpt-4o-mini"]


class TestPreRoutingSelectionCarriesToFallbacks:
    """#38832: a complexity/auto router picks a tier behind the router name, but fallback
    lookup kept using the router name, so the tier's configured chain never ran."""

    def test_selection_is_recorded_in_the_metadata_bucket(self):
        kwargs = {"model": "smart-router", "metadata": {}}
        record_pre_routing_selection(kwargs, "tier1")
        assert kwargs["metadata"]["pre_routing_selected_model"] == "tier1"
        assert get_pre_routing_selection(kwargs) == "tier1"

    def test_selection_is_recorded_in_the_litellm_metadata_bucket(self):
        kwargs = {"model": "smart-router", "litellm_metadata": {}}
        record_pre_routing_selection(kwargs, "tier2")
        assert get_pre_routing_selection(kwargs) == "tier2"

    def test_a_bucket_survives_the_kwargs_copy_that_fallbacks_run_on(self):
        """The bucket is shared by reference, which is the whole reason this works."""
        outer = {"model": "smart-router", "metadata": {}}
        inner = {**outer}
        record_pre_routing_selection(inner, "tier1")
        assert get_pre_routing_selection(outer) == "tier1"

    def test_no_selection_reads_as_none(self):
        assert get_pre_routing_selection({"model": "smart-router", "metadata": {}}) is None
        assert get_pre_routing_selection({"model": "smart-router"}) is None

    def test_missing_kwargs_is_a_no_op(self):
        """A caller with no kwargs must not raise, and must not leak the selection anywhere."""
        record_pre_routing_selection(None, "tier1")

        assert get_pre_routing_selection({}) is None

    def test_a_non_dict_bucket_is_ignored(self):
        kwargs = {"model": "smart-router", "metadata": "not-a-dict"}
        record_pre_routing_selection(kwargs, "tier1")
        assert get_pre_routing_selection(kwargs) is None

    def test_fallbacks_resolve_against_the_selected_tier(self):
        """The lookup the router performs, keyed on the tier rather than the router name."""
        fallbacks = [{"tier1": ["backup-a", "backup-b"]}, {"tier2": ["backup-c"]}]
        assert get_fallback_model_group(fallbacks=fallbacks, model_group="tier1")[0] == ["backup-a", "backup-b"]
        assert get_fallback_model_group(fallbacks=fallbacks, model_group="smart-router")[0] is None


class TestPreRoutingSelectionIsPerHop:
    """#38832 review: the buckets also carry whatever the caller sent, and a fallback hop
    inherits the previous hop's tier, so a hop must start without a selection."""

    def test_a_caller_supplied_selection_is_dropped(self):
        kwargs = {"model": "plain", "metadata": {"pre_routing_selected_model": "tier1"}}

        clear_pre_routing_selection(kwargs)

        assert get_pre_routing_selection(kwargs) is None
        assert "pre_routing_selected_model" not in kwargs["metadata"]

    def test_both_buckets_are_cleared(self):
        kwargs = {
            "metadata": {"pre_routing_selected_model": "tier1"},
            "litellm_metadata": {"pre_routing_selected_model": "tier2"},
        }

        clear_pre_routing_selection(kwargs)

        assert get_pre_routing_selection(kwargs) is None

    def test_the_rest_of_the_bucket_is_left_alone(self):
        kwargs = {"metadata": {"pre_routing_selected_model": "tier1", "tags": ["a"]}}

        clear_pre_routing_selection(kwargs)

        assert kwargs["metadata"] == {"tags": ["a"]}

    def test_clearing_is_a_no_op_without_a_usable_bucket(self):
        kwargs = {"model": "plain", "metadata": "not-a-dict"}

        clear_pre_routing_selection(None)
        clear_pre_routing_selection(kwargs)

        assert kwargs == {"model": "plain", "metadata": "not-a-dict"}

    def test_a_selection_recorded_after_clearing_is_kept(self):
        """Clearing runs before routing, so the hook's own write must survive it."""
        kwargs = {"model": "smart-router", "metadata": {"pre_routing_selected_model": "stale"}}

        clear_pre_routing_selection(kwargs)
        record_pre_routing_selection(kwargs, "tier1")

        assert get_pre_routing_selection(kwargs) == "tier1"


class TestOrderedFallbackLookupGroups:
    def test_tier_first_then_requested_group_deduped(self):
        from litellm.router_utils.fallback_event_handlers import (
            PRE_ROUTING_SELECTED_MODEL_KEY,
            fallback_lookup_groups,
        )

        kwargs = {"litellm_metadata": {PRE_ROUTING_SELECTED_MODEL_KEY: "tier1"}}
        assert fallback_lookup_groups(kwargs, "smart-router") == ("tier1", "smart-router")
        assert fallback_lookup_groups(kwargs, "tier1") == ("tier1",)
        assert fallback_lookup_groups({}, "smart-router") == ("smart-router",)
        assert fallback_lookup_groups({}, None) == ()

    def test_session_remap_keeps_the_bound_router_between_tier_and_requested_group(self):
        from litellm.router_utils.fallback_event_handlers import (
            PRE_ROUTING_SELECTED_MODEL_KEY,
            fallback_lookup_groups,
        )

        kwargs = {
            "litellm_metadata": {
                PRE_ROUTING_SELECTED_MODEL_KEY: "tier1",
                "model_group": "smart-router",
            }
        }

        assert fallback_lookup_groups(kwargs, "requested-model") == (
            "tier1",
            "smart-router",
            "requested-model",
        )
        assert fallback_lookup_groups({"metadata": {"model_group": []}}, "requested-model") == ("requested-model",)

    def test_fallback_hop_resumes_the_original_groups_chain_last(self):
        from litellm.router_utils.fallback_event_handlers import fallback_lookup_groups

        kwargs = {"metadata": {"model_group": "fb1", "original_model_group": "primary"}}

        assert fallback_lookup_groups(kwargs, "fb1") == ("fb1", "primary")
        assert fallback_lookup_groups({"metadata": {"original_model_group": 42}}, "fb1") == ("fb1",)

    def test_first_resolving_group_wins_and_generic_idx_survives_a_miss(self):
        from litellm.router_utils.fallback_event_handlers import (
            get_fallback_model_group_for_lookup_groups,
        )

        fallbacks = [{"tier1": ["backup-a"]}, {"smart-router": ["backup-b"]}, {"*": ["backup-c"]}]
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier1", "smart-router")) == (["backup-a"], None)
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier9", "smart-router")) == (["backup-b"], None)
        assert get_fallback_model_group_for_lookup_groups(fallbacks, ("tier9", "no-such")) == (["backup-c"], 2)
        assert get_fallback_model_group_for_lookup_groups([{"tier1": ["backup-a"]}], ("no", "nope")) == (None, None)


def test_attempted_fallback_targets_deduplicates_skips_in_order_and_keeps_first_reason() -> None:
    attempted: Final = AttemptedFallbackTargets()
    attempted.record_skip("a", FallbackSkipReason.NOT_AUTHORIZED)
    attempted.record_skip("a", FallbackSkipReason.NOT_AUTHORIZED)
    attempted.record_skip("b", FallbackSkipReason.OVER_BUDGET)
    attempted.record_skip("a", FallbackSkipReason.OVER_BUDGET)

    assert attempted.skipped == (
        SkippedFallbackTarget("a", FallbackSkipReason.NOT_AUTHORIZED),
        SkippedFallbackTarget("b", FallbackSkipReason.OVER_BUDGET),
    )


def test_get_fallback_model_group_matches_provider_prefixed_key():
    """A bare model group routed via a wildcard (e.g. "gpt-4o" through
    "openai/*") must match a fallback keyed on the provider-prefixed name,
    which is the form the Admin UI offers for wildcard routes."""
    fallbacks = [{"openai/gpt-4o": ["claude-3-haiku"]}]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="gpt-4o")

    assert fallback_model_group == ["claude-3-haiku"]


def test_get_fallback_model_group_exact_match_beats_prefixed_match():
    fallbacks = [
        {"openai/gpt-4o": ["claude-3-haiku"]},
        {"gpt-4o": ["gemini-1.5-flash"]},
    ]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="gpt-4o")

    assert fallback_model_group == ["gemini-1.5-flash"]


def test_get_fallback_model_group_prefixed_match_ignores_unknown_models():
    """Provider inference fails for unknown bare names - the lookup must not
    raise and must fall through to the generic fallback."""
    fallbacks = [
        {"openai/some-model": ["claude-3-haiku"]},
        {"*": ["gemini-1.5-flash"]},
    ]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="some-unknown-model-xyz")

    assert fallback_model_group == ["gemini-1.5-flash"]


def test_get_fallback_model_group_prefixed_match_skips_prefixed_model_group():
    """An already-prefixed model group must not double-prefix."""
    fallbacks = [{"openai/openai/gpt-4o": ["claude-3-haiku"]}]

    fallback_model_group, _ = get_fallback_model_group(fallbacks=fallbacks, model_group="openai/gpt-4o")

    assert fallback_model_group is None


def test_get_fallback_model_group_never_resolves_a_provider_without_a_prefixed_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An alias-style group name has no provider, and resolving it prints the SDK's provider-list banner,
    so the lookup only infers a provider when some key is spelled <provider>/<group>."""

    resolver: Final = MagicMock(return_value=("my-alias", "openai", None, None))
    monkeypatch.setattr(get_llm_provider_logic, "get_llm_provider", resolver)
    fallbacks: Final = [{"gpt-5.5-pro": ["claude-sonnet-4-6"]}, {"*": ["gpt-5.5-mini"]}]

    assert get_fallback_model_group(fallbacks=fallbacks, model_group="my-alias") == (["gpt-5.5-mini"], 1)
    resolver.assert_not_called()


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function")
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)


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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_record_pre_routing_selection_writes_only_the_internal_bucket():
    """The Anthropic request's own metadata field must never carry the tier stamp."""
    kwargs = {"metadata": {"user_id": "u1"}, "litellm_metadata": {}}

    record_pre_routing_selection(kwargs, "tier-x")

    assert kwargs["litellm_metadata"] == {PRE_ROUTING_SELECTED_MODEL_KEY: "tier-x"}
    assert kwargs["metadata"] == {"user_id": "u1"}


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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


@pytest.mark.asyncio
async def test_log_success_fallback_event():
    """
    Tests that successful fallback events are logged correctly
    """
    original_model_group = "gpt-3.5-turbo"
    kwargs = {"messages": [{"role": "user", "content": "Hello, world!"}]}
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    logger = CustomTestLogger()
    litellm.callbacks = [logger]

    # This test mainly checks if the function runs without errors
    await log_success_fallback_event(original_model_group, kwargs, original_exception)

    await asyncio.sleep(0.5)
    assert len(logger.success_fallback_events) == 1
    assert len(logger.failure_fallback_events) == 0
    assert logger.success_fallback_events[0] == (
        original_model_group,
        kwargs,
        original_exception,
    )


@pytest.mark.asyncio
async def test_log_failure_fallback_event():
    """
    Tests that failed fallback events are logged correctly
    """
    original_model_group = "gpt-3.5-turbo"
    kwargs = {"messages": [{"role": "user", "content": "Hello, world!"}]}
    original_exception = litellm.exceptions.InternalServerError(
        message="Simulated error",
        llm_provider="openai",
        model="gpt-3.5-turbo",
    )

    logger = CustomTestLogger()
    litellm.callbacks = [logger]

    # This test mainly checks if the function runs without errors
    await log_failure_fallback_event(original_model_group, kwargs, original_exception)

    await asyncio.sleep(0.5)

    assert len(logger.failure_fallback_events) == 1
    assert len(logger.success_fallback_events) == 0
    assert logger.failure_fallback_events[0] == (
        original_model_group,
        kwargs,
        original_exception,
    )


class CustomTestLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.success_fallback_events = []
        self.failure_fallback_events = []

    async def log_success_fallback_event(self, original_model_group, kwargs, original_exception):
        print(
            "in log_success_fallback_event for original_model_group: ",
            original_model_group,
        )
        self.success_fallback_events.append((original_model_group, kwargs, original_exception))

    async def log_failure_fallback_event(self, original_model_group, kwargs, original_exception):
        print(
            "in log_failure_fallback_event for original_model_group: ",
            original_model_group,
        )
        self.failure_fallback_events.append((original_model_group, kwargs, original_exception))


@pytest.mark.asyncio
async def test_a_role_less_request_is_not_retried_on_the_fallback_group() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "house",
                "litellm_params": {"model": "house-model", "custom_llm_provider": "openai", "mock_response": "ok"},
            },
            {"model_name": "backup", "litellm_params": {"model": "openai/gpt-5.6", "mock_response": "ok"}},
        ],
        fallbacks=[{"house": ["backup"]}],
        fallback_on_unsupported_params=False,
    )

    with pytest.raises(litellm.MalformedRequestFieldError, match="Missing required parameter") as exc_info:
        await router.acompletion(model="house", messages=json.loads('[{"content": "hi"}]'))
    assert "Fallback to backup" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_async_fallback_skips_a_target_the_rate_limit_check_rejects() -> None:
    checked: Final = []  # mutable-ok: records the targets the check saw

    async def refuse(*, model: str, request_kwargs: Mapping[str, object], ctx: object) -> bool:
        checked.append(model)
        return False

    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "api_base": "http://127.0.0.1:9"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "mock_response": "served"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
        fallback_rate_limit_check=refuse,
    )
    with pytest.raises(
        Exception, match="Fallback to fallback was skipped because the caller is over its per-model rate limit"
    ):
        await router.acompletion(model="primary", messages=[{"role": "user", "content": "hi"}])
    assert checked == ["fallback"]


@pytest.mark.asyncio
async def test_run_async_fallback_serves_a_target_the_rate_limit_check_admits() -> None:
    async def admit(*, model: str, request_kwargs: Mapping[str, object], ctx: object) -> bool:
        return True

    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "api_base": "http://127.0.0.1:9"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "mock_response": "served"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}],
        num_retries=0,
        fallback_rate_limit_check=admit,
    )
    response: Final = await router.acompletion(model="primary", messages=[{"role": "user", "content": "hi"}])
    assert response.choices[0].message.content == "served"


@pytest.mark.asyncio
async def test_run_async_fallback_reserves_a_repeated_target_once() -> None:
    checked: Final = []  # mutable-ok: records the targets the check saw

    async def admit(*, model: str, request_kwargs: Mapping[str, object], ctx: object) -> bool:
        checked.append(model)
        return True

    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "api_base": "http://127.0.0.1:9"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "api_base": "http://127.0.0.1:9"},
            },
        ],
        fallbacks=[{"primary": ["fallback", "fallback"]}],
        num_retries=0,
        fallback_rate_limit_check=admit,
    )
    with pytest.raises(Exception, match="model group 'primary' failed"):
        await router.acompletion(model="primary", messages=[{"role": "user", "content": "hi"}])
    assert checked == ["fallback"]


@pytest.mark.asyncio
async def test_run_async_fallback_never_re_reserves_the_requested_model_group() -> None:
    checked: Final = []  # mutable-ok: records the targets the check saw

    async def admit(*, model: str, request_kwargs: Mapping[str, object], ctx: object) -> bool:
        checked.append(model)
        return True

    router: Final = Router(
        model_list=[
            {
                "model_name": "primary",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "api_base": "http://127.0.0.1:9"},
            },
            {
                "model_name": "fallback",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "k", "api_base": "http://127.0.0.1:9"},
            },
        ],
        fallbacks=[{"primary": ["fallback"]}, {"fallback": ["primary"]}],
        num_retries=0,
        fallback_rate_limit_check=admit,
    )
    with pytest.raises(Exception, match="model group 'primary' failed"):
        await router.acompletion(model="primary", messages=[{"role": "user", "content": "hi"}])
    assert checked == ["fallback"]


@pytest.fixture(autouse=True)
def _python_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.rust_bridge import catalog
    from litellm.rust_bridge.configuration import Rollout

    monkeypatch.setattr(
        catalog,
        "RULES",
        tuple(
            rule
            for rule in catalog.RULES
            if not (rule.route is catalog.Route.MESSAGES and rule.rollout is Rollout.RUST_REQUIRED)
        ),
    )
