"""Unit tests for the AdaptiveRouterPostCallHook."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.router_strategy.adaptive_router.config import (
    ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY,
    SIGNAL_GATE_MIN_MESSAGES,
)
from litellm.router_strategy.adaptive_router.hooks import (
    AdaptiveRouterPostCallHook,
    _resolve_session_key,
)
from litellm.router_strategy.adaptive_router.signals import Turn


def _make_hook(claim: bool = True) -> AdaptiveRouterPostCallHook:
    fake_router = MagicMock()
    fake_router.record_turn = AsyncMock()
    fake_router.claim_or_check_owner = MagicMock(return_value=claim)
    return AdaptiveRouterPostCallHook(adaptive_router=fake_router)


def _resp_with_content(text: str, tool_calls=None):
    """Build a ModelResponse-like object with a single assistant message."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _long_messages(user_text: str = "ask"):
    """Return a message list at the SIGNAL_GATE_MIN_MESSAGES threshold."""
    base = [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second turn"},
    ]
    base.append({"role": "user", "content": user_text})
    # Pad to threshold if needed.
    while len(base) < SIGNAL_GATE_MIN_MESSAGES:
        base.append({"role": "user", "content": "filler"})
    return base


def _kwargs(
    *,
    messages=None,
    chosen="fast",
    extra_metadata=None,
    extra_litellm_params=None,
):
    metadata = {ADAPTIVE_ROUTER_CHOSEN_MODEL_KEY: chosen} if chosen else {}
    if extra_metadata:
        metadata.update(extra_metadata)
    lp = {"metadata": metadata}
    if extra_litellm_params:
        lp.update(extra_litellm_params)
    return {
        "model": "anthropic/claude-opus-4-7",
        "messages": messages if messages is not None else _long_messages(),
        "litellm_params": lp,
    }


# ---- _resolve_session_key ------------------------------------------------


def test_resolve_session_key_honors_litellm_session_id_on_litellm_params():
    key = _resolve_session_key({"litellm_params": {"litellm_session_id": "sess-A"}})
    assert key == "sess-A"


def test_resolve_session_key_honors_metadata_session_id():
    key = _resolve_session_key(
        {"litellm_params": {"metadata": {"session_id": "sess-B"}}}
    )
    assert key == "sess-B"


def test_resolve_session_key_returns_none_when_no_messages():
    assert _resolve_session_key({"litellm_params": {}}) is None
    assert _resolve_session_key({"litellm_params": {}, "messages": []}) is None


def test_resolve_session_key_derives_stable_hash_from_first_message():
    msgs = [{"role": "user", "content": "Hello, world"}]
    k1 = _resolve_session_key({"messages": msgs})
    k2 = _resolve_session_key({"messages": list(msgs)})
    assert k1 == k2
    assert k1 and len(k1) == 64  # sha256 hex


def test_resolve_session_key_does_not_prefix_sk():
    key = _resolve_session_key({"messages": [{"role": "user", "content": "hi"}]})
    assert key and not key.startswith("sk_")


def test_resolve_session_key_segments_by_identity_fields():
    """Same first message but different api keys must yield different keys."""
    msgs = [{"role": "user", "content": "same prompt"}]
    k_team_a = _resolve_session_key(
        {
            "messages": msgs,
            "litellm_params": {
                "metadata": {
                    "user_api_key_hash": "hash-A",
                    "user_api_key_team_id": "team-1",
                }
            },
        }
    )
    k_team_b = _resolve_session_key(
        {
            "messages": msgs,
            "litellm_params": {
                "metadata": {
                    "user_api_key_hash": "hash-B",
                    "user_api_key_team_id": "team-2",
                }
            },
        }
    )
    assert k_team_a != k_team_b


def test_resolve_session_key_changes_when_first_message_changes():
    k1 = _resolve_session_key({"messages": [{"role": "user", "content": "alpha"}]})
    k2 = _resolve_session_key({"messages": [{"role": "user", "content": "beta"}]})
    assert k1 != k2


# ---- _record gating -----------------------------------------------------


@pytest.mark.asyncio
async def test_hook_skips_when_below_signal_gate():
    """Conversations shorter than SIGNAL_GATE_MIN_MESSAGES should be ignored."""
    hook = _make_hook()
    short = [{"role": "user", "content": "hi"}]
    assert len(short) < SIGNAL_GATE_MIN_MESSAGES  # sanity
    kwargs = _kwargs(messages=short)
    await hook.async_log_success_event(kwargs, _resp_with_content("ok"), 0.0, 1.0)
    hook.adaptive_router.record_turn.assert_not_awaited()
    hook.adaptive_router.claim_or_check_owner.assert_not_called()


@pytest.mark.asyncio
async def test_hook_skips_when_no_messages():
    hook = _make_hook()
    kwargs = _kwargs(messages=[])
    await hook.async_log_success_event(kwargs, _resp_with_content("ok"), 0.0, 1.0)
    hook.adaptive_router.record_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_hook_skips_when_chosen_model_missing_from_metadata():
    hook = _make_hook()
    kwargs = _kwargs(chosen=None)
    await hook.async_log_success_event(kwargs, _resp_with_content("ok"), 0.0, 1.0)
    hook.adaptive_router.record_turn.assert_not_awaited()
    hook.adaptive_router.claim_or_check_owner.assert_not_called()


@pytest.mark.asyncio
async def test_hook_skips_when_owner_cache_mismatch():
    """A different model owns this conversation -> no attribution."""
    hook = _make_hook(claim=False)
    kwargs = _kwargs(chosen="fast")
    await hook.async_log_success_event(kwargs, _resp_with_content("ok"), 0.0, 1.0)
    hook.adaptive_router.claim_or_check_owner.assert_called_once()
    hook.adaptive_router.record_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_hook_records_turn_when_owner_claims():
    hook = _make_hook(claim=True)
    kwargs = _kwargs(chosen="smart", messages=_long_messages("ask"))
    await hook.async_log_success_event(
        kwargs, _resp_with_content("answer here"), 0.0, 1.0
    )
    call = hook.adaptive_router.record_turn.await_args
    assert call.kwargs["model_name"] == "smart"
    turn: Turn = call.kwargs["turn"]
    assert turn.user_content == "ask"
    assert turn.assistant_content == "answer here"
    assert turn.response_status == 200


@pytest.mark.asyncio
async def test_hook_uses_explicit_session_id_when_provided():
    """Explicit `litellm_session_id` is forwarded as the session key."""
    hook = _make_hook()
    kwargs = _kwargs(
        chosen="fast",
        extra_litellm_params={"litellm_session_id": "explicit-sess"},
    )
    await hook.async_log_success_event(kwargs, _resp_with_content("ok"), 0.0, 1.0)
    args, _ = hook.adaptive_router.claim_or_check_owner.call_args
    assert args[0] == "explicit-sess"
    assert hook.adaptive_router.record_turn.await_args.kwargs["session_id"] == (
        "explicit-sess"
    )


@pytest.mark.asyncio
async def test_hook_passes_tool_calls_through():
    hook = _make_hook()
    tc = {"name": "search", "arguments": '{"q":"x"}'}
    kwargs = _kwargs(chosen="fast")
    await hook.async_log_success_event(
        kwargs, _resp_with_content("calling tool", tool_calls=[tc]), 0.0, 1.0
    )
    turn: Turn = hook.adaptive_router.record_turn.await_args.kwargs["turn"]
    assert turn.tool_calls == [tc]


@pytest.mark.asyncio
async def test_hook_swallows_exceptions_from_record_turn():
    hook = _make_hook()
    hook.adaptive_router.record_turn.side_effect = RuntimeError("boom")
    kwargs = _kwargs(chosen="fast")
    # Must NOT raise — signal recording must never break a request.
    await hook.async_log_success_event(kwargs, _resp_with_content("ok"), 0.0, 1.0)


@pytest.mark.asyncio
async def test_hook_failure_event_uses_status_code_from_exception():
    hook = _make_hook()
    exc = MagicMock()
    exc.status_code = 429
    kwargs = _kwargs(chosen="fast")
    kwargs["exception"] = exc
    await hook.async_log_failure_event(kwargs, None, 0.0, 1.0)
    turn: Turn = hook.adaptive_router.record_turn.await_args.kwargs["turn"]
    assert turn.response_status == 429


# ---- async_post_call_success_hook (response header surfacing) ----------


@pytest.mark.asyncio
async def test_post_call_success_hook_sets_response_header():
    hook = _make_hook()
    response = MagicMock()
    response._hidden_params = {}

    await hook.async_post_call_success_hook(
        data={"metadata": {"adaptive_router_chosen_model": "smart"}},
        user_api_key_dict=MagicMock(),
        response=response,
    )

    assert (
        response._hidden_params["additional_headers"]["x-litellm-adaptive-router-model"]
        == "smart"
    )


@pytest.mark.asyncio
async def test_post_call_success_hook_preserves_existing_additional_headers():
    hook = _make_hook()
    response = MagicMock()
    response._hidden_params = {"additional_headers": {"x-existing": "keep-me"}}

    await hook.async_post_call_success_hook(
        data={"metadata": {"adaptive_router_chosen_model": "fast"}},
        user_api_key_dict=MagicMock(),
        response=response,
    )

    assert response._hidden_params["additional_headers"]["x-existing"] == "keep-me"
    assert (
        response._hidden_params["additional_headers"]["x-litellm-adaptive-router-model"]
        == "fast"
    )


@pytest.mark.asyncio
async def test_post_call_success_hook_noop_when_metadata_missing_key():
    hook = _make_hook()
    response = MagicMock()
    response._hidden_params = {}

    await hook.async_post_call_success_hook(
        data={"metadata": {"litellm_session_id": "sess-A"}},
        user_api_key_dict=MagicMock(),
        response=response,
    )

    assert response._hidden_params == {}


@pytest.mark.asyncio
async def test_post_call_success_hook_noop_when_no_metadata():
    hook = _make_hook()
    response = MagicMock()
    response._hidden_params = {}

    await hook.async_post_call_success_hook(
        data={},
        user_api_key_dict=MagicMock(),
        response=response,
    )

    assert response._hidden_params == {}


@pytest.mark.asyncio
async def test_post_call_success_hook_noop_when_hidden_params_not_dict():
    hook = _make_hook()

    class _NoHiddenParams:
        pass

    response = _NoHiddenParams()

    await hook.async_post_call_success_hook(
        data={"metadata": {"adaptive_router_chosen_model": "smart"}},
        user_api_key_dict=MagicMock(),
        response=response,
    )

    assert not hasattr(response, "_hidden_params")
