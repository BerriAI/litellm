"""
Tests for `litellm.expose_router_debug_in_errors`.

The Router historically appended internal config names (model_group,
fallback_model_group, fallback failure detail, deployment timeouts,
context_window_fallbacks dict, etc.) onto the message of the exception
it re-raises. That message is then surfaced to clients by
ProxyException, leaking the proxy's internal wiring.

These tests verify that with the flag OFF (default) those strings do
NOT appear in the raised exception's message, and with the flag ON the
upstream debug behavior is restored.

Five leak sites were gated in `litellm/router.py`:

1. Deployment timeout debug after `litellm.Timeout`
2. ContextWindowExceededError fallback hint
3. ContentPolicyViolationError fallback hint
4. "No fallback model group found for..." when fallbacks dict misses
5. "Received Model Group=...\\nAvailable Model Group Fallbacks=..."
   (always fires on terminal raise from the fallback orchestrator)

Site 5 is the broadest — it fires for every failing call that goes
through the fallback orchestrator with any non-context-window /
non-content-policy error, regardless of whether `fallbacks` is set.
The tests below mainly exercise sites 2 and 5, which together prove
the gate works for both ContextWindow-typed and generic errors.
"""

from __future__ import annotations

import pytest

import litellm
from litellm import Router

_RECEIVED_MODEL_GROUP_PHRASE = "Received Model Group="
_AVAILABLE_FALLBACKS_PHRASE = "Available Model Group Fallbacks="
_CONTEXT_WINDOW_HINT_PHRASE = "context_window_fallbacks="
_INTERNAL_MODEL_GROUP_NAME = "all-anthropic/claude-secret-internal"


def _router_with_rate_limit_failure() -> Router:
    return Router(
        model_list=[
            {
                "model_name": _INTERNAL_MODEL_GROUP_NAME,
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "litellm.RateLimitError",
                },
                "model_info": {"id": "secret-deployment-id"},
            },
        ],
        num_retries=0,
    )


def _router_with_context_window_failure() -> Router:
    return Router(
        model_list=[
            {
                "model_name": _INTERNAL_MODEL_GROUP_NAME,
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "litellm.ContextWindowExceededError",
                },
                "model_info": {"id": "secret-deployment-id"},
            },
        ],
        num_retries=0,
    )


@pytest.fixture(autouse=True)
def _reset_expose_flag():
    """Each test starts with the flag in its default (off) state."""
    original = litellm.expose_router_debug_in_errors
    litellm.expose_router_debug_in_errors = False
    try:
        yield
    finally:
        litellm.expose_router_debug_in_errors = original


def test_flag_defaults_off():
    assert litellm.expose_router_debug_in_errors is False


# --- Site 5: "Received Model Group=..." on terminal raise --------------------


@pytest.mark.asyncio
async def test_default_does_not_leak_received_model_group():
    router = _router_with_rate_limit_failure()
    with pytest.raises(litellm.RateLimitError) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
        )
    msg = excinfo.value.message
    assert _RECEIVED_MODEL_GROUP_PHRASE not in msg, msg
    assert _AVAILABLE_FALLBACKS_PHRASE not in msg, msg
    assert _INTERNAL_MODEL_GROUP_NAME not in msg, msg


@pytest.mark.asyncio
async def test_flag_on_leaks_received_model_group():
    litellm.expose_router_debug_in_errors = True
    router = _router_with_rate_limit_failure()
    with pytest.raises(litellm.RateLimitError) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
        )
    msg = excinfo.value.message
    assert _RECEIVED_MODEL_GROUP_PHRASE in msg, msg
    assert _AVAILABLE_FALLBACKS_PHRASE in msg, msg
    assert _INTERNAL_MODEL_GROUP_NAME in msg, msg


# --- Site 2: ContextWindowExceededError fallback hint ------------------------


@pytest.mark.asyncio
async def test_default_does_not_leak_context_window_fallback_hint():
    router = _router_with_context_window_failure()
    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
        )
    msg = excinfo.value.message
    assert _CONTEXT_WINDOW_HINT_PHRASE not in msg, msg
    assert _RECEIVED_MODEL_GROUP_PHRASE not in msg, msg
    assert _INTERNAL_MODEL_GROUP_NAME not in msg, msg


@pytest.mark.asyncio
async def test_flag_on_leaks_context_window_fallback_hint():
    litellm.expose_router_debug_in_errors = True
    router = _router_with_context_window_failure()
    with pytest.raises(litellm.ContextWindowExceededError) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
        )
    msg = excinfo.value.message
    assert _CONTEXT_WINDOW_HINT_PHRASE in msg, msg
    # Site 5 also fires for ContextWindow errors that exit the
    # orchestrator without fallback resolution, so the model_group
    # name should leak when the flag is on.
    assert _INTERNAL_MODEL_GROUP_NAME in msg, msg


# --- Site 4: "No fallback model group found..." when fallbacks miss ---------


@pytest.mark.asyncio
async def test_default_does_not_leak_when_no_fallback_group_found():
    router = Router(
        model_list=[
            {
                "model_name": _INTERNAL_MODEL_GROUP_NAME,
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "litellm.RateLimitError",
                },
                "model_info": {"id": "secret-deployment-id"},
            },
        ],
        # Fallbacks defined for a different model_group, so resolution
        # ends with fallback_model_group=None and hits site 4.
        fallbacks=[{"some-other-group": ["some-other-target"]}],
        num_retries=0,
    )
    with pytest.raises(litellm.RateLimitError) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
        )
    msg = excinfo.value.message
    assert "No fallback model group found" not in msg, msg
    assert "some-other-group" not in msg, msg
    assert _INTERNAL_MODEL_GROUP_NAME not in msg, msg


@pytest.mark.asyncio
async def test_flag_on_leaks_when_no_fallback_group_found():
    litellm.expose_router_debug_in_errors = True
    router = Router(
        model_list=[
            {
                "model_name": _INTERNAL_MODEL_GROUP_NAME,
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key",
                    "mock_response": "litellm.RateLimitError",
                },
                "model_info": {"id": "secret-deployment-id"},
            },
        ],
        fallbacks=[{"some-other-group": ["some-other-target"]}],
        num_retries=0,
    )
    with pytest.raises(litellm.RateLimitError) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
        )
    msg = excinfo.value.message
    assert "No fallback model group found" in msg, msg
    assert _INTERNAL_MODEL_GROUP_NAME in msg, msg
