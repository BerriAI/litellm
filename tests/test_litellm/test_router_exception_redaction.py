"""
Tests for `litellm.expose_router_debug_in_errors`.

The Router historically appended internal config names (model_group,
fallback_model_group, fallback failure detail, deployment timeouts,
context_window_fallbacks dict, etc.) onto the message of the exception
it re-raises. That message is then surfaced to clients by
ProxyException, leaking the proxy's internal wiring.

The flag defaults to True to preserve historical behavior (no
breaking change for existing deployments). Set it to False to redact
those strings from the raised exception's message.

These tests verify that with the flag ON (default) the historical
leak strings appear in the raised exception's message, and with the
flag OFF the proxy's internal wiring is redacted.

Five leak sites are gated in `litellm/router.py`:

1. Deployment timeout debug after `litellm.Timeout`
2. ContextWindowExceededError fallback hint
3. ContentPolicyViolationError fallback hint
4. "No fallback model group found for..." when fallbacks dict misses
5. "Received Model Group=...\\nAvailable Model Group Fallbacks=..."
   (always fires on terminal raise from the fallback orchestrator)

Site 5 is the broadest — it fires for every failing call that goes
through the fallback orchestrator with any non-context-window /
non-content-policy error, regardless of whether `fallbacks` is set.
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
    """Each test starts with the flag in its default (on) state."""
    original = litellm.expose_router_debug_in_errors
    litellm.expose_router_debug_in_errors = True
    try:
        yield
    finally:
        litellm.expose_router_debug_in_errors = original


def test_flag_defaults_on():
    assert litellm.expose_router_debug_in_errors is True


# --- Site 5: "Received Model Group=..." on terminal raise --------------------


@pytest.mark.asyncio
async def test_flag_off_does_not_leak_received_model_group():
    litellm.expose_router_debug_in_errors = False
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
async def test_default_leaks_received_model_group():
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
async def test_flag_off_does_not_leak_context_window_fallback_hint():
    litellm.expose_router_debug_in_errors = False
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
async def test_default_leaks_context_window_fallback_hint():
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
    # name leaks under the default behavior.
    assert _INTERNAL_MODEL_GROUP_NAME in msg, msg


# --- Site 4: "No fallback model group found..." when fallbacks miss ---------


@pytest.mark.asyncio
async def test_flag_off_does_not_leak_when_no_fallback_group_found():
    litellm.expose_router_debug_in_errors = False
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
async def test_default_leaks_when_no_fallback_group_found():
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


# --- Site 1: Deployment timeout debug on litellm.Timeout --------------------


def _router_with_plain_deployment() -> Router:
    """Plain deployment, no preconfigured mock_response — caller supplies via kwargs.

    Exception instances cannot live in `model_list[*].litellm_params` because
    `Router.__init__` deep-copies model_list and several LiteLLM exceptions
    (Timeout, ContentPolicyViolationError) require positional args that
    `__reduce__` cannot reconstruct. Passing the trigger at call-site bypasses
    the deepcopy entirely.
    """
    return Router(
        model_list=[
            {
                "model_name": _INTERNAL_MODEL_GROUP_NAME,
                "litellm_params": {"model": "gpt-4o", "api_key": "key"},
                "model_info": {"id": "secret-deployment-id"},
            },
        ],
        num_retries=0,
    )


@pytest.mark.asyncio
async def test_flag_off_does_not_leak_deployment_timeout_debug():
    litellm.expose_router_debug_in_errors = False
    router = _router_with_plain_deployment()
    with pytest.raises(litellm.Timeout) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
            mock_timeout=True,
            timeout=0.001,
        )
    msg = excinfo.value.message
    assert "Deployment Info: request_timeout:" not in msg, msg


@pytest.mark.asyncio
async def test_default_leaks_deployment_timeout_debug():
    router = _router_with_plain_deployment()
    with pytest.raises(litellm.Timeout) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
            mock_timeout=True,
            timeout=0.001,
        )
    msg = excinfo.value.message
    assert "Deployment Info: request_timeout:" in msg, msg


# --- Site 3: ContentPolicyViolationError fallback hint (no fallback set) ----


def _content_policy_error() -> litellm.ContentPolicyViolationError:
    return litellm.ContentPolicyViolationError(
        message="mocked policy violation",
        model="gpt-4o",
        llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_flag_off_does_not_leak_content_policy_fallback_hint():
    litellm.expose_router_debug_in_errors = False
    router = _router_with_plain_deployment()
    with pytest.raises(litellm.ContentPolicyViolationError) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
            mock_response=_content_policy_error(),
        )
    msg = excinfo.value.message
    assert "content_policy_fallback=" not in msg, msg
    assert _INTERNAL_MODEL_GROUP_NAME not in msg, msg


@pytest.mark.asyncio
async def test_default_leaks_content_policy_fallback_hint():
    router = _router_with_plain_deployment()
    with pytest.raises(litellm.ContentPolicyViolationError) as excinfo:
        await router.acompletion(
            model=_INTERNAL_MODEL_GROUP_NAME,
            messages=[{"role": "user", "content": "hi"}],
            mock_response=_content_policy_error(),
        )
    msg = excinfo.value.message
    assert "content_policy_fallback=" in msg, msg
    assert _INTERNAL_MODEL_GROUP_NAME in msg, msg
