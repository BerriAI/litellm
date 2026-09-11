import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.router import Router, _has_mcp_tool, _is_silent_experiment_marker


class _NonCopyableSpan:
    """Mimics an OTel Span which raises on deepcopy, forcing safe_deep_copy
    to fall back to the original reference."""

    def __deepcopy__(self, memo):
        raise TypeError("OTel spans cannot be deepcopied")


class _FakeUserAPIKeyAuth:
    """Mimics UserAPIKeyAuth which contains a parent_otel_span that is not
    deepcopy-able. This is what actually causes safe_deep_copy to fail for
    the metadata dict in production — safe_deep_copy handles the top-level
    litellm_parent_otel_span specially (pops it before copying), but does
    NOT handle user_api_key_auth.parent_otel_span inside it."""

    def __init__(self, key_alias, parent_otel_span):
        self.key_alias = key_alias
        self.parent_otel_span = parent_otel_span

    def __deepcopy__(self, memo):
        raise TypeError("Contains OTel span that cannot be deepcopied")


def test_get_silent_experiment_kwargs():
    """
    Test _get_silent_experiment_kwargs returns isolated kwargs with silent experiment metadata.

    Uses a non-copyable user_api_key_auth (mimicking the real proxy scenario)
    so that safe_deep_copy falls back to the original metadata reference —
    exercising the identity-check fix path.
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    mock_span = _NonCopyableSpan()
    mock_auth = _FakeUserAPIKeyAuth(
        key_alias="HaneefKeyNonTeamProd",
        parent_otel_span=mock_span,
    )
    kwargs = {
        "metadata": {
            "foo": "bar",
            "litellm_parent_otel_span": mock_span,
            "user_api_key_auth": mock_auth,
        },
        "litellm_call_id": "call-123",
        "stream": True,
        "proxy_server_request": {"body": {"model": "test"}},
    }
    result = router._get_silent_experiment_kwargs(**kwargs)
    assert result["metadata"]["is_silent_experiment"] is True
    assert result["metadata"]["foo"] == "bar"
    assert "litellm_call_id" not in result
    # stream must be forced to False so callbacks fire in background
    assert result["stream"] is False
    # proxy_server_request must be preserved for spend log metadata
    assert "proxy_server_request" in result
    # CRITICAL: metadata must be a DIFFERENT dict object than the original,
    # so that setting model_group / is_silent_experiment on the silent dict
    # doesn't corrupt the primary call's metadata.
    assert result["metadata"] is not kwargs["metadata"]
    # OTel span must be stripped from the silent copy — it's not safe to use
    # across event loops (silent experiment runs in a new event loop).
    assert "litellm_parent_otel_span" not in result["metadata"]
    # Original metadata must NOT be mutated — must carry the real span,
    # not safe_deep_copy's temporary "placeholder" string.
    assert "is_silent_experiment" not in kwargs["metadata"]
    assert kwargs["metadata"]["litellm_parent_otel_span"] is mock_span
    assert kwargs["metadata"]["user_api_key_auth"] is mock_auth
    # Shallow copy must preserve user_api_key_auth so the silent experiment
    # can attribute billing / spend logs to the correct key/team.
    assert result["metadata"]["user_api_key_auth"] is mock_auth


def test_silent_experiment_completion_direct():
    """
    Test _silent_experiment_completion directly (for router code coverage).
    Mocks router.completion to avoid real API call.
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "hi"}]
    with patch.object(router, "acompletion", new_callable=AsyncMock, return_value=None):
        router._silent_experiment_completion(
            silent_model="gpt-3.5-turbo",
            messages=messages,
        )


@pytest.mark.asyncio
async def test_silent_experiment_acompletion_direct():
    """
    Test _silent_experiment_acompletion directly (for router code coverage).
    Mocks router.acompletion to avoid real API call.
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "hi"}]
    with patch.object(router, "acompletion", new_callable=AsyncMock, return_value=None):
        await router._silent_experiment_acompletion(
            silent_model="gpt-3.5-turbo",
            messages=messages,
        )


@pytest.mark.asyncio
async def test_router_silent_experiment_acompletion():
    """
    Test that silent_model triggers a background acompletion call
    and that the silent_model parameter is stripped from both calls.
    """
    model_list = [
        {
            "model_name": "primary-model",
            "litellm_params": {
                "model": "openai/gpt-3.5-turbo",
                "api_key": "fake-key",
                "silent_model": "silent-model",
            },
        },
        {
            "model_name": "silent-model",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "fake-key",
            },
        },
    ]

    router = Router(model_list=model_list)

    # Use AsyncMock for async function mocking
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "hello"}}])
    mock_acompletion = AsyncMock(return_value=mock_response)

    # Patch at the litellm.router module level where it's imported and used
    with patch.object(litellm, "acompletion", mock_acompletion):
        response = await router.acompletion(
            model="primary-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert response.choices[0].message.content == "hello"

        # Give the background task a moment to trigger (it's an asyncio task)
        await asyncio.sleep(0.1)

        # Should have 2 calls: one for primary, one for silent
        assert mock_acompletion.call_count == 2

        # Check call arguments
        call_args_list = mock_acompletion.call_args_list

        # Verify no silent_model in any call to litellm.acompletion
        for call in call_args_list:
            args, kwargs = call
            assert "silent_model" not in kwargs
            if "metadata" in kwargs:
                # One call should have is_silent_experiment=True
                pass

        # Find the silent call
        silent_call = next(
            (c for c in call_args_list if c[1].get("metadata", {}).get("is_silent_experiment") is True),
            None,
        )
        assert silent_call is not None
        assert silent_call[1]["model"] == "openai/gpt-4"

        # Find the primary call
        primary_call = next(
            (c for c in call_args_list if not c[1].get("metadata", {}).get("is_silent_experiment")),
            None,
        )
        assert primary_call is not None
        assert primary_call[1]["model"] == "openai/gpt-3.5-turbo"


def test_router_silent_experiment_completion():
    """
    Test that silent_model triggers a background completion call (sync)
    and that the silent_model parameter is stripped.
    """
    model_list = [
        {
            "model_name": "primary-model",
            "litellm_params": {
                "model": "openai/gpt-3.5-turbo",
                "api_key": "fake-key",
                "silent_model": "silent-model",
            },
        },
        {
            "model_name": "silent-model",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "fake-key",
            },
        },
    ]

    router = Router(model_list=model_list)

    # Mock litellm.acompletion
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "hello"}}])

    # We need an async mock for acompletion
    async def mock_acompletion(*args, **kwargs):
        return mock_response

    mock_acompletion_mock = AsyncMock(side_effect=mock_acompletion)
    mock_completion_mock = MagicMock(return_value=mock_response)

    # Patch at the litellm module level
    with (
        patch.object(litellm, "acompletion", mock_acompletion_mock),
        patch.object(litellm, "completion", mock_completion_mock),
    ):
        response = router.completion(
            model="primary-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert response.choices[0].message.content == "hello"

        # The sync background call uses a thread pool. We might need to wait.
        time.sleep(2.0)

        # Should have 1 acompletion call (the silent background call)
        assert mock_acompletion_mock.call_count == 1

        call_args_list = mock_acompletion_mock.call_args_list

        # Verify no silent_model in any call
        for call in call_args_list:
            args, kwargs = call
            assert "silent_model" not in kwargs

        # Find the silent call
        silent_call = next(
            (c for c in call_args_list if c[1].get("metadata", {}).get("is_silent_experiment") is True),
            None,
        )
        assert silent_call is not None
        assert silent_call[1]["model"] == "openai/gpt-4"
        # Verify model_group is set to the silent model name for correct metric attribution
        assert silent_call[1]["metadata"]["model_group"] == "silent-model"


# ---------------------------------------------------------------------------
# Generic router path (Responses API / Anthropic Messages) — issues #31888, #34890
# ---------------------------------------------------------------------------


def _generic_silent_model_list():
    return [
        {
            "model_name": "primary-model",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "fake-key",
                "silent_model": "silent-model",
            },
        },
        {
            "model_name": "silent-model",
            "litellm_params": {
                "model": "openai/gpt-4o",
                "api_key": "fake-key",
            },
        },
    ]


def _split_generic_calls(mock):
    """Return (primary_call, silent_call) from a mocked generic handler."""
    silent = next(
        (c for c in mock.call_args_list if (c.kwargs.get("litellm_metadata") or {}).get("is_silent_experiment")),
        None,
    )
    primary = next(
        (c for c in mock.call_args_list if not (c.kwargs.get("litellm_metadata") or {}).get("is_silent_experiment")),
        None,
    )
    return primary, silent


@pytest.mark.asyncio
async def test_router_silent_experiment_aresponses():
    """
    Regression for #31888: silent_model on a deployment must fire a background
    aresponses call when the primary request goes through /v1/responses.
    """
    mock_aresponses = AsyncMock(return_value=MagicMock())
    mock_aresponses.__name__ = "aresponses"

    router = Router(model_list=_generic_silent_model_list())
    router.aresponses = router.factory_function(mock_aresponses, call_type="aresponses")
    await router.aresponses(
        model="primary-model",
        input=[{"role": "user", "content": "hi"}],
        metadata={"trace": "user-supplied"},
    )
    await asyncio.sleep(0.1)

    assert mock_aresponses.call_count == 2
    primary_call, silent_call = _split_generic_calls(mock_aresponses)
    assert primary_call is not None
    assert silent_call is not None

    assert primary_call.kwargs["model"] == "openai/gpt-4o-mini"
    assert primary_call.kwargs["litellm_metadata"]["model_group"] == "primary-model"

    assert silent_call.kwargs["model"] == "openai/gpt-4o"
    # model_group is overridden so metrics/logging attribute the mirror to the silent model
    assert silent_call.kwargs["litellm_metadata"]["model_group"] == "silent-model"

    # The router-only setting must never reach the provider handler (#34890)
    for call in mock_aresponses.call_args_list:
        assert "silent_model" not in call.kwargs

    # `metadata` is the OpenAI Responses request-body field: the marker must not be
    # written there, and the caller's value must pass through untouched.
    assert silent_call.kwargs["metadata"] == {"trace": "user-supplied"}
    assert primary_call.kwargs["metadata"] == {"trace": "user-supplied"}


@pytest.mark.asyncio
async def test_router_silent_experiment_anthropic_messages():
    """
    Regression for #34890: same contract for /v1/messages. Anthropic validates the
    `metadata` body field strictly, so the marker must live in litellm_metadata.
    """
    mock_messages = AsyncMock(return_value=MagicMock())
    mock_messages.__name__ = "anthropic_messages"

    router = Router(model_list=_generic_silent_model_list())
    router.aanthropic_messages = router.factory_function(mock_messages, call_type="anthropic_messages")
    await router.aanthropic_messages(
        model="primary-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=16,
    )
    await asyncio.sleep(0.1)

    assert mock_messages.call_count == 2
    primary_call, silent_call = _split_generic_calls(mock_messages)
    assert primary_call is not None
    assert silent_call is not None
    assert primary_call.kwargs["model"] == "openai/gpt-4o-mini"
    assert silent_call.kwargs["model"] == "openai/gpt-4o"
    assert silent_call.kwargs["litellm_metadata"]["model_group"] == "silent-model"
    for call in mock_messages.call_args_list:
        assert "silent_model" not in call.kwargs
        assert "is_silent_experiment" not in (call.kwargs.get("metadata") or {})


@pytest.mark.asyncio
async def test_router_silent_experiment_generic_does_not_corrupt_primary_metadata():
    """
    The primary call's litellm_metadata must not see the silent marker or the
    silent model_group, even when deepcopy falls back to a shared reference.
    """
    mock_aresponses = AsyncMock(return_value=MagicMock())
    mock_aresponses.__name__ = "aresponses"

    router = Router(model_list=_generic_silent_model_list())
    router.aresponses = router.factory_function(mock_aresponses, call_type="aresponses")
    litellm_metadata = {
        "user_api_key_auth": _FakeUserAPIKeyAuth(
            key_alias="primary-key",
            parent_otel_span=_NonCopyableSpan(),
        )
    }
    await router.aresponses(
        model="primary-model",
        input=[{"role": "user", "content": "hi"}],
        litellm_metadata=litellm_metadata,
    )
    await asyncio.sleep(0.1)

    assert mock_aresponses.call_count == 2
    primary_call, silent_call = _split_generic_calls(mock_aresponses)
    assert primary_call is not None
    assert silent_call is not None
    assert primary_call.kwargs["litellm_metadata"]["model_group"] == "primary-model"
    assert "is_silent_experiment" not in primary_call.kwargs["litellm_metadata"]
    assert silent_call.kwargs["litellm_metadata"]["model_group"] == "silent-model"


@pytest.mark.asyncio
async def test_router_silent_experiment_skips_non_allowlisted_generic_call_types():
    """
    Only side-effect-free inference call types are mirrored. The same generic
    helper serves file/fine-tuning/passthrough calls that must not be replayed
    against the silent deployment.
    """
    sentinel_response = MagicMock()
    mock_file_content = AsyncMock(return_value=sentinel_response)
    mock_file_content.__name__ = "afile_content"

    router = Router(model_list=_generic_silent_model_list())
    router.afile_content = router.factory_function(mock_file_content, call_type="afile_content")
    response = await router.afile_content(model="primary-model", file_id="file-123")
    await asyncio.sleep(0.1)

    assert response is sentinel_response
    assert mock_file_content.call_count == 1
    assert "silent_model" not in mock_file_content.call_args.kwargs


def test_silent_experiment_generic_kwargs_skips_recursion_and_mcp():
    """
    The generic-path snapshot must return None when the request is already a
    silent experiment (marker in either litellm_metadata or metadata) or when it
    carries MCP tooling that the provider may execute.
    """
    router = Router(model_list=_generic_silent_model_list())
    base = {"input": [{"role": "user", "content": "hi"}]}

    assert router._silent_experiment_generic_kwargs(**base, litellm_metadata={"is_silent_experiment": True}) is None
    assert router._silent_experiment_generic_kwargs(**base, metadata={"is_silent_experiment": True}) is None
    assert router._silent_experiment_generic_kwargs(**base, mcp_servers=[{"type": "url", "url": "x"}]) is None
    assert (
        router._silent_experiment_generic_kwargs(
            **base, tools=[{"type": "mcp", "server_label": "x", "require_approval": "never"}]
        )
        is None
    )

    snapshot = router._silent_experiment_generic_kwargs(**base, tools=[{"type": "web_search"}])
    assert snapshot is not None
    assert snapshot["litellm_metadata"]["is_silent_experiment"] is True
    assert snapshot["stream"] is False
    assert "metadata" not in snapshot


def test_silent_experiment_generic_kwargs_snapshots_before_mutation():
    """
    Regression: the snapshot is taken synchronously, so metadata merged into the
    caller's kwargs afterwards (deployment tags etc.) never reaches the mirror.
    """
    router = Router(model_list=_generic_silent_model_list())
    litellm_metadata = {"model_group": "primary-model"}
    snapshot = router._silent_experiment_generic_kwargs(input="hi", litellm_metadata=litellm_metadata)
    assert snapshot is not None
    litellm_metadata["tags"] = ["primary-deployment-tag"]
    assert "tags" not in snapshot["litellm_metadata"]


@pytest.mark.asyncio
async def test_silent_experiment_ageneric_error_is_caught():
    """
    A failing mirror must never propagate to the caller.
    """
    router = Router(model_list=_generic_silent_model_list())

    with patch.object(
        router,
        "_ageneric_api_call_with_fallbacks",
        new_callable=AsyncMock,
        side_effect=Exception("downstream failure"),
    ):
        result = await router._silent_experiment_ageneric(
            silent_model="silent-model",
            original_function=litellm.aresponses,
            silent_kwargs={"input": [{"role": "user", "content": "hi"}]},
        )

    assert result is None


def test_is_silent_experiment_marker():
    assert _is_silent_experiment_marker({"is_silent_experiment": True}) is True
    assert _is_silent_experiment_marker({"is_silent_experiment": False}) is False
    assert _is_silent_experiment_marker({}) is False
    assert _is_silent_experiment_marker(None) is False
    assert _is_silent_experiment_marker("is_silent_experiment") is False


def test_has_mcp_tool():
    assert _has_mcp_tool([{"type": "mcp", "server_label": "x"}]) is True
    assert _has_mcp_tool([{"type": "web_search"}, {"type": "mcp"}]) is True
    assert _has_mcp_tool([{"type": "web_search"}]) is False
    assert _has_mcp_tool([]) is False
    assert _has_mcp_tool(None) is False
    assert _has_mcp_tool("mcp") is False
