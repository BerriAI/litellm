import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.router import Router


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
    # Original metadata must NOT be mutated
    assert "is_silent_experiment" not in kwargs["metadata"]
    assert kwargs["metadata"]["litellm_parent_otel_span"] is mock_span
    assert kwargs["metadata"]["user_api_key_auth"] is mock_auth


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
            (
                c
                for c in call_args_list
                if c[1].get("metadata", {}).get("is_silent_experiment") is True
            ),
            None,
        )
        assert silent_call is not None
        assert silent_call[1]["model"] == "openai/gpt-4"

        # Find the primary call
        primary_call = next(
            (
                c
                for c in call_args_list
                if not c[1].get("metadata", {}).get("is_silent_experiment")
            ),
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
    with patch.object(litellm, "acompletion", mock_acompletion_mock), patch.object(
        litellm, "completion", mock_completion_mock
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
            (
                c
                for c in call_args_list
                if c[1].get("metadata", {}).get("is_silent_experiment") is True
            ),
            None,
        )
        assert silent_call is not None
        assert silent_call[1]["model"] == "openai/gpt-4"
        # Verify model_group is set to the silent model name for correct metric attribution
        assert silent_call[1]["metadata"]["model_group"] == "silent-model"
