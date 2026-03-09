import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.router import Router


def test_get_silent_experiment_kwargs():
    """
    Test _get_silent_experiment_kwargs returns isolated kwargs with silent experiment metadata.
    Direct call for router code coverage.
    """
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    kwargs = {"metadata": {"foo": "bar"}, "litellm_call_id": "call-123"}
    result = router._get_silent_experiment_kwargs(**kwargs)
    assert result["metadata"]["is_silent_experiment"] is True
    assert result["metadata"]["foo"] == "bar"
    assert "litellm_call_id" not in result


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
    with patch.object(router, "completion", return_value=None):
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

    # Mock litellm.completion for primary call and litellm.acompletion for shadow call
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "hello"}}])
    mock_completion = MagicMock(return_value=mock_response)
    mock_acompletion = AsyncMock(return_value=mock_response)

    # Patch at the litellm module level
    with patch.object(litellm, "completion", mock_completion), patch.object(
        litellm, "acompletion", mock_acompletion
    ):
        response = router.completion(
            model="primary-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert response.choices[0].message.content == "hello"

        # The sync background call uses a thread pool. We might need to wait a bit.
        import time

        time.sleep(0.5)

        # Should have 1 call to completion (primary) and 1 call to acompletion (shadow)
        assert mock_completion.call_count == 1
        assert mock_acompletion.call_count == 1

        primary_args, primary_kwargs = mock_completion.call_args
        silent_args, silent_kwargs = mock_acompletion.call_args

        # Verify no silent_model in any call
        assert "silent_model" not in primary_kwargs
        assert "silent_model" not in silent_kwargs

        assert silent_kwargs.get("metadata", {}).get("is_silent_experiment") is True
        assert silent_kwargs["model"] == "openai/gpt-4"
        assert silent_kwargs.get("stream") is False

        assert (
            primary_kwargs.get("metadata", {}).get("is_silent_experiment") is not True
        )
        assert primary_kwargs["model"] == "openai/gpt-3.5-turbo"


def test_silent_experiment_forces_stream_false():
    """Verify that _get_silent_experiment_kwargs() sets stream=False even if stream=True in kwargs."""
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    kwargs = {"stream": True, "metadata": {}}
    result = router._get_silent_experiment_kwargs(**kwargs)
    assert result["stream"] is False


def test_silent_experiment_sets_zero_retries():
    """Verify num_retries=0 in silent kwargs."""
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    kwargs = {"num_retries": 3, "metadata": {}}
    result = router._get_silent_experiment_kwargs(**kwargs)
    assert result["num_retries"] == 0


@pytest.mark.asyncio
async def test_silent_experiment_streaming_primary_triggers_shadow():
    """
    Mock a streaming primary request and verify the silent model call is made with stream=False
    and both primary + silent calls complete.
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

    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "hello"}}])
    mock_acompletion = AsyncMock(return_value=mock_response)

    with patch.object(litellm, "acompletion", mock_acompletion):
        await router.acompletion(
            model="primary-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )

        await asyncio.sleep(0.1)

        assert mock_acompletion.call_count == 2
        calls = mock_acompletion.call_args_list

        silent_call = next(
            (
                c
                for c in calls
                if c[1].get("metadata", {}).get("is_silent_experiment") is True
            ),
            None,
        )
        assert silent_call is not None
        assert silent_call[1].get("stream") is False


def test_silent_experiment_sync_uses_async_path():
    """Verify the sync _silent_experiment_completion calls acompletion (not completion)."""
    model_list = [
        {
            "model_name": "gpt-3.5-turbo",
            "litellm_params": {"model": "gpt-3.5-turbo", "api_key": "fake-key"},
        },
    ]
    router = Router(model_list=model_list)
    messages = [{"role": "user", "content": "hi"}]

    with patch.object(
        router, "acompletion", new_callable=AsyncMock, return_value=None
    ) as mock_acompletion:
        router._silent_experiment_completion(
            silent_model="gpt-3.5-turbo",
            messages=messages,
        )
        mock_acompletion.assert_called_once()
