import asyncio
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.router import Router


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

    # Mock litellm.acompletion
    mock_acompletion = MagicMock()
    # Create a future that resolves to a ModelResponse
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "hello"}}])
    future = asyncio.Future()
    future.set_result(mock_response)
    mock_acompletion.return_value = future

    with patch("litellm.acompletion", mock_acompletion):
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

    # Mock litellm.completion
    mock_completion = MagicMock()
    mock_response = litellm.ModelResponse(choices=[{"message": {"content": "hello"}}])
    mock_completion.return_value = mock_response

    with patch("litellm.completion", mock_completion):
        response = router.completion(
            model="primary-model",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert response.choices[0].message.content == "hello"

        # The sync background call uses a thread pool. We might need to wait a bit.
        import time

        time.sleep(0.5)

        # Should have 2 calls
        assert mock_completion.call_count == 2

        call_args_list = mock_completion.call_args_list

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
