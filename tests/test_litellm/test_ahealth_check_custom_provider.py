import types
from unittest.mock import patch, AsyncMock

import pytest

import litellm


@pytest.mark.asyncio
async def test_ahealth_check_preserves_custom_model_with_explicit_provider_and_api_base():
    """
    When both custom_llm_provider and api_base are set, ahealth_check should
    preserve the full user-specified model string and not call get_llm_provider.
    """

    # Full model string that would normally be split by get_llm_provider
    model = "openai/gpt-oss-20b"

    model_params = {
        "model": model,
        "api_key": "sk-test",  # dummy key; call is mocked
        "api_base": "https://example.openai.compatible.endpoint",  # explicit base
        "custom_llm_provider": "openai",  # explicit provider
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Return value for litellm.acompletion that mimics a response with headers
    mocked_response = types.SimpleNamespace(_hidden_params={"headers": {}})

    # Patch get_llm_provider on the module where it's used to ensure it's NOT called
    with patch("litellm.main.get_llm_provider", side_effect=AssertionError("get_llm_provider should not be called when both custom_llm_provider and api_base are set")):
        # Patch acompletion to capture the call and return a minimal response
        with patch.object(litellm, "acompletion", new=AsyncMock(return_value=mocked_response)) as mock_acomp:
            result = await litellm.ahealth_check(model_params=model_params, mode="chat")

            # Health check should return a dict (based on headers)
            assert isinstance(result, dict)

            # Ensure acompletion was invoked once
            assert mock_acomp.await_count == 1

            # Verify it preserved the full model string
            called_kwargs = mock_acomp.await_args.kwargs
            assert called_kwargs.get("model") == model

    # No-op assertion comment to re-trigger CI
    assert True
