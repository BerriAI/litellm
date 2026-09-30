import pytest
from unittest.mock import AsyncMock, patch
from litellm.litellm_core_utils.fallback_utils import async_completion_with_fallbacks


@pytest.mark.asyncio
async def test_async_completion_with_top_level_fallbacks():
    """Verify that top-level fallbacks keyword argument is properly extracted and used."""
    mock_response = AsyncMock()
    mock_response.choices = []

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        # First call fails, second succeeds
        mock_acompletion.side_effect = [
            Exception("Primary model failed"),
            mock_response,
        ]

        res = await async_completion_with_fallbacks(
            model="primary-failing-model",
            fallbacks=["secondary-fallback-model"],
            messages=[{"role": "user", "content": "hello"}],
        )

        assert mock_acompletion.call_count == 2
        # First attempt with primary model
        assert mock_acompletion.call_args_list[0].kwargs["model"] == "primary-failing-model"
        # Second attempt with fallback model
        assert mock_acompletion.call_args_list[1].kwargs["model"] == "secondary-fallback-model"
        assert res is not None


@pytest.mark.asyncio
async def test_async_completion_with_nested_fallbacks():
    """Verify backwards compatibility with nested kwargs fallbacks."""
    mock_response = AsyncMock()
    mock_response.choices = []

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = [
            Exception("Primary model failed"),
            mock_response,
        ]

        res = await async_completion_with_fallbacks(
            model="primary-failing-model",
            kwargs={"fallbacks": ["nested-fallback-model"]},
            messages=[{"role": "user", "content": "hello"}],
        )

        assert mock_acompletion.call_count == 2
        assert mock_acompletion.call_args_list[0].kwargs["model"] == "primary-failing-model"
        assert mock_acompletion.call_args_list[1].kwargs["model"] == "nested-fallback-model"
        assert res is not None
