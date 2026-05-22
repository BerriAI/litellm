import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.fallback_utils import async_completion_with_fallbacks


@pytest.mark.asyncio
async def test_dict_fallback_does_not_mutate_original():
    """Reusing the same fallback dict across calls must not lose the model key."""
    fallback = {"model": "gpt-4o-mini", "temperature": 0.5}
    fallback_original_copy = dict(fallback)

    mock_response = AsyncMock()
    mock_response.return_value = {"choices": [{"message": {"content": "ok"}}]}

    with patch("litellm.acompletion", new=mock_response):
        await async_completion_with_fallbacks(
            model="gpt-4o",
            kwargs={"fallbacks": [fallback]},
        )
        # Original fallback dict must be unchanged after the call
        assert fallback == fallback_original_copy, (
            "fallback dict was mutated: "
            f"expected {fallback_original_copy}, got {fallback}"
        )

        # Second call with the same dict must still work (model key still present)
        await async_completion_with_fallbacks(
            model="gpt-4o",
            kwargs={"fallbacks": [fallback]},
        )
        assert fallback == fallback_original_copy


@pytest.mark.asyncio
async def test_dict_fallback_nested_mutable_not_mutated():
    """Nested mutable values in the fallback dict must not be modified."""
    extra_headers = {"X-Custom": "value"}
    fallback = {"model": "gpt-4o-mini", "extra_headers": extra_headers}

    mock_response = AsyncMock()
    mock_response.return_value = {"choices": [{"message": {"content": "ok"}}]}

    with patch("litellm.acompletion", new=mock_response):
        await async_completion_with_fallbacks(
            model="gpt-4o",
            kwargs={"fallbacks": [fallback]},
        )
        assert fallback["model"] == "gpt-4o-mini"
        assert fallback["extra_headers"] is extra_headers
