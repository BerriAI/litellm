import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.fallback_utils import async_completion_with_fallbacks


def _ok():
    """Minimal non-None response that satisfies the `if response is not None` check."""
    return MagicMock()


@pytest.mark.asyncio
async def test_dict_fallback_does_not_mutate_original():
    """Reusing the same fallback dict across calls must not lose the model key.

    The primary model must fail so the loop actually reaches the dict fallback
    entry and exercises the mutation fix.
    """
    fallback = {"model": "gpt-4o-mini", "temperature": 0.5}
    fallback_original = {"model": "gpt-4o-mini", "temperature": 0.5}

    # Call 1: primary raises → fallback dict path runs → returns ok
    # Call 2: same pattern — model key must still be present in fallback
    mock = AsyncMock(side_effect=[
        Exception("primary failed"),  # call 1, primary "gpt-4o"
        _ok(),                        # call 1, fallback "gpt-4o-mini"
        Exception("primary failed"),  # call 2, primary "gpt-4o"
        _ok(),                        # call 2, fallback "gpt-4o-mini"
    ])

    with patch("litellm.acompletion", new=mock):
        await async_completion_with_fallbacks(
            model="gpt-4o",
            kwargs={"fallbacks": [fallback]},
        )
        assert fallback == fallback_original, (
            f"fallback dict mutated after call 1: got {fallback}"
        )

        await async_completion_with_fallbacks(
            model="gpt-4o",
            kwargs={"fallbacks": [fallback]},
        )
        assert fallback == fallback_original, (
            f"fallback dict mutated after call 2: got {fallback}"
        )


@pytest.mark.asyncio
async def test_dict_fallback_nested_mutable_not_mutated():
    """Nested mutable values in the fallback dict must survive after a call."""
    extra_headers = {"X-Custom": "value"}
    fallback = {"model": "gpt-4o-mini", "extra_headers": extra_headers}

    mock = AsyncMock(side_effect=[
        Exception("primary failed"),
        _ok(),
    ])

    with patch("litellm.acompletion", new=mock):
        await async_completion_with_fallbacks(
            model="gpt-4o",
            kwargs={"fallbacks": [fallback]},
        )
        assert fallback["model"] == "gpt-4o-mini"
        assert fallback["extra_headers"] is extra_headers
        assert extra_headers == {"X-Custom": "value"}
