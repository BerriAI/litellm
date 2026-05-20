"""Tests for litellm_core_utils.fallback_utils module."""

from unittest.mock import patch

import pytest

import litellm
from litellm.litellm_core_utils.fallback_utils import (
    async_completion_with_fallbacks,
)


@pytest.mark.asyncio
async def test_async_completion_with_fallbacks_does_not_mutate_caller_fallbacks():
    """Regression test for https://github.com/BerriAI/litellm/issues/28251.

    ``async_completion_with_fallbacks`` previously called ``.pop("model", ...)`` on
    the caller's fallback dict directly, permanently stripping the ``model`` key
    on the first invocation and breaking every subsequent call that reused the
    same config.
    """

    fallbacks = [{"model": "gpt-4o-mini", "api_key": "fallback-key"}]
    original_fallbacks = [{**f} for f in fallbacks]

    call_count = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_count["n"] += 1
        # Fail on the primary model so we exercise the dict-fallback path.
        if call_count["n"] == 1:
            raise RuntimeError("primary boom")
        return "ok"

    with patch.object(litellm, "acompletion", side_effect=fake_acompletion):
        result = await async_completion_with_fallbacks(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            kwargs={"fallbacks": fallbacks},
        )

    assert result == "ok"
    assert (
        fallbacks == original_fallbacks
    ), "async_completion_with_fallbacks must not mutate the caller's fallback dicts"
    assert fallbacks[0]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_async_completion_with_fallbacks_dict_fallback_is_reusable():
    """A single fallback config must remain usable across multiple invocations."""

    fallbacks = [{"model": "gpt-4o-mini", "api_key": "fallback-key"}]

    call_count = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_count["n"] += 1
        # Fail the primary on every call, succeed on the dict fallback.
        if kwargs.get("model") == "gpt-4o":
            raise RuntimeError("primary boom")
        assert kwargs.get("model") == "gpt-4o-mini"
        assert kwargs.get("api_key") == "fallback-key"
        return "ok"

    with patch.object(litellm, "acompletion", side_effect=fake_acompletion):
        for _ in range(3):
            result = await async_completion_with_fallbacks(
                model="gpt-4o",
                messages=[{"role": "user", "content": "hi"}],
                kwargs={"fallbacks": fallbacks},
            )
            assert result == "ok"

    assert fallbacks[0]["model"] == "gpt-4o-mini"
    assert fallbacks[0]["api_key"] == "fallback-key"


@pytest.mark.asyncio
async def test_async_completion_with_fallbacks_safe_memory_mode_still_safe(
    monkeypatch,
):
    """The mutation guard must hold even when ``litellm.safe_memory_mode=True``.

    ``safe_deep_copy`` returns its argument unchanged when safe_memory_mode is
    enabled (see ``litellm_core_utils.core_helpers.safe_deep_copy``). Without
    the extra ``dict(fallback)`` shallow copy, ``.pop("model", ...)`` would
    still mutate the caller's dict on this code path.
    """

    monkeypatch.setattr(litellm, "safe_memory_mode", True)

    fallbacks = [{"model": "gpt-4o-mini", "api_key": "fallback-key"}]
    original_fallbacks = [{**f} for f in fallbacks]

    call_count = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("primary boom")
        return "ok"

    with patch.object(litellm, "acompletion", side_effect=fake_acompletion):
        result = await async_completion_with_fallbacks(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            kwargs={"fallbacks": fallbacks},
        )

    assert result == "ok"
    assert fallbacks == original_fallbacks, (
        "safe_memory_mode=True must not regress the mutation guard "
        "(safe_deep_copy returns the original object in this mode)"
    )
