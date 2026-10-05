import asyncio

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_empty_choices_chat_completion_is_not_cached():
    litellm.cache = Cache(type="local")
    msgs = [{"role": "user", "content": "repro-empty-choices-cache"}]
    try:
        first = await litellm.acompletion(
            model="gpt-4o",
            messages=msgs,
            mock_response=ModelResponse(choices=[]),
            caching=True,
        )
        assert first.choices == []
        await asyncio.sleep(0.2)
        second = await litellm.acompletion(
            model="gpt-4o",
            messages=msgs,
            mock_response="hi",
            caching=True,
        )
        assert second.choices[0].message.content == "hi"
    finally:
        litellm.cache = None
