import litellm
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_text_completion_include_usage(sync_mode):
    """Test text completion with include_usage"""
    last_chunk = None
    if sync_mode:
        response = await litellm.atext_completion(
            model="gpt-3.5-turbo",
            prompt="Hello, world!",
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in response:
            print(chunk)
            last_chunk = chunk
    else:
        response = litellm.text_completion(
            model="gpt-3.5-turbo",
            prompt="Hello, world!",
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in response:
            print(chunk)
            last_chunk = chunk

    assert last_chunk is not None
    assert last_chunk.usage is not None
    assert last_chunk.usage.prompt_tokens > 0
    assert last_chunk.usage.completion_tokens > 0
    assert last_chunk.usage.total_tokens > 0
