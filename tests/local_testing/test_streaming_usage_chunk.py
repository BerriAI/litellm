"""
Regression test for https://github.com/BerriAI/litellm/issues/28735

When stream_options={"include_usage": True} is set, the final usage chunk
must have choices: [] per the OpenAI Chat Completions streaming spec.
Previously model_response_creator() was setting a default choice on this
synthetic chunk, violating the spec.
"""
import pytest
import litellm


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o-mini",
        "gpt-4o",
    ],
)
@pytest.mark.parametrize(
    "sync",
    [True, False],
)
@pytest.mark.asyncio
async def test_openai_stream_options_usage_chunk_choices_empty(model, sync):
    """Verify the usage chunk has choices: [] per the OpenAI spec."""
    litellm.enable_preview_features = True
    chunks = []
    if sync:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "say GM"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in response:
            chunks.append(chunk)
    else:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "say GM"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in response:
            chunks.append(chunk)

    usage_chunks = [c for c in chunks if getattr(c, "usage", None) is not None]
    assert usage_chunks, "No usage chunk found"
    last_usage_chunk = usage_chunks[-1]
    assert last_usage_chunk.choices == [], (
        f"Usage chunk choices must be [] but got {last_usage_chunk.choices!r}"
    )
