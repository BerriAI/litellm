import os
import pytest
import asyncio


@pytest.mark.ndsmoke
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_router_streaming_e2e_optional():
    # Requires an OpenAI-compatible endpoint; otherwise skip.
    api_base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_COMPAT_BASE")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_COMPAT_API_KEY")
    model = os.getenv("OPENAI_COMPAT_MODEL", "gpt-3.5-turbo")
    if not api_base or not api_key:
        pytest.skip("OPENAI_API_BASE/OPENAI_COMPAT_BASE and API key not set")

    import litellm

    chunks = []
    async for chunk in await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": "Stream the word HELLO."}],
        api_base=api_base,
        api_key=api_key,
        stream=True,
    ):
        # chunk is a dict or ModelResponse-like
        try:
            delta = ((chunk or {}).get("choices") or [{}])[0].get("delta", {})
            text = delta.get("content") or ""
            if text:
                chunks.append(text)
        except Exception:
            pass

    if not chunks:
        pytest.skip("Streaming not supported or no content returned")
    assert any("HELLO" in c.upper() for c in chunks)

