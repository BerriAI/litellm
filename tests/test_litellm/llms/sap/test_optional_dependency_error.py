import pytest

from litellm import APIConnectionError

@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_sap_raises_optional_dependency_error_completion(monkeypatch, sync_mode):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    import litellm

    model = "sap/gpt-4o"
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(APIConnectionError) as err:
        if sync_mode:
            litellm.completion(model=model, messages=messages)
        else:
            await litellm.acompletion(model=model, messages=messages)
    assert "gen-ai-hub package is required" in str(err.value)

@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_sap_raises_optional_dependency_error_embedding(monkeypatch, sync_mode):
    monkeypatch.delenv("PYTHONPATH", raising=False)

    import litellm

    model = "sap/text-embedding-3-small"
    text = "Hello"

    with pytest.raises(APIConnectionError) as err:
        if sync_mode:
            litellm.embedding(model=model, input=text)
        else:
            await litellm.aembedding(model=model, input=text)
    assert "gen-ai-hub package is required" in str(err.value)