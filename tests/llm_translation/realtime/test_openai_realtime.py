from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.types.realtime import RealtimeQueryParams


def test_realtime_query_params_construction():
    """
    Test that query params are constructed correctly by the proxy server logic
    """
    from litellm.types.realtime import RealtimeQueryParams

    # Test case 1: intent is None (should not be included)
    model = "gpt-4o-realtime-preview"
    intent = None

    query_params: RealtimeQueryParams = {"model": model}
    if intent is not None:
        query_params["intent"] = intent

    assert "model" in query_params
    assert query_params["model"] == model
    assert "intent" not in query_params

    # Test case 2: intent is provided (should be included)
    intent = "chat"
    query_params2: RealtimeQueryParams = {"model": model}
    if intent is not None:
        query_params2["intent"] = intent

    assert "model" in query_params2
    assert query_params2["model"] == model
    assert "intent" in query_params2
    assert query_params2["intent"] == intent


@pytest.mark.asyncio
async def test_realtime_query_params_use_normalized_model_name(monkeypatch):
    """
    Ensure query params overwrite model with normalized provider model name.
    """
    from litellm.realtime_api import main as realtime_main

    mock_async_realtime = AsyncMock()
    monkeypatch.setattr(
        realtime_main,
        "openai_realtime",
        MagicMock(async_realtime=mock_async_realtime),
    )

    def fake_get_llm_provider(model, api_base=None, api_key=None):
        return ("gpt-4o-realtime-preview", "openai", None, None)

    monkeypatch.setattr(realtime_main, "get_llm_provider", fake_get_llm_provider)

    query_params: RealtimeQueryParams = {
        "model": "openai/gpt-4o-realtime-preview",
        "intent": "chat",
    }

    await realtime_main._arealtime(
        model="openai/gpt-4o-realtime-preview",
        websocket=MagicMock(),
        api_key="sk-test",
        query_params=query_params,
        litellm_logging_obj=MagicMock(),
    )

    called_kwargs = mock_async_realtime.call_args.kwargs
    assert called_kwargs["query_params"]["model"] == "gpt-4o-realtime-preview"
    assert called_kwargs["query_params"]["intent"] == "chat"


@pytest.mark.asyncio
async def test_realtime_query_params_preserve_missing_model(monkeypatch):
    """
    OpenAI-compatible transcription clients can connect with only
    ?intent=transcription and send the model in session.update. Do not add
    model= back into the upstream query params when the client omitted it.
    """
    from litellm.realtime_api import main as realtime_main

    mock_async_realtime = AsyncMock()
    monkeypatch.setattr(
        realtime_main,
        "openai_realtime",
        MagicMock(async_realtime=mock_async_realtime),
    )

    def fake_get_llm_provider(model, api_base=None, api_key=None):
        return ("gpt-realtime-whisper", "openai", None, None)

    monkeypatch.setattr(realtime_main, "get_llm_provider", fake_get_llm_provider)

    query_params: RealtimeQueryParams = {"intent": "transcription"}

    await realtime_main._arealtime(
        model="gpt-realtime-whisper",
        websocket=MagicMock(),
        api_key="sk-test",
        query_params=query_params,
        litellm_logging_obj=MagicMock(),
    )

    called_kwargs = mock_async_realtime.call_args.kwargs
    assert called_kwargs["query_params"] == {"intent": "transcription"}
