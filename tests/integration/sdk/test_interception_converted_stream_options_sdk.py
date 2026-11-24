from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration._support import interception_vendor as iv
from integration._support.wire import Wire, wire_server
from openai.types.responses import WebSearchToolParam

import litellm
from litellm.integrations.websearch_interception.handler import WebSearchInterceptionLogger
from litellm.responses.streaming_iterator import (
    MockResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import OutputTextDeltaEvent, ResponsesAPIStreamingResponse


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Iterator[Wire]:
    monkeypatch.setattr(litellm, "callbacks", [WebSearchInterceptionLogger(enabled_providers=["azure"])])
    with wire_server(iv.respond) as served:
        yield served


def _text(events: Sequence[ResponsesAPIStreamingResponse]) -> str:
    return "".join(event.delta for event in events if isinstance(event, OutputTextDeltaEvent))


def _assert_one_clean_call(wire: Wire, name: str) -> None:
    received: Final = iv.received(wire.drain(), name)
    (call,) = received.model_calls
    assert call.get("stream") is not True and "stream_options" not in call, call
    assert [key for key in call if key.startswith("_")] == [], call
    assert received.searches == (), received


def test_sync_responses_stream_with_web_search_keeps_stream_options_off_azure(wire: Wire) -> None:
    name: Final = iv.deployment("plain")
    stream: Final = litellm.responses(
        model=f"azure/{name}",
        input=f"Search for {name}",
        tools=[WebSearchToolParam(type="web_search")],
        stream=True,
        stream_options={"include_obfuscation": True},
        api_base=wire.url,
        api_key=iv.AZURE_KEY,
        api_version=iv.API_VERSION,
        num_retries=0,
    )
    assert isinstance(stream, (SyncResponsesAPIStreamingIterator, MockResponsesAPIStreamingIterator)), stream
    events: Final = tuple(stream)
    assert _text(events) == iv.plain(name), events
    _assert_one_clean_call(wire, name)


async def test_async_responses_stream_with_web_search_keeps_stream_options_off_azure(wire: Wire) -> None:
    name: Final = iv.deployment("plain")
    stream: Final = await litellm.aresponses(
        model=f"azure/{name}",
        input=f"Search for {name}",
        tools=[WebSearchToolParam(type="web_search")],
        stream=True,
        stream_options={"include_obfuscation": True},
        api_base=wire.url,
        api_key=iv.AZURE_KEY,
        api_version=iv.API_VERSION,
        num_retries=0,
    )
    assert isinstance(stream, (ResponsesAPIStreamingIterator, MockResponsesAPIStreamingIterator)), stream
    events: Final = tuple([event async for event in stream])
    assert _text(events) == iv.plain(name), events
    _assert_one_clean_call(wire, name)
