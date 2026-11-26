import time
from unittest.mock import patch

import pytest

import litellm
from litellm.responses.streaming_iterator import MockResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse


@pytest.mark.asyncio
async def test_aresponses_mock_response_non_streaming():
    with patch("httpx.AsyncClient.send", side_effect=AssertionError("Unexpected HTTP request")):
        response = await litellm.aresponses(
            model="openai/gpt-4.1",
            input="hi",
            mock_response="hello world",
            stream=False,
            api_key="mock-key",
        )
    assert isinstance(response, ResponsesAPIResponse)
    assert response.output[0].content[0].text == "hello world"


@pytest.mark.asyncio
async def test_aresponses_mock_response_streaming():
    with patch("httpx.AsyncClient.send", side_effect=AssertionError("Unexpected HTTP request")):
        stream_response = await litellm.aresponses(
            model="openai/gpt-4.1",
            input="hi",
            mock_response="hello world",
            stream=True,
            api_key="mock-key",
        )
    assert isinstance(stream_response, MockResponsesAPIStreamingIterator)

    events = []
    collected_text = ""
    async for event in stream_response:
        events.append(event)
        if event.type == "response.output_text.delta":
            collected_text += event.delta

    assert collected_text == "hello world"
    event_types = [event.type for event in events]
    assert "response.created" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.completed" in event_types


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_aresponses_mock_delay(stream: bool):
    delay = 0.2
    start_time = time.time()
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        input="hi",
        mock_response="delayed answer",
        mock_delay=delay,
        stream=stream,
        api_key="mock-key",
    )
    elapsed = time.time() - start_time
    assert elapsed >= 0.18, f"Expected elapsed >= 0.18s, got {elapsed:.2f}s"

    if stream:
        assert isinstance(response, MockResponsesAPIStreamingIterator)
        text = "".join([e.delta async for e in response if e.type == "response.output_text.delta"])
        assert text == "delayed answer"
    else:
        assert isinstance(response, ResponsesAPIResponse)
        assert response.output[0].content[0].text == "delayed answer"


def test_responses_mock_response_non_streaming():
    with patch("httpx.Client.send", side_effect=AssertionError("Unexpected HTTP request")):
        response = litellm.responses(
            model="openai/gpt-4.1",
            input="hi",
            mock_response="sync hello",
            stream=False,
            api_key="mock-key",
        )
    assert isinstance(response, ResponsesAPIResponse)
    assert response.output[0].content[0].text == "sync hello"


def test_responses_mock_response_streaming():
    with patch("httpx.Client.send", side_effect=AssertionError("Unexpected HTTP request")):
        stream_response = litellm.responses(
            model="openai/gpt-4.1",
            input="hi",
            mock_response="sync stream hello",
            stream=True,
            api_key="mock-key",
        )
    assert isinstance(stream_response, MockResponsesAPIStreamingIterator)

    collected_text = ""
    events = []
    for event in stream_response:
        events.append(event)
        if event.type == "response.output_text.delta":
            collected_text += event.delta

    assert collected_text == "sync stream hello"
    event_types = [event.type for event in events]
    assert "response.created" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.completed" in event_types


@pytest.mark.parametrize("stream", [False, True])
def test_responses_mock_delay_sync(stream: bool):
    delay = 0.2
    start_time = time.time()
    response = litellm.responses(
        model="openai/gpt-4.1",
        input="hi",
        mock_response="delayed sync",
        mock_delay=delay,
        stream=stream,
        api_key="mock-key",
    )
    elapsed = time.time() - start_time
    assert elapsed >= 0.18, f"Expected elapsed >= 0.18s, got {elapsed:.2f}s"

    if stream:
        assert isinstance(response, MockResponsesAPIStreamingIterator)
        text = "".join([e.delta for e in response if e.type == "response.output_text.delta"])
        assert text == "delayed sync"
    else:
        assert isinstance(response, ResponsesAPIResponse)
        assert response.output[0].content[0].text == "delayed sync"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_aresponses_empty_mock_response(stream: bool):
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        input="hi",
        mock_response="",
        stream=stream,
        api_key="mock-key",
    )
    if stream:
        assert isinstance(response, MockResponsesAPIStreamingIterator)
        deltas = [e.delta async for e in response if e.type == "response.output_text.delta"]
        assert "".join(deltas) == ""
    else:
        assert isinstance(response, ResponsesAPIResponse)
        assert response.output[0].content[0].text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
async def test_aresponses_assigns_a_synthetic_id_to_structured_mocks(stream: bool, as_dict: bool) -> None:
    from litellm.responses.utils import ResponsesAPIRequestUtils

    supplied = ResponsesAPIResponse(id="resp_fixture", created_at=1741476542, output=[])
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        input="hi",
        mock_response=supplied.model_dump() if as_dict else supplied,
        stream=stream,
        api_key="mock-key",
    )

    if stream:
        assert isinstance(response, MockResponsesAPIStreamingIterator)
        response_ids = {event.response.id async for event in response if event.type == "response.completed"}
        assert len(response_ids) == 1
        response_id = response_ids.pop()
    else:
        assert isinstance(response, ResponsesAPIResponse)
        response_id = response.id
    decoded = ResponsesAPIRequestUtils.decode_responses_api_response_id(response_id)
    assert decoded["response_id"].startswith("resp_mock_")
    assert supplied.id == "resp_fixture"
