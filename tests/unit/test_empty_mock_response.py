from unittest.mock import patch

import pytest

import litellm


@pytest.mark.parametrize("mock_response", ["", "Hello"])
@pytest.mark.parametrize("stream", [False, True])
def test_completion_mock_response(mock_response: str, stream: bool) -> None:
    with patch("httpx.Client.send", side_effect=AssertionError("Unexpected HTTP request")):
        response = litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=mock_response,
            stream=stream,
        )
        if stream:
            content = "".join(chunk.choices[0].delta.content or "" for chunk in response)
        else:
            content = response.choices[0].message.content
    assert content == mock_response


@pytest.mark.asyncio
@pytest.mark.parametrize("mock_response", ["", "Hello"])
@pytest.mark.parametrize("stream", [False, True])
async def test_acompletion_mock_response(mock_response: str, stream: bool) -> None:
    with (
        patch(
            "httpx.AsyncClient.send",
            side_effect=AssertionError("Unexpected HTTP request"),
        ),
        patch("httpx.Client.send", side_effect=AssertionError("Unexpected HTTP request")),
    ):
        response = await litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            mock_response=mock_response,
            stream=stream,
        )
        if stream:
            content = "".join([chunk.choices[0].delta.content or "" async for chunk in response])
        else:
            content = response.choices[0].message.content
    assert content == mock_response
