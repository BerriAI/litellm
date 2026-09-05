from __future__ import annotations

from collections.abc import Generator, Mapping

import pytest

import litellm
from litellm.rust_bridge import responses
from litellm.rust_bridge.callbacks import OneShotCallbackHandle
from litellm.types.llms.openai import ResponsesAPIResponse


def _response_payload() -> dict[str, object]:
    return {
        "id": "resp_rust",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5",
        "output": [],
    }


class RecordingResponses:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def __call__(
        self,
        request: Mapping[str, object],
        callback_adapter: OneShotCallbackHandle | None,
    ) -> dict[str, object]:
        self.requests.append(dict(request))
        return _response_payload()


class RecordingAresponses:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def __call__(
        self,
        request: Mapping[str, object],
        callback_adapter: OneShotCallbackHandle | None,
    ) -> dict[str, object]:
        self.requests.append(dict(request))
        return _response_payload()


@pytest.fixture(autouse=True)
def reset_responses_endpoint() -> Generator[None]:
    responses.set_rust_responses(sync=None, asynchronous=None)
    yield
    responses.set_rust_responses(sync=None, asynchronous=None)


def test_top_level_responses_uses_shared_rust_handoff() -> None:
    binding = RecordingResponses()
    responses.set_rust_responses(sync=binding)

    result = litellm.responses(model="openai/gpt-5", input="hello", rust=True)

    assert isinstance(result, ResponsesAPIResponse)
    assert result.model == "gpt-5"
    assert binding.requests[0]["input"] == "hello"
    assert binding.requests[0]["model"] == "gpt-5"


def test_mock_response_finishes_before_native_dispatch() -> None:
    binding = RecordingResponses()
    responses.set_rust_responses(sync=binding)

    result = litellm.responses(
        model="openai/gpt-5",
        input="hello",
        mock_response="mocked",
        rust=True,
    )

    assert isinstance(result, ResponsesAPIResponse)
    assert binding.requests == []


@pytest.mark.asyncio
async def test_top_level_aresponses_uses_shared_rust_handoff_once() -> None:
    binding = RecordingAresponses()
    responses.set_rust_responses(asynchronous=binding)

    result = await litellm.aresponses(model="openai/gpt-5", input="hello", rust=True)

    assert isinstance(result, ResponsesAPIResponse)
    assert result.model == "gpt-5"
    assert len(binding.requests) == 1
    assert binding.requests[0]["input"] == "hello"
    assert binding.requests[0]["model"] == "gpt-5"
