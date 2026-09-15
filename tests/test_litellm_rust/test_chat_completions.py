from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # public callables have legacy partial annotations

import pytest

import litellm
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

pytestmark = pytest.mark.requires_rust_extension

ANTHROPIC_RESPONSE: Final = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "hello from rust lifecycle"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 4},
}


class SyncCompletion(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: list[object],
        max_tokens: int,
        api_key: str,
        api_base: str,
    ) -> object: ...


class AsyncCompletion(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: list[object],
        max_tokens: int,
        api_key: str,
        api_base: str,
    ) -> Awaitable[object]: ...


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_public_chat_uses_one_native_lifecycle_and_one_provider_request(asynchronous: bool) -> None:
    with recording_service() as service:
        service.enqueue(ResponseSpec(body=ANTHROPIC_RESPONSE))
        sync: Final = cast(SyncCompletion, litellm.completion)  # pyright: ignore[reportUnknownMemberType]  # legacy signature
        async_call: Final = cast(AsyncCompletion, litellm.acompletion)  # pyright: ignore[reportUnknownMemberType]  # legacy signature
        messages: Final[list[object]] = [{"role": "user", "content": "hi"}]
        response_value: Final = (
            await async_call(
                model="anthropic/claude-sonnet-4-5",
                messages=messages,
                max_tokens=16,
                api_key="test-key",
                api_base=service.base_url,
            )
            if asynchronous
            else sync(
                model="anthropic/claude-sonnet-4-5",
                messages=messages,
                max_tokens=16,
                api_key="test-key",
                api_base=service.base_url,
            )
        )

    assert isinstance(response_value, ModelResponse)
    response: Final = response_value
    assert response.choices[0].message.content == "hello from rust lifecycle"
    assert response._hidden_params["additional_headers"] == {  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # public response metadata
        "x-litellm-rust": "true"
    }
    assert len(service.requests) == 1
    assert service.requests[0].path == "/v1/messages"
    assert service.requests[0].body == {
        "max_tokens": 16,
        "messages": [{"content": [{"text": "hi", "type": "text"}], "role": "user"}],
        "model": "claude-sonnet-4-5",
    }
