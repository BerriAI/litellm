from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # public callables have legacy partial annotations

import pytest

import litellm
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

pytestmark = pytest.mark.requires_rust_extension

ANTHROPIC_RESPONSE: Final = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "hello from rust messages"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 4},
}


class SyncMessages(Protocol):
    def __call__(
        self,
        *,
        max_tokens: int,
        messages: list[object],
        model: str,
        api_key: str,
        api_base: str,
    ) -> object: ...


class AsyncMessages(Protocol):
    def __call__(
        self,
        *,
        max_tokens: int,
        messages: list[object],
        model: str,
        api_key: str,
        api_base: str,
    ) -> Awaitable[object]: ...


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_public_messages_uses_one_native_lifecycle_and_provider_request(asynchronous: bool) -> None:
    with recording_service() as service:
        service.enqueue(ResponseSpec(body=ANTHROPIC_RESPONSE))
        sync: Final = cast(SyncMessages, litellm.anthropic.create)  # pyright: ignore[reportUnknownMemberType]  # legacy signature
        async_call: Final = cast(AsyncMessages, litellm.anthropic.acreate)  # pyright: ignore[reportUnknownMemberType]  # legacy signature
        messages: Final[list[object]] = [{"role": "user", "content": "hi"}]
        value: Final = (
            await async_call(
                max_tokens=16,
                messages=messages,
                model="anthropic/claude-sonnet-4-5",
                api_key="test-key",
                api_base=service.base_url,
            )
            if asynchronous
            else sync(
                max_tokens=16,
                messages=messages,
                model="anthropic/claude-sonnet-4-5",
                api_key="test-key",
                api_base=service.base_url,
            )
        )

    response: Final = cast(dict[str, object], value)
    content: Final = cast(list[dict[str, object]], response["content"])
    hidden: Final = cast(dict[str, object], response["_hidden_params"])
    assert content[0]["text"] == "hello from rust messages"
    assert hidden["additional_headers"] == {"x-litellm-rust": "true"}
    assert len(service.requests) == 1
    assert service.requests[0].path == "/v1/messages"
    assert service.requests[0].body == {
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "model": "claude-sonnet-4-5",
        "stream": False,
    }
