from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # public callables have legacy partial annotations

import pytest

import litellm
from litellm.types.utils import TranscriptionResponse
from tests.test_litellm_rust.support.recording_server import ResponseSpec, recording_service

pytestmark = pytest.mark.requires_rust_extension

BEDROCK_RESPONSE: Final = {"output": {"message": {"content": [{"text": "hello from rust"}]}}}


class SyncTranscription(Protocol):
    def __call__(self, *, model: str, file: object, api_base: str, **kwargs: object) -> object: ...


class AsyncTranscription(Protocol):
    def __call__(self, *, model: str, file: object, api_base: str, **kwargs: object) -> Awaitable[object]: ...


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_public_transcription_uses_one_native_lifecycle_and_provider_request(asynchronous: bool) -> None:
    with recording_service() as service:
        service.enqueue(ResponseSpec(body=BEDROCK_RESPONSE))
        sync: Final = cast(SyncTranscription, litellm.transcription)  # pyright: ignore[reportUnknownMemberType]  # legacy signature
        async_call: Final = cast(AsyncTranscription, litellm.atranscription)  # pyright: ignore[reportUnknownMemberType]  # legacy signature
        kwargs: Final[dict[str, object]] = {
            "aws_access_key_id": "access-key",
            "aws_secret_access_key": "secret-key",
            "aws_region_name": "us-east-1",
        }
        value: Final = (
            await async_call(
                model="bedrock/mistral.voxtral-mini-3b-2507",
                file=("audio.wav", b"audio", "audio/wav"),
                api_base=service.base_url,
                **kwargs,
            )
            if asynchronous
            else sync(
                model="bedrock/mistral.voxtral-mini-3b-2507",
                file=("audio.wav", b"audio", "audio/wav"),
                api_base=service.base_url,
                **kwargs,
            )
        )

    assert isinstance(value, TranscriptionResponse)
    assert value.text == "hello from rust"
    assert len(service.requests) == 1
    assert service.requests[0].path == "/model/mistral.voxtral-mini-3b-2507/converse"
    assert service.requests[0].headers["authorization"].startswith("AWS4-HMAC-SHA256")
    body: Final = cast(dict[str, object], service.requests[0].body)
    assert "YXVkaW8=" in str(body)
