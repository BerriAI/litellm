import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_PATH
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import TranscriptionResponse

MODEL: Final = "qwen-audio-3.0-asr-flash"
GATEWAY: Final = "https://gateway.example/plan"


def _transport(expected_messages: list[Mapping[str, object]], expected_parameters: Mapping[str, object]):
    def transport(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{GATEWAY}/{IMAGE_PATH}"
        assert request.headers["authorization"] == "Bearer test-token-plan-key"
        assert json.loads(request.content) == {
            "model": MODEL,
            "input": {"messages": expected_messages},
            "parameters": expected_parameters,
        }
        return httpx.Response(
            200,
            json={
                "output": {"text": "First sentence. Second sentence.", "sentence": {"text": "Second sentence."}},
                "usage": {"duration": 4},
            },
        )

    return httpx.MockTransport(transport)


AUDIO_MESSAGE: Final = {
    "role": "user",
    "content": [{"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,YXVkaW8="}}],
}


def _assert_full_transcript(response: TranscriptionResponse) -> None:
    assert response.text == "First sentence. Second sentence."
    assert response.usage is not None
    assert response.usage.model_dump() == {"type": "duration", "seconds": 4.0}


def test_transcription_sends_inline_audio_with_prompt_and_language_hint() -> None:
    transport: Final = _transport(
        [{"role": "user", "content": [{"type": "input_text", "text": "Qwen and Alibaba"}]}, AUDIO_MESSAGE],
        {"format": "wav", "language_hints": ["en"]},
    )
    with httpx.Client(transport=transport) as http_client:
        response: Final = litellm.transcription(
            model=f"alibaba_token_plan/{MODEL}",
            file=("recording.wav", b"audio"),
            language="en",
            prompt="Qwen and Alibaba",
            response_format="json",
            api_key="test-token-plan-key",
            api_base=f"{GATEWAY}/compatible-mode/v1",
            client=HTTPHandler(client=http_client),
        )
    assert isinstance(response, TranscriptionResponse)
    _assert_full_transcript(response)


@pytest.mark.asyncio
async def test_async_transcription_without_options_takes_the_format_from_the_file_name() -> None:
    client: Final = AsyncHTTPHandler(transport=_transport([AUDIO_MESSAGE], {"format": "wav"}))
    async with client.client:
        response: Final = await litellm.atranscription(
            model=f"alibaba_token_plan/{MODEL}",
            file=("recording.wav", b"audio"),
            api_key="test-token-plan-key",
            api_base=f"{GATEWAY}/compatible-mode/v1",
            client=client,
        )
    assert isinstance(response, TranscriptionResponse)
    _assert_full_transcript(response)


def test_transcription_errors_keep_the_provider_status() -> None:
    transport: Final = httpx.MockTransport(
        lambda _request: httpx.Response(400, json={"code": "InvalidParameter", "message": "Audio too long"})
    )
    with (
        httpx.Client(transport=transport) as http_client,
        pytest.raises(BaseLLMException, match="Audio too long") as error,
    ):
        litellm.transcription(
            model=f"alibaba_token_plan/{MODEL}",
            file=("recording.wav", b"audio"),
            api_key="test-token-plan-key",
            api_base=f"{GATEWAY}/compatible-mode/v1",
            client=HTTPHandler(client=http_client),
        )
    assert error.value.status_code == 400


@pytest.mark.parametrize("response_format", ["srt", "vtt", "verbose_json"])
def test_unsupported_response_format_is_rejected_unless_dropped(response_format: str) -> None:
    with httpx.Client(transport=_transport([AUDIO_MESSAGE], {"format": "wav"})) as http_client:
        with pytest.raises(litellm.UnsupportedParamsError, match=response_format):
            litellm.transcription(
                model=f"alibaba_token_plan/{MODEL}",
                file=("recording.wav", b"audio"),
                response_format=response_format,
                api_key="test-token-plan-key",
                api_base=f"{GATEWAY}/compatible-mode/v1",
                client=HTTPHandler(client=http_client),
            )
        response: Final = litellm.transcription(
            model=f"alibaba_token_plan/{MODEL}",
            file=("recording.wav", b"audio"),
            response_format=response_format,
            drop_params=True,
            api_key="test-token-plan-key",
            api_base=f"{GATEWAY}/compatible-mode/v1",
            client=HTTPHandler(client=http_client),
        )
    assert isinstance(response, TranscriptionResponse)
    _assert_full_transcript(response)
