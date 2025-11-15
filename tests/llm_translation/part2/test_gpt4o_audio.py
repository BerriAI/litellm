import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse
from litellm.types.utils import StreamingChoices, ChatCompletionAudioResponse
import base64
import requests


def check_non_streaming_response(completion):
    assert completion.choices[0].message.audio is not None, "Audio response is missing"
    assert isinstance(
        completion.choices[0].message.audio, ChatCompletionAudioResponse
    ), "Invalid audio response type"
    assert len(completion.choices[0].message.audio.data) > 0, "Audio data is empty"


async def check_streaming_response(completion):
    _audio_bytes = None
    _audio_transcript = None
    _audio_id = None
    async for chunk in completion:
        print(chunk)
        _choice: StreamingChoices = chunk.choices[0]
        if _choice.delta.audio is not None:
            if _choice.delta.audio.get("data") is not None:
                _audio_bytes = _choice.delta.audio["data"]
            if _choice.delta.audio.get("transcript") is not None:
                _audio_transcript = _choice.delta.audio["transcript"]
            if _choice.delta.audio.get("id") is not None:
                _audio_id = _choice.delta.audio["id"]
    # Atleast one chunk should have set _audio_bytes, _audio_transcript, _audio_id
    assert _audio_bytes is not None
    assert _audio_transcript is not None
    assert _audio_id is not None


@pytest.mark.asyncio
# @pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.parametrize("stream", [True, False])
async def test_audio_output_from_model(stream):
    audio_format = "pcm16"
    if stream is False:
        audio_format = "wav"
    litellm.set_verbose = False
    try:
        completion = await litellm.acompletion(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "pcm16"},
            messages=[{"role": "user", "content": "response in 1 word - yes or no"}],
            stream=stream,
        )
    except litellm.Timeout as e:
        print(e)
        pytest.skip("Skipping test due to timeout")
    except Exception as e:
        if "openai-internal" in str(e):
            pytest.skip("Skipping test due to openai-internal error")

    if stream is True:
        await check_streaming_response(completion)

    else:
        print("response= ", completion)
        check_non_streaming_response(completion)
        wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
        with open("dog.wav", "wb") as f:
            f.write(wav_bytes)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("model", ["gpt-4o-audio-preview"]) # "gpt-4o-audio-preview", 
async def test_audio_input_to_model(stream, model):
    # Fetch the audio file and convert it to a base64 encoded string
    audio_format = "pcm16"
    if stream is False:
        audio_format = "wav"
    litellm._turn_on_debug()
    litellm.drop_params = True
    url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
    response = requests.get(url)
    response.raise_for_status()
    wav_data = response.content
    encoded_string = base64.b64encode(wav_data).decode("utf-8")
    try:
        completion = await litellm.acompletion(
            model=model,
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": audio_format},
            stream=stream,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this recording?"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded_string, "format": "wav"},
                        },
                    ],
                },
            ],
        )
    except litellm.Timeout as e:
        print(e)
        pytest.skip("Skipping test due to timeout")
    except Exception as e:
        if "openai-internal" in str(e):
            pytest.skip("Skipping test due to openai-internal error")
        raise e
    if stream is True:
        await check_streaming_response(completion)
    else:
        print("response= ", completion)

        check_non_streaming_response(completion)
        wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
        with open("dog.wav", "wb") as f:
            f.write(wav_bytes)
