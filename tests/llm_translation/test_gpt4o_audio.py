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
import base64
import requests


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_audio_output_from_model():
    litellm.set_verbose = True
    completion = await litellm.acompletion(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[{"role": "user", "content": "response in 1 word - yes or no"}],
    )

    print("response= ", completion)

    print(completion.choices[0])

    assert completion.choices[0].message.audio is not None
    assert isinstance(
        completion.choices[0].message.audio, litellm.types.utils.ChatCompletionAudio
    )
    assert len(completion.choices[0].message.audio.data) > 0

    wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
    with open("dog.wav", "wb") as f:
        f.write(wav_bytes)


@pytest.mark.asyncio
async def test_audio_input_to_model():
    # Fetch the audio file and convert it to a base64 encoded string
    url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
    response = requests.get(url)
    response.raise_for_status()
    wav_data = response.content
    encoded_string = base64.b64encode(wav_data).decode("utf-8")

    completion = await litellm.acompletion(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
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

    print(completion.choices[0].message)
