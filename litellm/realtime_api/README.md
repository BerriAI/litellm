Abstraction / Routing logic for OpenAI's `/v1/realtime` endpoints.

Supported endpoints:
- WebSocket: `/v1/realtime` (with `intent=transcription` for transcription-only sessions)
- HTTP: `/v1/realtime/client_secrets`, `/v1/realtime/transcription_sessions`

Supported providers: OpenAI, Azure OpenAI, Bedrock, Vertex AI, xAI, Gemini, Meta, Soniox, Gnani (WebSocket transcription).

For user-facing documentation and usage examples, see the litellm-docs repo.

### Gnani Prisma WebSocket transcription

Configure a gateway deployment named `gnani-prisma` with `model: gnani/prisma-v3.0` or `model: gnani/prisma-v2.5` and `api_key: os.environ/GNANI_API_KEY`. Connect to the deployment using the OpenAI Python client. This example reads a 24 kHz mono PCM16 WAV file and prints its first final transcript.

```python
import asyncio
import base64
import os
import wave

from openai import AsyncOpenAI


async def transcribe(path):
    async with AsyncOpenAI(base_url="http://localhost:4000/v1", api_key=os.environ["LITELLM_MASTER_KEY"]) as client:
        async with client.realtime.connect(model="gnani-prisma", extra_query={"intent": "transcription"}) as ws:
            await ws.session.update(session={
                "type": "transcription",
                "audio": {"input": {"transcription": {"model": "gnani-prisma", "language": "en"}}},
            })
            with wave.open(path, "rb") as audio:
                assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (24000, 1, 2)
                while chunk := audio.readframes(768):
                    await ws.input_audio_buffer.append(audio=base64.b64encode(chunk).decode())
            await ws.input_audio_buffer.commit()
            async for event in ws:
                if event.type == "error":
                    raise RuntimeError(event.error.message)
                if event.type == "conversation.item.input_audio_transcription.completed":
                    print(event.transcript)
                    return


asyncio.run(transcribe("speech.wav"))
```

Gnani detects speech boundaries on the server. A commit finishes the current utterance immediately. Each final transcript has the same item ID as its `input_audio_buffer.committed` event. Use one language per session update, or `indic-auto` with Prisma v3.0. Set provider-specific connection options such as `format` in deployment `extra_headers`. If the connection drops, open a fresh session and continue from the last completed utterance.
