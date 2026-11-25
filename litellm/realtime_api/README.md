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

Gnani detects speech boundaries on the server. A commit finishes the current utterance immediately, and a commit after Gnani already finished the utterance on its own gets no reply. Each final transcript has the same item ID as its `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped` and `input_audio_buffer.committed` events, and its `usage.seconds` is the length of that utterance, with silence between utterances billed with the next transcript or when the session closes. The proxy forwards audio to Gnani at real-time pace, so send each append as audio arrives, at most four seconds of audio per append. The cost map has no Gnani price, so set `input_cost_per_second` on the deployment to track spend and enforce budgets. Use one language per session update, or `indic-auto` with Prisma v3.0. Prisma v2.5 needs one explicit language before the first audio append. Gnani supports only server_vad with its own settings, so a session update that sets server_vad tuning (`threshold`, `prefix_padding_ms`, `silence_duration_ms`), another turn detection, `turn_detection: null` or a transcription `prompt` fails unless `drop_params` is on, which drops them and keeps server_vad. A rejected client event returns an `error` event and the session stays open. Set provider-specific connection options such as `format` in deployment `extra_headers`. Gnani closes a session after 15 minutes, or after 60 seconds without detected speech, which `x-idle-timeout-sec` (5 to 900) in deployment `extra_headers` changes. If the connection drops, open a fresh session and continue from the last completed utterance.
