# Gnani Timbre speech synthesis

Set `GNANI_API_KEY` on the machine running LiteLLM. Pass a voice from the selected model's catalog, such as Nalini for `gnani/timbre-v2.5` or Shivani for `gnani/timbre-v3.0`. Both models support automatic language selection and explicit language codes such as `hi-IN`, `en-IN` and `hi-en`. LiteLLM sends the input text unchanged, including inline audio tags for v3.0. [Voice catalog](https://docs.gnani.ai/api/TTS/available-voices) and [audio tags](https://docs.gnani.ai/api/TTS/tts-audio-tags).

```python
from pathlib import Path

from litellm import aspeech, speech

request = {
    "model": "gnani/timbre-v3.0",
    "input": "नमस्ते! आज हम गणित सीखेंगे।",
    "voice": "Shivani",
    "language": "hi-IN",
    "response_format": "wav",
    "audio_config": {"sample_rate": 24000},
}
Path("lesson.wav").write_bytes(speech(**request).content)


async def synthesize_lesson() -> bytes:
    return (await aspeech(**request)).content
```

For the proxy, register a deployment and use its alias with your LiteLLM key.

```yaml
model_list:
  - model_name: gnani-timbre-v3
    litellm_params:
      model: gnani/timbre-v3.0
      api_key: os.environ/GNANI_API_KEY
    model_info:
      health_check_voice: Shivani

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

```bash
curl --fail-with-body http://localhost:4000/v1/audio/speech \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gnani-timbre-v3","input":"नमस्ते! आज हम गणित सीखेंगे।","voice":"Shivani","language":"hi-IN","response_format":"wav","audio_config":{"sample_rate":24000}}' \
  -o lesson.wav
```

REST synthesis returns the complete audio. The default format is MP3. Use `response_format` for `mp3`, `wav`, `opus` (Ogg Opus) or `pcm` (raw 16-bit little-endian audio, 24 kHz by default). Supply Gnani's `audio_config` to set a sample rate, MP3 bitrate, encoding or container. For native container selection, omit `response_format`. For example, `{"container":"mulaw"}` and `{"container":"alaw"}` return raw 8 kHz G.711. With a native configuration, Gnani supplies defaults for omitted fields. The SDK and proxy label raw linear PCM as `audio/pcm`. OpenAI SDK clients can pass `language` and `audio_config` in `extra_body`.

Timbre v2.5 accepts speeds from 0.85 through 1.15. LiteLLM also accepts `slow`, `medium` and `fast` for that model. Timbre v3.0 supports numeric speeds of 0.8, 1.0 and 1.2. Voices and language codes are validated by Gnani. [REST reference](https://docs.gnani.ai/api/TTS/tts-inference).

To track estimated spend, set `model_info.input_cost_per_character` on the deployment to your contracted rate in USD. LiteLLM counts non-whitespace characters in the submitted text. Gnani returns audio without billing units, and the catalog leaves both model prices unset.

## Streaming speech

Set `stream=True` to receive binary audio as Gnani produces it. The SDK decodes each SSE event. Use `iter_bytes()` or `await result.aiter_bytes()` and close the response if playback stops early.

```python
from contextlib import aclosing, closing

from litellm import aspeech, speech

request = {
    "model": "gnani/timbre-v3.0",
    "voice": "Shivani",
    "input": "नमस्ते! आज हम गणित सीखेंगे।",
    "response_format": "pcm",
    "stream": True,
}
with closing(speech(**request)) as result, open("lesson.pcm", "wb") as output:
    for chunk in result.iter_bytes():
        output.write(chunk)


async def save_lesson():
    async with aclosing(await aspeech(**request)) as result:
        with open("lesson.pcm", "wb") as output:
            async for chunk in await result.aiter_bytes():
                output.write(chunk)
```

The proxy accepts the same `stream` flag at `/v1/audio/speech`. This example plays raw mono PCM as it arrives.

```bash
curl --fail --silent --show-error --no-buffer http://localhost:4000/v1/audio/speech \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gnani-timbre-v3","voice":"Shivani","input":"नमस्ते! आज हम गणित सीखेंगे।","response_format":"pcm","stream":true}' \
  | ffplay -nodisp -autoexit -f s16le -ar 24000 -ch_layout mono -i pipe:0
```

PCM, WAV, MP3 and G.711 support incremental delivery. Gnani sends Ogg Opus in a single event at the end. An upstream error or disconnect before completion interrupts the audio stream. [Gnani streaming reference](https://docs.gnani.ai/api/TTS/tts-sse).
