# Volcengine

## Overview

| Property | Details |
|-------|-------|
| Provider Route on LiteLLM | `volcengine/` |
| Chat API Key | `VOLCENGINE_API_KEY` |
| Speech API Key | `VOLCENGINE_SPEECH_KEY` |
| Supported Operations | `/chat/completions`, `/responses`, `/audio/transcriptions`, `/audio/speech`, `/realtime` |

Volcengine chat and responses models use Ark's OpenAI-compatible APIs. Volcengine speech models use the OpenSpeech WebSocket APIs behind LiteLLM's OpenAI-compatible audio endpoints.

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["VOLCENGINE_API_KEY"] = ""     # Ark API key for chat/responses
os.environ["VOLCENGINE_SPEECH_KEY"] = ""  # Speech API key for STT/TTS/realtime
```

## Audio Transcriptions

Volcengine speech recognition is exposed through LiteLLM's OpenAI-compatible `/v1/audio/transcriptions` endpoint.

```python showLineNumbers title="Volcengine STT"
from pathlib import Path
from litellm import transcription

response = transcription(
    model="volcengine/volc.seedasr.sauc.duration",
    file=Path("speech.wav").read_bytes(),
    response_format="json",
)

print(response.text)
```

Proxy configuration:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: volc.seedasr.sauc.duration
    litellm_params:
      model: volcengine/volc.seedasr.sauc.duration
      api_key: os.environ/VOLCENGINE_SPEECH_KEY
      api_base: wss://openspeech.bytedance.com/api/v3/sauc/bigmodel
```

Supported STT model names:

- `volc.bigasr.sauc.duration`
- `volc.bigasr.sauc.concurrent`
- `volc.seedasr.sauc.duration`
- `volc.seedasr.sauc.concurrent`

Supported OpenAI parameters:

- `language`
- `response_format=json`

## Text To Speech

Volcengine text to speech is exposed through LiteLLM's OpenAI-compatible `/v1/audio/speech` endpoint.

```python showLineNumbers title="Volcengine TTS"
from litellm import speech

response = speech(
    model="volcengine/seed-tts-2.0",
    input="Hello from LiteLLM",
    voice="zh_female_vv_uranus_bigtts",
    response_format="pcm",
)

with open("speech.pcm", "wb") as f:
    f.write(response.content)
```

Proxy configuration:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: seed-tts-2.0
    litellm_params:
      model: volcengine/seed-tts-2.0
      api_key: os.environ/VOLCENGINE_SPEECH_KEY
      api_base: wss://openspeech.bytedance.com/api/v3/tts/bidirection
      resource_id: seed-tts-2.0
    model_info:
      mode: audio_speech
      health_check_voice: zh_female_vv_uranus_bigtts
```

Supported OpenAI parameters:

- `voice`
- `response_format=pcm`
- `response_format=wav`

OpenAI voice names such as `alloy` are accepted and mapped to the default Volcengine voice. Pass a Volcengine voice id to select a specific speaker.

Supported TTS model names:

- `seed-tts-2.0`
- `seed-tts-2.0-standard`
- `seed-tts-2.0-expressive`
- `seed-tts-1.0`
- `seed-tts-1.0-concurr`
- `seed-icl-2.0`
- `seed-icl-1.0`
- `seed-icl-1.0-concurr`

## Realtime Dialogue

Volcengine realtime dialogue is exposed through LiteLLM's OpenAI-compatible `/v1/realtime` WebSocket endpoint.

Proxy configuration with API Key auth:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: volc.speech.dialog
    litellm_params:
      model: volcengine/volc.speech.dialog
      api_key: os.environ/VOLCENGINE_SPEECH_KEY
      api_base: wss://openspeech.bytedance.com/api/v3/realtime/dialogue
    model_info:
      mode: realtime
```

If realtime dialogue uses a separate speech API key in your project, set `VOLCENGINE_REALTIME_API_KEY` and reference that instead. LiteLLM still accepts the older App-ID / Access-Key headers for legacy projects, but new console projects should use API Key auth.

LiteLLM automatically sends the additional Volcengine WebSocket headers required by the realtime API:

- `X-Api-Resource-Id: volc.speech.dialog`
- `X-Api-App-Key: PlgvMymc7f3tQnJ6`
- `X-Api-Connect-Id: <uuid>`

Supported realtime model names:

- `volc.speech.dialog`

Supported client events:

- `session.update`
- `input_audio_buffer.append`

LiteLLM sends Volcengine `StartConnection` on WebSocket connect, maps the first `session.update` to `StartSession`, forwards `input_audio_buffer.append` PCM bytes as realtime audio, normalizes Volcengine output audio to OpenAI-compatible `pcm16`, and emits `response.output_audio.delta` events.

## Notes

- `VOLCENGINE_SPEECH_KEY` is the Speech API key used with the `X-Api-Key` header. It is separate from the Ark API key used for chat and responses.
- STT defaults to `wss://openspeech.bytedance.com/api/v3/sauc/bigmodel` and supports the official SAUC resource ids listed above.
- TTS defaults to `wss://openspeech.bytedance.com/api/v3/tts/bidirection` and `seed-tts-2.0`.
- Realtime Dialogue defaults to `wss://openspeech.bytedance.com/api/v3/realtime/dialogue` and the official resource id `volc.speech.dialog`.
- Realtime Dialogue uses `VOLCENGINE_SPEECH_KEY` as `X-Api-Key` for newer speech console projects. Use `VOLCENGINE_REALTIME_API_KEY` only when realtime has a separate key.
- If your Volcengine console shows a different realtime `Resource-Id` or `App-Key`, pass it through `extra_headers` / proxy headers; LiteLLM preserves explicit `X-Api-*` values.
