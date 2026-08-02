# OpenAI Realtime translation and transcription

LiteLLM supports the dedicated OpenAI Realtime translation family, the current reasoning Realtime models, and the latest file and live transcription models.

| Model | LiteLLM API | Primary use |
| --- | --- | --- |
| `gpt-realtime-2` | Realtime WebSocket/WebRTC | Reasoning speech-to-speech |
| `gpt-realtime-2.1` | Realtime WebSocket/WebRTC | Updated reasoning speech-to-speech |
| `gpt-realtime-2.1-mini` | Realtime WebSocket/WebRTC | Lower-cost reasoning speech-to-speech |
| `gpt-realtime-translate` | Realtime Translation WebSocket/WebRTC | Continuous speech translation |
| `gpt-realtime-whisper` | Realtime transcription | Low-latency source transcription |
| `gpt-transcribe` | `/v1/audio/transcriptions` | Completed audio files, optionally streamed |
| `gpt-live-transcribe` | Realtime transcription | Live transcription with tunable delay |

Use the `azure/` prefix for Azure OpenAI deployments, for example `azure/gpt-realtime-2.1` or `azure/gpt-transcribe`. LiteLLM selects Azure's GA Realtime protocol for these models and the Azure OpenAI v1 API for `gpt-transcribe`. API keys, Entra bearer tokens, and configured Azure token providers are supported.

## Translation over WebSocket

Connect a WebSocket client to the LiteLLM proxy and send the normal translation session and audio events:

```text
wss://LITELLM_PROXY/v1/realtime/translations?model=gpt-realtime-translate
```

Configure the target language after connecting:

```json
{
  "type": "session.update",
  "session": {
    "audio": {
      "input": {
        "transcription": {"model": "gpt-realtime-whisper"},
        "noise_reduction": {"type": "near_field"}
      },
      "output": {"language": "es"}
    }
  }
}
```

Append continuous base64-encoded 24 kHz PCM16 audio with `session.input_audio_buffer.append`. Translation sessions emit `session.output_audio.delta` and `session.output_transcript.delta`; when input transcription is enabled, they also emit `session.input_transcript.delta`.

## Translation over WebRTC

Create a short-lived client secret on the server:

```python
import litellm

response = await litellm.acreate_realtime_translation_client_secret(
    model="gpt-realtime-translate",
    session={
        "audio": {
            "input": {
                "transcription": {"model": "gpt-realtime-whisper"},
                "noise_reduction": {"type": "near_field"},
            },
            "output": {"language": "es"},
        }
    },
)
client_secret = response.json()["value"]
```

The equivalent proxy endpoint is:

```text
POST /v1/realtime/translations/client_secrets
```

Send the browser's SDP offer as a raw `application/sdp` request using the encrypted client secret returned by the LiteLLM proxy:

```text
POST /v1/realtime/translations/calls
Authorization: Bearer LITELLM_ENCRYPTED_CLIENT_SECRET
Content-Type: application/sdp

v=0
...
```

The unversioned `/realtime/translations/...` and Azure-compatible `/openai/v1/realtime/translations/...` aliases are also available. Client secrets are bound to their endpoint family: a standard Realtime secret cannot be replayed against Translation, and a Translation secret cannot be replayed against the standard Realtime call endpoint.

## File transcription with `gpt-transcribe`

The non-streaming response includes the transcript, detected languages when returned by the model, and usage:

```python
import litellm

with open("meeting.wav", "rb") as audio:
    response = litellm.transcription(
        model="gpt-transcribe",
        file=audio,
        languages=["en", "fr"],
        keywords=["LiteLLM", "Microsoft Foundry"],
    )

print(response.text)
print(response.languages)
```

Set `stream=True` to receive the OpenAI SDK's typed `transcript.text.delta` and `transcript.text.done` events:

```python
import litellm

with open("meeting.wav", "rb") as audio:
    stream = litellm.transcription(
        model="gpt-transcribe",
        file=audio,
        languages=["en", "fr"],
        keywords=["LiteLLM"],
        stream=True,
    )
    for event in stream:
        if event.type == "transcript.text.delta":
            print(event.delta, end="", flush=True)
```

`language` and `languages` are mutually exclusive. `gpt-transcribe` supports JSON responses; Whisper-only formats such as `srt`, `vtt`, and `verbose_json` are rejected before the provider call. `gpt-live-transcribe` accepts continuous audio through Realtime and is intentionally rejected by the file transcription API.

## Live transcription

Open a transcription-intent Realtime session and configure `gpt-live-transcribe` in the nested audio transcription object:

```text
wss://LITELLM_PROXY/v1/realtime?intent=transcription
```

```json
{
  "type": "session.update",
  "session": {
    "type": "transcription",
    "audio": {
      "input": {
        "transcription": {
          "model": "gpt-live-transcribe",
          "delay": "low",
          "languages": ["en", "fr"],
          "keywords": ["LiteLLM"]
        }
      }
    }
  }
}
```

The same nested configuration can be supplied when creating a standard Realtime client secret. LiteLLM authorizes and routes against the nested transcription model rather than a caller-controlled top-level alias.
