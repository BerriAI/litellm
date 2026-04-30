# Soniox

LiteLLM integration for [Soniox](https://soniox.com)'s async speech-to-text API.

This provider exposes Soniox's async transcription pipeline (`stt-async-v4`) through LiteLLM's standard `litellm.transcription()` / `/audio/transcriptions` endpoint. The handler orchestrates Soniox's multi-step async flow (file upload → create transcription → poll until complete → fetch transcript → optional cleanup) behind a single OpenAI-compatible call.

## Quick start

```python
import litellm

# 1. Transcribe a local file (uploads to /v1/files, then runs the async pipeline)
with open("audio.wav", "rb") as f:
    response = litellm.transcription(
        model="soniox/stt-async-v4",
        file=f,
        api_key="sk-...",  # or set SONIOX_API_KEY
    )
print(response.text)

# 2. Transcribe a publicly hosted URL (no upload)
response = litellm.transcription(
    model="soniox/stt-async-v4",
    file=None,
    audio_url="https://example.com/clip.mp3",
)

# 3. Reuse a previously uploaded file_id
response = litellm.transcription(
    model="soniox/stt-async-v4",
    file=None,
    file_id="file_abc123",
)
```

The async variant works identically:

```python
response = await litellm.atranscription(
    model="soniox/stt-async-v4",
    file=None,
    audio_url="https://example.com/clip.mp3",
)
```

## Authentication

Either set `SONIOX_API_KEY` in the environment or pass `api_key="..."` to `litellm.transcription()`. The handler sends `Authorization: Bearer ${SONIOX_API_KEY}` on every request.

The API base defaults to `https://api.soniox.com`; override with `SONIOX_API_BASE` or by passing `api_base=...`.

## Supported parameters

### OpenAI-mapped

| OpenAI param | Soniox param | Notes |
|---|---|---|
| `language` | `language_hints` | The OpenAI `language` value is prepended to `language_hints` (deduped). |

### Soniox-native passthrough kwargs

Pass any of these as keyword arguments to `litellm.transcription()`:

| kwarg | Description |
|---|---|
| `audio_url` | URL to a publicly accessible audio file. Avoids the upload step. |
| `file_id` | Reuse an existing Soniox file. Avoids the upload step. |
| `language_hints` | List of language codes to guide recognition. |
| `language_hints_strict` | Bool. When true, restricts decoding to the hinted languages. |
| `enable_language_identification` | Bool. Adds per-token `language` field. |
| `enable_speaker_diarization` | Bool. Adds per-token `speaker` field. |
| `context` | Free-form context string (e.g. domain, vocabulary). |
| `translation` | Translation config object (see Soniox docs). |
| `client_reference_id` | Your own ID for traceability. |
| `webhook_url` | Soniox will POST to this URL when the transcription completes. |
| `webhook_auth_header_name` / `webhook_auth_header_value` | Optional auth header for webhook calls. |

Even when a webhook is configured, the handler still polls the transcription endpoint to deliver a synchronous result back to the caller.

### Handler-only kwargs

These tune handler behaviour and are **not** sent to Soniox:

| kwarg | Default | Description |
|---|---|---|
| `soniox_polling_interval` | `1.0` | Seconds between status polls. |
| `soniox_max_polling_attempts` | `1800` | Max polls before raising HTTP 504. (1800 × 1s ≈ 30 min.) |
| `soniox_cleanup` | `["file", "transcription"]` | Resources to delete after fetching the transcript. Use `[]` to skip cleanup, or `["file"]` / `["transcription"]` to delete only one. Cleanup runs even on the error path. |
| `filename` | `<derived>` | Override the filename sent in the `/v1/files` multipart upload. |

## Response shape

The returned `TranscriptionResponse` follows the OpenAI shape:

- `text` — When diarization or language identification is enabled, the text is rendered with SDK-style `Speaker N:` and `[lang]` tags so the structure survives a plain-text destination. Otherwise it is the API-provided plain transcript.
- `task` — Always `"transcribe"`.
- `duration` — Audio duration in seconds (when Soniox returns `audio_duration_ms`).
- `language` — Set when all tokens share a single language.

The full Soniox payload (transcription metadata + raw tokens with timing/speaker/language data) is preserved on `response._hidden_params["soniox_raw"]`:

```python
raw = response._hidden_params["soniox_raw"]
tokens = raw["transcript"]["tokens"]
duration_ms = raw["transcription"]["audio_duration_ms"]
```

## Diarization example

```python
response = litellm.transcription(
    model="soniox/stt-async-v4",
    file=None,
    audio_url="https://example.com/two_speakers.wav",
    enable_speaker_diarization=True,
    enable_language_identification=True,
)
print(response.text)
# Speaker 1:
# [en] Hi, how are you?
#
# Speaker 2:
# [en] I'm doing well, thanks.
```

## Proxy config

```yaml
model_list:
  - model_name: soniox-stt
    litellm_params:
      model: soniox/stt-async-v4
      api_key: os.environ/SONIOX_API_KEY
```

Then call the proxy's `/audio/transcriptions` endpoint as you would any OpenAI-compatible STT model.

## Limitations

- v1 covers async transcription only. Real-time WebSocket transcription is not yet supported.
- There is no provider-specific pass-through route (e.g. `/soniox/...`); use `audio_url` / `file_id` to access files you've already uploaded out-of-band.
- Pricing is intentionally not registered (no `input_cost_per_second`); LiteLLM cost tracking will treat Soniox calls as unknown cost. Consult [Soniox pricing](https://soniox.com/pricing) for the per-second rate and override locally if needed.
