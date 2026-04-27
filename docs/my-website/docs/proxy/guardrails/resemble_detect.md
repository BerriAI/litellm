import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Resemble AI Detect

Use [Resemble AI Detect](https://www.resemble.ai/detect) to scan audio, video, and image URLs referenced in LLM requests for deepfake / synthetic media. The guardrail blocks the request when Resemble labels the media as `fake` or when the aggregated score exceeds the configured threshold.

Resemble Detect works across three modalities:

- **Audio** — detects cloned voices and TTS-generated speech (ElevenLabs, Resemble AI, OpenAI, PlayHT, etc.)
- **Image** — detects facial deepfakes and generative image manipulation
- **Video** — frame-level detection with a single aggregated verdict

It runs asynchronously: LiteLLM submits the media URL, polls for the verdict, and either passes or blocks the LLM call.

## Quick Start

### 1. Get an API key

Create an API token at [app.resemble.ai/account/api](https://app.resemble.ai/account/api).

### 2. Define the guardrail in your `config.yaml`

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "resemble-deepfake-detect"
    litellm_params:
      guardrail: resemble
      mode: "pre_call"
      api_key: os.environ/RESEMBLE_API_KEY
      # Optional: override the API base
      # api_base: https://app.resemble.ai/api/v2
      # Block media with aggregated_score >= threshold (default 0.5)
      resemble_threshold: 0.5
      # Optional: force a modality (audio | video | image)
      # resemble_media_type: audio
      # Identify the TTS vendor that produced flagged audio
      resemble_audio_source_tracing: true
      # Do not persist media on Resemble after the scan
      resemble_zero_retention_mode: true
      # Block the request if Resemble is unreachable (default: fail open)
      resemble_fail_closed: false
```

#### Supported values for `mode`

- `pre_call` — runs **before** the LLM call. Blocks the request if the media is flagged.
- `during_call` — runs **in parallel** with the LLM call for lower latency. Still blocks on flagged media.

### 3. Start LiteLLM

```shell
litellm --config config.yaml --detailed_debug
```

### 4. Send a multimodal request

The guardrail looks for media URLs in (in order):

1. OpenAI-style multimodal content parts (`image_url`, `input_audio`)
2. Anthropic-style `source.url` parts (image, document)
3. Any `https://…` URL in message text that ends in a known audio/video/image extension
4. `metadata.mediaUrl` (key configurable via `resemble_metadata_key`)

<Tabs>
<TabItem label="OpenAI image_url part" value="openai-image">

```shell
curl -i http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Is this photo real?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/face.jpg"}}
        ]
      }
    ],
    "guardrails": ["resemble-deepfake-detect"]
  }'
```

</TabItem>

<TabItem label="Plain text with audio URL" value="text-audio">

```shell
curl -i http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Transcribe https://example.com/clip.mp3 please"}
    ],
    "guardrails": ["resemble-deepfake-detect"]
  }'
```

</TabItem>

<TabItem label="metadata.mediaUrl" value="metadata">

```shell
curl -i http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "Analyze the uploaded clip"}
    ],
    "metadata": {"mediaUrl": "https://example.com/clip.wav"},
    "guardrails": ["resemble-deepfake-detect"]
  }'
```

</TabItem>
</Tabs>

### 5. Example blocked response

```json
{
  "error": {
    "message": {
      "error": "Resemble Detect flagged media as synthetic",
      "resemble": {
        "uuid": "a1b2c3d4-5e6f-7890-abcd-ef0123456789",
        "media_url": "https://example.com/clip.mp3",
        "media_type": "audio",
        "label": "fake",
        "score": 0.95,
        "threshold": 0.5,
        "reason": "Resemble Detect flagged media as fake (score=0.95, threshold=0.5)",
        "audio_source_tracing": {
          "label": "elevenlabs",
          "error_message": null
        }
      }
    }
  }
}
```

## Configuration reference

| Parameter                        | Type       | Default                             | Description                                                                                                    |
| -------------------------------- | ---------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `api_key`                        | string     | `RESEMBLE_API_KEY` env var          | Resemble AI API token.                                                                                         |
| `api_base`                       | string     | `https://app.resemble.ai/api/v2`    | Override the Resemble API base URL (useful for sovereign deployments).                                         |
| `resemble_threshold`             | number     | `0.5`                               | Aggregated score above which media is treated as fake (0.0–1.0).                                               |
| `resemble_media_type`            | enum       | auto                                | Force `audio`, `video`, or `image`. Omit for auto-detect from extension / content type.                        |
| `resemble_audio_source_tracing`  | bool       | `false`                             | Return which TTS vendor generated flagged audio (ElevenLabs, Resemble AI, OpenAI, etc.).                        |
| `resemble_use_reverse_search`    | bool       | `false`                             | (Image only) search the web for matching images to improve accuracy.                                            |
| `resemble_zero_retention_mode`   | bool       | `false`                             | Automatically delete submitted media after detection. URLs are redacted and filenames are tokenized.            |
| `resemble_metadata_key`          | string     | `"mediaUrl"`                        | Key under request `metadata` to read the media URL from when it is not present in the message content.         |
| `resemble_poll_interval_seconds` | number     | `2.0`                               | How often to poll Resemble for the detection result.                                                            |
| `resemble_poll_timeout_seconds`  | number     | `60.0`                              | Maximum total time to wait for a detection result before failing.                                               |
| `resemble_fail_closed`           | bool       | `false`                             | If `true`, Resemble API errors **block** the request. If `false` (default), errors are logged and ignored.      |

## Zero Retention Mode

For workflows where you cannot retain media on Resemble's infrastructure (e.g. HIPAA/financial compliance), set `resemble_zero_retention_mode: true`. Resemble will tokenize filenames, redact submitted URLs from logs, and delete the media artifact after the scan completes. The verdict is still returned synchronously.

## Audio source tracing

When `resemble_audio_source_tracing: true`, the blocked-response `resemble.audio_source_tracing` object contains the source model that produced the cloned audio:

```json
{
  "audio_source_tracing": {
    "label": "elevenlabs",
    "error_message": null
  }
}
```

Possible labels include `elevenlabs`, `resemble_ai`, `openai`, `playht`, `azure_neural`, `google_tts`, and others. This is useful for incident triage and attributing cloned-voice abuse back to the generating vendor.
