import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# WatsonX Audio Transcription

Use WatsonX's Whisper models for audio transcription through LiteLLM.

## Quick Start

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python showLineNumbers title="transcription.py"
import litellm

response = litellm.transcription(
    model="watsonx/whisper-large-v3-turbo",
    file=open("audio.mp3", "rb"),
    api_base="https://us-south.ml.cloud.ibm.com",
    api_key="your-api-key",
    project_id="your-project-id"
)
print(response.text)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Add to your `config.yaml`:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: whisper-large-v3-turbo
    litellm_params:
      model: watsonx/whisper-large-v3-turbo
      api_key: os.environ/WATSONX_APIKEY
      api_base: os.environ/WATSONX_URL
      project_id: os.environ/WATSONX_PROJECT_ID
```

2. Make a request:

```bash
curl http://localhost:4000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-1234" \
  -F file="@audio.mp3" \
  -F model="whisper-large-v3-turbo"
```

</TabItem>
</Tabs>

## Supported Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model ID (e.g., `watsonx/whisper-large-v3-turbo`) |
| `file` | file | Audio file to transcribe |
| `language` | string | Language code (e.g., `en`) |
| `prompt` | string | Optional prompt to guide transcription |
| `temperature` | float | Sampling temperature (0-1) |
| `response_format` | string | `json`, `text`, `srt`, `verbose_json`, `vtt` |

