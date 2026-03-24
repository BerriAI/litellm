# AssemblyAI

Pass-through endpoints for AssemblyAI - call AssemblyAI endpoints, in native format (no translation).

| Feature | Supported | Notes |
|-------|-------|-------|
| Cost Tracking | ✅ | works across all integrations |
| Logging | ✅ | works across all integrations |


Supports **ALL** AssemblyAI Endpoints

[**See All AssemblyAI Endpoints**](https://www.assemblyai.com/docs/api-reference)


## Supported Routes

| AssemblyAI Service | LiteLLM Route | AssemblyAI Base URL |
|-------------------|---------------|---------------------|
| Speech-to-Text (US) | `/assemblyai/*` | `api.assemblyai.com` |
| Speech-to-Text (EU) | `/eu.assemblyai/*` | `eu.api.assemblyai.com` |

## Quick Start

Let's call the AssemblyAI [`/v2/transcripts` endpoint](https://www.assemblyai.com/docs/api-reference/transcripts)

1. Add AssemblyAI API Key to your environment

```bash
export ASSEMBLYAI_API_KEY=""
```

2. Start LiteLLM Proxy

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

3. Test it!

Let's call the AssemblyAI [`/v2/transcripts` endpoint](https://www.assemblyai.com/docs/api-reference/transcripts). Includes commented-out [Speech Understanding](https://www.assemblyai.com/docs/speech-understanding) features you can toggle on.

```python
import assemblyai as aai

aai.settings.base_url = "http://0.0.0.0:4000/assemblyai" # <your-proxy-base-url>/assemblyai
aai.settings.api_key = "Bearer sk-1234" # Bearer <your-virtual-key>

# Use a publicly-accessible URL
audio_file = "https://assembly.ai/wildfires.mp3"

# Or use a local file:
# audio_file = "./example.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    speaker_labels=True,
    # Speech understanding features
    # sentiment_analysis=True,
    # entity_detection=True,
    # auto_chapters=True,
    # summarization=True,
    # summary_type=aai.SummarizationType.bullets,
    # redact_pii=True,
    # content_safety=True,
)

transcript = aai.Transcriber().transcribe(audio_file, config=config)

if transcript.status == aai.TranscriptStatus.error:
    raise RuntimeError(f"Transcription failed: {transcript.error}")

print(f"\nFull Transcript:\n\n{transcript.text}")

# Optionally print speaker diarization results
# for utterance in transcript.utterances:
#     print(f"Speaker {utterance.speaker}: {utterance.text}")
```

4. [Prompting with Universal-3 Pro](https://www.assemblyai.com/docs/speech-to-text/prompting) (optional)

```python
import assemblyai as aai

aai.settings.base_url = "http://0.0.0.0:4000/assemblyai" # <your-proxy-base-url>/assemblyai
aai.settings.api_key = "Bearer sk-1234" # Bearer <your-virtual-key>

audio_file = "https://assemblyaiassets.com/audios/verbatim.mp3"

config = aai.TranscriptionConfig(
    speech_models=["universal-3-pro", "universal-2"],
    language_detection=True,
    prompt="Produce a transcript suitable for conversational analysis. Every disfluency is meaningful data. Include: fillers (um, uh, er, ah, hmm, mhm, like, you know, I mean), repetitions (I I, the the), restarts (I was- I went), stutters (th-that, b-but, no-not), and informal speech (gonna, wanna, gotta)",
)

transcript = aai.Transcriber().transcribe(audio_file, config)

print(transcript.text)
```

## Calling AssemblyAI EU endpoints

If you want to send your request to the AssemblyAI EU endpoint, you can do so by setting the `LITELLM_PROXY_BASE_URL` to `<your-proxy-base-url>/eu.assemblyai`


```python
import assemblyai as aai

aai.settings.base_url = "http://0.0.0.0:4000/eu.assemblyai" # <your-proxy-base-url>/eu.assemblyai
aai.settings.api_key = "Bearer sk-1234" # Bearer <your-virtual-key>

# Use a publicly-accessible URL
audio_file = "https://assembly.ai/wildfires.mp3"

# Or use a local file:
# audio_file = "./path/to/file.mp3"

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(audio_file)
print(transcript)
print(transcript.id)
```

## LLM Gateway

Use AssemblyAI's [LLM Gateway](https://www.assemblyai.com/docs/llm-gateway) as an OpenAI-compatible provider — a unified API for Claude, GPT, and Gemini models with full LiteLLM logging, guardrails, and cost tracking support.

[**See Available Models**](https://www.assemblyai.com/docs/llm-gateway#available-models)

### Usage

#### LiteLLM Python SDK

```python
import litellm
import os

os.environ["ASSEMBLYAI_API_KEY"] = "your-assemblyai-api-key"

response = litellm.completion(
    model="assemblyai/claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)

print(response.choices[0].message.content)
```

#### LiteLLM Proxy

1. Config

```yaml
model_list:
  - model_name: assemblyai/*
    litellm_params:
      model: assemblyai/*
      api_key: os.environ/ASSEMBLYAI_API_KEY
```

2. Start proxy

```bash
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it!

```python
import requests

headers = {
    "authorization": "Bearer sk-1234"  # Bearer <your-virtual-key>
}

response = requests.post(
    "http://0.0.0.0:4000/v1/chat/completions",
    headers=headers,
    json={
        "model": "assemblyai/claude-sonnet-4-5-20250929",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 1000
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```
