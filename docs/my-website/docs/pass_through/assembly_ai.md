# Assembly AI

Pass-through endpoints for Assembly AI - call Assembly AI endpoints, in native format (no translation).

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ✅ | works across all integrations |
| Logging | ✅ | works across all integrations |


Supports **ALL** Assembly AI Endpoints

[**See All Assembly AI Endpoints**](https://www.assemblyai.com/docs/api-reference)

## Quick Start

Let's call the Assembly AI [`/v2/transcripts` endpoint](https://www.assemblyai.com/docs/api-reference/transcripts)

1. Add Assembly AI API Key to your environment 

```bash
export ASSEMBLYAI_API_KEY=""
```

2. Start LiteLLM Proxy 

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

Let's call the Assembly AI `/v2/transcripts` endpoint

```python
LITELLM_VIRTUAL_KEY = "sk-1234" # <your-virtual-key>
LITELLM_PROXY_BASE_URL = "http://0.0.0.0:4000/assemblyai" # <your-proxy-base-url>/assemblyai

aai.settings.api_key = f"Bearer {LITELLM_VIRTUAL_KEY}"
aai.settings.base_url = LITELLM_PROXY_BASE_URL

# URL of the file to transcribe
FILE_URL = "https://assembly.ai/wildfires.mp3"

# You can also transcribe a local file by passing in a file path
# FILE_URL = './path/to/file.mp3'

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)
print(transcript)
print(transcript.id)
```

