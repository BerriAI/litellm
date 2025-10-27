# Assembly AI

Pass-through endpoints for Assembly AI - call Assembly AI endpoints, in native format (no translation).

| Feature | Supported | Notes | 
|-------|-------|-------|
| Cost Tracking | ✅ | works across all integrations |
| Logging | ✅ | works across all integrations |


Supports **ALL** Assembly AI Endpoints

[**See All Assembly AI Endpoints**](https://www.assemblyai.com/docs/api-reference)


<iframe width="840" height="500" src="https://www.loom.com/embed/aac3f4d74592448992254bfa79b9f62d?sid=267cd0ab-d92b-42fa-b97a-9f385ef8930c" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

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
import assemblyai as aai

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

## Calling Assembly AI EU endpoints

If you want to send your request to the Assembly AI EU endpoint, you can do so by setting the `LITELLM_PROXY_BASE_URL` to `<your-proxy-base-url>/eu.assemblyai`


```python
import assemblyai as aai

LITELLM_VIRTUAL_KEY = "sk-1234" # <your-virtual-key>
LITELLM_PROXY_BASE_URL = "http://0.0.0.0:4000/eu.assemblyai" # <your-proxy-base-url>/eu.assemblyai

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
