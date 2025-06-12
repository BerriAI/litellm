import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /audio/transcriptions

Use this to loadbalance across Azure + OpenAI. 

## Quick Start

### LiteLLM Python SDK

```python showLineNumbers
from litellm import transcription
import os 

# set api keys 
os.environ["OPENAI_API_KEY"] = ""
audio_file = open("/path/to/audio.mp3", "rb")

response = transcription(model="whisper", file=audio_file)

print(f"response: {response}")
```

### LiteLLM Proxy

### Add model to config 


<Tabs>
<TabItem value="openai" label="OpenAI">

```yaml showLineNumbers
model_list:
- model_name: whisper
  litellm_params:
    model: whisper-1
    api_key: os.environ/OPENAI_API_KEY
  model_info:
    mode: audio_transcription
    
general_settings:
  master_key: sk-1234
```
</TabItem>
<TabItem value="openai+azure" label="OpenAI + Azure">

```yaml showLineNumbers
model_list:
- model_name: whisper
  litellm_params:
    model: whisper-1
    api_key: os.environ/OPENAI_API_KEY
  model_info:
    mode: audio_transcription
- model_name: whisper
  litellm_params:
    model: azure/azure-whisper
    api_version: 2024-02-15-preview
    api_base: os.environ/AZURE_EUROPE_API_BASE
    api_key: os.environ/AZURE_EUROPE_API_KEY
  model_info:
    mode: audio_transcription

general_settings:
  master_key: sk-1234
```

</TabItem>
</Tabs>

### Start proxy 

```bash
litellm --config /path/to/config.yaml 

# RUNNING on http://0.0.0.0:8000
```

### Test 

<Tabs>
<TabItem value="curl" label="Curl">

```bash
curl --location 'http://0.0.0.0:8000/v1/audio/transcriptions' \
--header 'Authorization: Bearer sk-1234' \
--form 'file=@"/Users/krrishdholakia/Downloads/gettysburg.wav"' \
--form 'model="whisper"'
```

</TabItem>
<TabItem value="openai" label="OpenAI Python SDK">

```python showLineNumbers
from openai import OpenAI
client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:8000"
)


audio_file = open("speech.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper",
  file=audio_file
)
```
</TabItem>
</Tabs>

## Fallbacks

LiteLLM supports fallbacks for audio transcription, allowing you to automatically retry failed requests with different models or providers.

### Router Configuration

<Tabs>
<TabItem value="same-provider" label="Same Provider Fallbacks">

```python showLineNumbers
from litellm import Router
import os

# Configure multiple deployments of the same model with fallbacks
router = Router(
    model_list=[
        {
            "model_name": "whisper-primary",
            "litellm_params": {
                "model": "whisper-1",
                "api_key": os.environ["OPENAI_API_KEY"],
            }
        },
        {
            "model_name": "whisper-backup",
            "litellm_params": {
                "model": "whisper-1", 
                "api_key": os.environ["OPENAI_BACKUP_KEY"],
            }
        }
    ],
    fallbacks=[
        {"whisper-primary": ["whisper-backup"]}
    ],
    num_retries=2
)

# Use with fallbacks
audio_file = open("/path/to/audio.mp3", "rb")
response = await router.atranscription(
    file=audio_file,
    model="whisper-primary"
)
```

</TabItem>
<TabItem value="cross-provider" label="Cross-Provider Fallbacks">

```python showLineNumbers
from litellm import Router
import os

# Configure fallbacks across different providers
router = Router(
    model_list=[
        {
            "model_name": "openai-whisper",
            "litellm_params": {
                "model": "whisper-1",
                "api_key": os.environ["OPENAI_API_KEY"],
            }
        },
        {
            "model_name": "azure-whisper",
            "litellm_params": {
                "model": "azure/whisper-deployment",
                "api_key": os.environ["AZURE_API_KEY"],
                "api_base": os.environ["AZURE_API_BASE"],
                "api_version": "2024-02-15-preview"
            }
        },
        {
            "model_name": "groq-whisper",
            "litellm_params": {
                "model": "groq/whisper-large-v3",
                "api_key": os.environ["GROQ_API_KEY"],
            }
        }
    ],
    fallbacks=[
        {"openai-whisper": ["azure-whisper", "groq-whisper"]}
    ],
    num_retries=1
)

# Use with cross-provider fallbacks
audio_file = open("/path/to/audio.mp3", "rb")
response = await router.atranscription(
    file=audio_file,
    model="openai-whisper"  # Will fallback to Azure, then Groq if needed
)
```

</TabItem>
</Tabs>

### Proxy Configuration with Fallbacks

```yaml showLineNumbers
model_list:
  # Primary OpenAI deployment
  - model_name: whisper-primary
    litellm_params:
      model: whisper-1
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: audio_transcription
      
  # Azure backup deployment  
  - model_name: whisper-azure-backup
    litellm_params:
      model: azure/whisper-deployment
      api_version: 2024-02-15-preview
      api_base: os.environ/AZURE_API_BASE
      api_key: os.environ/AZURE_API_KEY
    model_info:
      mode: audio_transcription
      
  # Groq backup deployment
  - model_name: whisper-groq-backup
    litellm_params:
      model: groq/whisper-large-v3
      api_key: os.environ/GROQ_API_KEY
    model_info:
      mode: audio_transcription

# Configure fallback chain using router_settings
router_settings:
  fallbacks: [{"whisper-primary": ["whisper-azure-backup", "whisper-groq-backup"]}]
  num_retries: 2
  timeout: 600  # 10 minutes for audio processing

general_settings:
  master_key: sk-1234
```

### Test Fallbacks

To test fallbacks, you can use `mock_testing_fallbacks=true` to trigger fallback behavior:

<Tabs>
<TabItem value="curl" label="Curl">

```bash
# Test with mock_testing_fallbacks to trigger fallbacks
curl --location 'http://0.0.0.0:8000/v1/audio/transcriptions' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: multipart/form-data' \
--form 'file=@"/path/to/audio.mp3"' \
--form 'model="whisper-primary"' \
--form 'mock_testing_fallbacks="true"'
```

</TabItem>
<TabItem value="python" label="Python SDK">

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:8000"
)

audio_file = open("/path/to/audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-primary",
    file=audio_file,
    extra_body={
        "mock_testing_fallbacks": True  # Trigger fallbacks for testing
    }
)
print(transcript.text)
```

</TabItem>
</Tabs>

### Advanced Fallback Configuration

LiteLLM supports different types of fallbacks for audio transcription:

#### Content Policy Fallbacks
Fallback to different providers if content policy violations occur:

```yaml
router_settings:
  content_policy_fallbacks: [{"whisper-primary": ["whisper-groq-backup"]}]
```

#### Default Fallbacks
Set default fallbacks for any misconfigured models:

```yaml
router_settings:
  default_fallbacks: ["whisper-groq-backup"]
  fallbacks: [{"whisper-primary": ["whisper-azure-backup", "whisper-groq-backup"]}]
```

#### Client-Side Fallbacks
You can also specify fallbacks directly in the request:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234", 
    base_url="http://0.0.0.0:8000"
)

audio_file = open("/path/to/audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-primary",
    file=audio_file,
    extra_body={
        "fallbacks": ["whisper-azure-backup"]  # Client-side fallback
    }
)
```

### Key Benefits

- **Reliability**: Automatic failover if primary service is down
- **Rate Limit Handling**: Switch providers when hitting rate limits  
- **Cost Optimization**: Use cheaper providers as backups
- **Geographic Redundancy**: Fallback to different regions/providers

## Supported Providers

- OpenAI
- Azure
- [Fireworks AI](./providers/fireworks_ai.md#audio-transcription)
- [Groq](./providers/groq.md#speech-to-text---whisper)
- [Deepgram](./providers/deepgram.md)