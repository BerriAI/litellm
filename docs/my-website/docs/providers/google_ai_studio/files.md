import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# [BETA] Google AI Studio (Gemini) Files API

Use this to upload files to Google AI Studio (Gemini).

Useful to pass in large media files to Gemini's `/generateContent` endpoint.

| Action | Supported | 
|----------|-----------|
| `create` | Yes |
| `delete` | No |
| `retrieve` | No |
| `list` | No |

## Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import base64
import requests
from litellm import completion, create_file
import os


### UPLOAD FILE ### 

# Fetch the audio file and convert it to a base64 encoded string
url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
response = requests.get(url)
response.raise_for_status()
wav_data = response.content
encoded_string = base64.b64encode(wav_data).decode('utf-8')


file = create_file(
    file=wav_data,
    purpose="user_data",
    extra_headers={"custom-llm-provider": "gemini"},
    api_key=os.getenv("GEMINI_API_KEY"),
)

print(f"file: {file}")

assert file is not None


### GENERATE CONTENT ### 
completion = completion(
    model="gemini-2.0-flash",
    messages=[
        {
            "role": "user",
            "content": [
                { 
                    "type": "text",
                    "text": "What is in this recording?"
                },
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                        "filename": "my-test-name",
                        "format": "audio/wav"
                    }
                }
            ]
        },
    ]
)

print(completion.choices[0].message)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
    - model_name: "gemini-2.0-flash"
      litellm_params:
        model: gemini/gemini-2.0-flash
        api_key: os.environ/GEMINI_API_KEY
```

2. Start proxy

```bash
litellm --config config.yaml
```

3. Test it

```python
import base64
import requests
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234"
)

# Fetch the audio file and convert it to a base64 encoded string
url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
response = requests.get(url)
response.raise_for_status()
wav_data = response.content
encoded_string = base64.b64encode(wav_data).decode('utf-8')


file = client.files.create(
    file=wav_data,
    purpose="user_data",
    extra_body={"target_model_names": "gemini-2.0-flash"}
)

print(f"file: {file}")

assert file is not None

completion = client.chat.completions.create(
    model="gemini-2.0-flash",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": [
                { 
                    "type": "text",
                    "text": "What is in this recording?"
                },
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                        "filename": "my-test-name",
                        "format": "audio/wav"
                    }
                }
            ]
        },
    ],
    extra_body={"drop_params": True}
)

print(completion.choices[0].message)
```




</TabItem>
</Tabs>

## Azure Blob Storage Integration

LiteLLM supports using Azure Blob Storage as a target storage backend for Gemini file uploads. This allows you to store files in Azure Data Lake Storage Gen2 instead of Google's managed storage.

### Step 1: Setup Azure Blob Storage

Configure your Azure Blob Storage account by setting the following environment variables:

**Required Environment Variables:**
- `AZURE_STORAGE_ACCOUNT_NAME` - Your Azure Storage account name
- `AZURE_STORAGE_FILE_SYSTEM` - The container/filesystem name where files will be stored
- `AZURE_STORAGE_ACCOUNT_KEY` - Your account key

### Step 2: Pass Azure Blob Storage as Target Storage

When uploading files, specify `target_storage: "azure_storage"` to use Azure Blob Storage instead of the default storage.

**Supported File Types:**

Azure Blob Storage supports all Gemini-compatible file types:

- **Images**: PNG, JPEG, WEBP
- **Audio**: AAC, FLAC, MP3, MPA, MPEG, MPGA, OPUS, PCM, WAV, WEBM
- **Video**: FLV, MOV, MPEG, MPEGPS, MPG, MP4, WEBM, WMV, 3GPP
- **Documents**: PDF, TXT

> **Note:** Only small files can be sent as inline data because the total request size limit is 20 MB.


### Step 3: Upload Files with Azure Blob Storage for Gemini

<Tabs>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
    - model_name: "gemini-2.5-flash"
      litellm_params:
        model: gemini/gemini-2.5-flash
        api_key: os.environ/GEMINI_API_KEY
```

2. Set environment variables

```bash
export AZURE_STORAGE_ACCOUNT_NAME="your-storage-account"
export AZURE_STORAGE_FILE_SYSTEM="your-container-name"
export AZURE_STORAGE_ACCOUNT_KEY="your-account-key"
```
or add them in your `.env`

3. Start proxy

```bash
litellm --config config.yaml
```

4. Upload file with Azure Blob Storage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234"
)

# Upload file to Azure Blob Storage
file = client.files.create(
    file=open("document.pdf", "rb"),
    purpose="user_data",
    extra_body={
        "target_model_names": "gemini-2.0-flash",
        "target_storage": "azure_storage"  # 👈 Use Azure Blob Storage
    }
)

print(f"File uploaded to Azure Blob Storage: {file.id}")

# Use the file with Gemini
completion = client.chat.completions.create(
    model="gemini-2.0-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this document"},
                {
                    "type": "file",
                    "file": {
                        "file_id": file.id,
                    }
                }
            ]
        }
    ]
)

print(completion.choices[0].message.content)
```

</TabItem>
<TabItem value="curl" label="cURL">

```bash
# Upload file with Azure Blob Storage
curl -X POST "http://0.0.0.0:4000/v1/files" \
  -H "Authorization: Bearer sk-1234" \
  -F "file=@document.pdf" \
  -F "purpose=user_data" \
  -F "target_storage=azure_storage" \
  -F "target_model_names=gemini-2.0-flash" \
  -F "custom_llm_provider=gemini"

# Use the file with Gemini
curl -X POST "http://0.0.0.0:4000/v1/chat/completions" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.0-flash",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Summarize this document"},
          {
            "type": "file",
            "file": {
              "file_id": "file-id-from-upload",
              "format": "application/pdf"
            }
          }
        ]
      }
    ]
  }'
```

</TabItem>
</Tabs>

:::info
Files uploaded to Azure Blob Storage are stored in your Azure account and can be accessed via the returned file ID. The file URL format is: `https://{account}.blob.core.windows.net/{container}/{path}`
:::

