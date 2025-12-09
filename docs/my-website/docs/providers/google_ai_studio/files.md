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

