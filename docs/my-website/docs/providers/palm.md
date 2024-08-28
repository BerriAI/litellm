# PaLM API - Google

:::warning

Warning: [The PaLM API is decomissioned by Google](https://ai.google.dev/palm_docs/deprecation) The PaLM API is scheduled to be decomissioned in October 2024. Please upgrade to the Gemini API or Vertex AI API

:::

## Pre-requisites
* `pip install -q google-generativeai`

## Sample Usage
```python
from litellm import completion
import os

os.environ['PALM_API_KEY'] = ""
response = completion(
    model="palm/chat-bison", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['PALM_API_KEY'] = ""
response = completion(
    model="palm/chat-bison", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Chat Models
| Model Name       | Function Call                        | Required OS Variables    |
|------------------|--------------------------------------|-------------------------|
| chat-bison       | `completion('palm/chat-bison', messages)` | `os.environ['PALM_API_KEY']` |
