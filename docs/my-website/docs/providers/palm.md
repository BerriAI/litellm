# PaLM API - Google

## Pre-requisites
* `pip install -q google-generativeai`

## Sample Usage
```python
import litellm
import os

os.environ['PALM_API_KEY'] = ""
response = completion(
    model="palm/chat-bison", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)
```

## Sample Usage - Streaming
```python
import litellm
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
