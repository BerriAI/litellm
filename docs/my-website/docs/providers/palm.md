# PaLM API - Google
https://developers.generativeai.google/products/palm

## Pre-requisites
* `pip install google-cloud-aiplatform`
* Authentication: 
    * run `gcloud auth application-default login` See [Google Cloud Docs](https://cloud.google.com/docs/authentication/external/set-up-adc)
    * Alternatively you can set `application_default_credentials.json`

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
| Model Name       | Function Call                        |
|------------------|--------------------------------------|
| chat-bison       | `completion('palm/chat-bison', messages)`     |