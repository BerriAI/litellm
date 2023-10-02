# AWS Bedrock
Anthropic, Amazon Titan, A121 LLMs are Supported on Bedrock

## Pre-Requisites
LiteLLM requires `boto3` to be installed on your system for Bedrock requests
```shell
pip install boto3>=1.28.57
```

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_Bedrock.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Required Environment Variables
```python
os.environ["AWS_ACCESS_KEY_ID"] = ""  # Access key
os.environ["AWS_SECRET_ACCESS_KEY"] = "" # Secret access key
os.environ["AWS_REGION_NAME"] = "" # us-east-1, us-east-2, us-west-1, us-west-2
```

## Usage
```python
import os 
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="bedrock/anthropic.claude-instant-v1", 
            messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

## Supported AWS Bedrock Models
Here's an example of using a bedrock model with LiteLLM 

| Model Name               | Command                                                          | Environment Variables                                              |
|--------------------------|------------------------------------------------------------------|---------------------------------------------------------------------|
| Anthropic Claude-V2      | `completion(model='bedrock/anthropic.claude-v2', messages=messages)`   | `os.environ['ANTHROPIC_ACCESS_KEY_ID']`, `os.environ['ANTHROPIC_SECRET_ACCESS_KEY']`           |
| Anthropic Claude-Instant V1 | `completion(model='bedrock/anthropic.claude-instant-v1', messages=messages)` | `os.environ['ANTHROPIC_ACCESS_KEY_ID']`, `os.environ['ANTHROPIC_SECRET_ACCESS_KEY']`           |
| Anthropic Claude-V1      | `completion(model='bedrock/anthropic.claude-v1', messages=messages)`   | `os.environ['ANTHROPIC_ACCESS_KEY_ID']`, `os.environ['ANTHROPIC_SECRET_ACCESS_KEY']`           |
| AI21 J2-Ultra            | `completion(model='bedrock/ai21.j2-ultra', messages=messages)` | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| AI21 J2-Mid              | `completion(model='bedrock/ai21.j2-mid', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |


## Streaming

```python
import os 
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
            model="bedrock/anthropic.claude-instant-v1", 
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            stream=True
)

for chunk in response:
    print(chunk)
```

### Example Streaming Output Chunk
```json
{
  "choices": [
    {
      "finish_reason": null,
      "index": 0,
      "delta": {
        "content": "ase can appeal the case to a higher federal court. If a higher federal court rules in a way that conflicts with a ruling from a lower federal court or conflicts with a ruling from a higher state court, the parties involved in the case can appeal the case to the Supreme Court. In order to appeal a case to the Sup"
      }
    }
  ],
  "created": null,
  "model": "amazon.titan-tg1-large",
  "usage": {
    "prompt_tokens": null,
    "completion_tokens": null,
    "total_tokens": null
  }
}
```
