import Image from '@theme/IdealImage';

# LangFuse Tutorial

LangFuse is open Source Observability & Analytics for LLM Apps
Detailed production traces and a granular view on quality, cost and latency

<Image img={require('../../img/langfuse.png')} />

## Usage - log all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)
liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

## Pre-Requisites
```shell
pip install litellm langfuse
```

### Using Callbacks
Get your Langfuse API Keys from https://cloud.langfuse.com/

Use just 2 lines of code, to instantly log your responses **across all providers** with langfuse:

```python
litellm.success_callback = ["langfuse"]
```

### Complete code

```python
import litellm
from litellm import completion
import os

# from https://cloud.langfuse.com/
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""


# OpenAI and Cohere keys 
# You can use any of the litellm supported providers: https://docs.litellm.ai/docs/providers
os.environ['OPENAI_API_KEY']=""
os.environ['COHERE_API_KEY']=""

# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["langfuse"] 
 
# openai call
response = completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
 
print(response)

# cohere call
response = completion(
  model="command-nightly",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm cohere"}
  ]
)

print(response)

```
