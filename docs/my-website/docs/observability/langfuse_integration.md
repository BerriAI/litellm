import Image from '@theme/IdealImage';

# Langfuse - Logging LLM Input/Output

LangFuse is open Source Observability & Analytics for LLM Apps
Detailed production traces and a granular view on quality, cost and latency

<Image img={require('../../img/langfuse.png')} />

## Usage - log all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)
liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

## Pre-Requisites
Ensure you have run `pip install langfuse` for this integration
```shell
pip install langfuse litellm
```

## Quick Start
```python
# pip install langfuse 
import litellm
import os

# from https://cloud.langfuse.com/
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ['OPENAI_API_KEY']=""

# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["langfuse"] 
 
# openai call
response = litellm.completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ]
)
```

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/logging_observability/LiteLLM_Langfuse.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Get your Langfuse API Keys from https://cloud.langfuse.com/

Use just 2 lines of code, to instantly log your responses **across all providers** with langfuse:

```python
litellm.success_callback = ["langfuse"]
```

## API keys for Langfuse
Set the following variables in your .env
```python
# Required 
os.environ["LANGFUSE_SECRET_KEY"]
os.environ["LANGFUSE_PUBLIC_KEY"]

# Optional, defaults to https://cloud.langfuse.com
os.environ["LANGFUSE_HOST"] # optional
```

## Complete code

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

## Advanced
### Set Custom Generation names, pass metadata

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

# set langfuse as a callback, litellm will send the data to langfuse
litellm.success_callback = ["langfuse"] 
 
# openai call
response = completion(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hi 👋 - i'm openai"}
  ],
  metadata = {
    "generation_name": "litellm-ishaan-gen", # set langfuse generation name
    # custom metadata fields
    "project": "litellm-proxy" 
  }
)
 
print(response)

```



### Example - FastAPI Server with LiteLLM + Langfuse
https://replit.com/@BerriAI/LiteLLMFastAPILangfuse

<iframe src="https://replit.com/@BerriAI/LiteLLMFastAPILangfuse?embed=1" width="800" height="600"></iframe>

