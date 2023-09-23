import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LiteLLM - Getting Started

import QuickStart from '../src/components/QuickStart.js'

## **Call 100+ LLMs using the same Input/Output Format**

## Basic usage 
<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_Getting_Started.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<Tabs>
<TabItem value="openai" label="OpenAI">

```python
from litellm import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your openai key

response = completion(
  model="gpt-3.5-turbo", 
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>
<TabItem value="anthropic" label="Anthropic">

```python
from litellm import completion
import os

## set ENV variables
os.environ["ANTHROPIC_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your openai key

response = completion(
  model="claude-2", 
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

<TabItem value="vertex" label="VertexAI">

```python
from litellm import completion
import os

# auth: run 'gcloud auth application-default'
os.environ["VERTEX_PROJECT"] = "hardy-device-386718"
os.environ["VERTEX_LOCATION"] = "us-central1"

response = completion(
  model="chat-bison", 
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>

<TabItem value="hugging" label="HuggingFace">

```python
from litellm import completion 
import os

os.environ["HUGGINGFACE_API_KEY"] = "huggingface_api_key" 

# e.g. Call 'WizardLM/WizardCoder-Python-34B-V1.0' hosted on HF Inference endpoints
response = completion(
  model="huggingface/WizardLM/WizardCoder-Python-34B-V1.0",
  messages=[{ "content": "Hello, how are you?","role": "user"}], 
  api_base="https://my-endpoint.huggingface.cloud"
)

print(response)
```

</TabItem>

<TabItem value="azure" label="Azure OpenAI">

```python
from litellm import completion
import os

## set ENV variables
os.environ["AZURE_API_KEY"] = ""
os.environ["AZURE_API_BASE"] = ""
os.environ["AZURE_API_VERSION"] = ""

# azure call
response = completion(
  "azure/<your_deployment_id>", 
  messages = [{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>


<TabItem value="ollama" label="Ollama">

```python
from litellm import completion

response = completion(
            model="ollama/llama2", 
            messages = [{ "content": "Hello, how are you?","role": "user"}], 
            api_base="http://localhost:11434"
)
```
</TabItem>

</Tabs>



## Streaming

Same example from before. Just pass in `stream=True` in the completion args. 
```python
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)

# cohere call
response = completion("command-nightly", messages, stream=True)

print(response)
```

More details 👉 
* [streaming + async](./completion/stream.md)
* [tutorial for streaming Llama2 on TogetherAI](./tutorials/TogetherAI_liteLLM.md)

## Exception handling 

LiteLLM maps exceptions across all supported providers to the OpenAI exceptions. All our exceptions inherit from OpenAI's exception types, so any error-handling you have for that, should work out of the box with LiteLLM. 

```python 
from openai.errors import OpenAIError
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "bad-key"
try: 
    # some code 
    completion(model="claude-instant-1", messages=[{"role": "user", "content": "Hey, how's it going?"}])
except OpenAIError as e:
    print(e)
```

## Calculate Costs & Usage

## Caching with LiteLLM

## LiteLLM API

## Send Logs to Promptlayer


More details 👉 
* [exception mapping](./exception_mapping.md)
* [retries + model fallbacks for completion()](./completion/reliable_completions.md)
* [tutorial for model fallbacks with completion()](./tutorials/fallbacks.md)