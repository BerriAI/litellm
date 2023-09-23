# LiteLLM - Getting Started

import QuickStart from '../src/components/QuickStart.js'

## **Call 100+ LLMs using the same Input/Output Format**

## Basic usage 

```python
from litellm import completion
import os


## set ENV variables
os.environ["OPENAI_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your openai key
os.environ["COHERE_API_KEY"] = "sk-litellm-7_NPZhMGxY2GoHC59LgbDw" # [OPTIONAL] replace with your cohere key

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)
```

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