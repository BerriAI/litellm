---
displayed_sidebar: tutorialSidebar
---
# 🚅 LiteLLM

import CrispChat from '../src/components/CrispChat.js'

Call all LLM APIs using the OpenAI format [Anthropic, Huggingface, Cohere, TogetherAI, Azure, OpenAI, etc.]

LiteLLM manages:
- Translating inputs to the provider's completion and embedding endpoints
- Guarantees consistent output, text responses will always be available at `['choices'][0]['message']['content']`
- Exception mapping - common exceptions across providers are mapped to the OpenAI exception types

## Quick Start
Code Sample: [Getting Started Notebook](https://colab.research.google.com/drive/1gR3pY-JzDZahzpVdbGBtrNGDBmzUNJaJ?usp=sharing)

```shell
pip install litellm
```

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

LiteLLM supports streaming the model response back, pass `stream=True` to get a streaming iterator in response.
Streaming is supported for all models.

```python
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for chunk in response:
    print(chunk['choices'][0]['delta'])

# claude 2
result = completion('claude-2', messages, stream=True)
for chunk in result:
  print(chunk['choices'][0]['delta'])
```

# Support / talk with founders

- [Our calendar 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

# Why did we build this

- **Need for simplicity**: Our code started to get extremely complicated managing & translating calls between Azure, OpenAI, Cohere
