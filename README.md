# *🚅 litellm*
[![PyPI Version](https://img.shields.io/pypi/v/litellm.svg)](https://pypi.org/project/litellm/)
[![PyPI Version](https://img.shields.io/badge/stable%20version-v0.1.345-blue?color=green&link=https://pypi.org/project/litellm/0.1.1/)](https://pypi.org/project/litellm/0.1.1/)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/BerriAI/litellm/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/BerriAI/litellm/tree/main)
![Downloads](https://img.shields.io/pypi/dm/litellm)
[![litellm](https://img.shields.io/badge/%20%F0%9F%9A%85%20liteLLM-OpenAI%7CAzure%7CAnthropic%7CPalm%7CCohere%7CReplicate%7CHugging%20Face-blue?color=green)](https://github.com/BerriAI/litellm)

Get Support / Join the community 👉 [![](https://dcbadge.vercel.app/api/server/wuPM9dRgDw)](https://discord.gg/wuPM9dRgDw)

a simple & light package to call OpenAI, Azure, Cohere, Anthropic API Endpoints 

litellm manages:
- translating inputs to completion and embedding endpoints
- guarantees consistent output, text responses will always be available at `['choices'][0]['message']['content']`

# usage

Read the docs - https://litellm.readthedocs.io/en/latest/

## quick start
```
pip install litellm
```

```python
from litellm import completion

## set ENV variables
# ENV variables can be set in .env file, too. Example in .env.example
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)

# azure openai call
response = completion("chatgpt-test", messages, azure=True)

# openrouter call
response = completion("google/palm-2-codechat-bison", messages)
```
Code Sample: [Getting Started Notebook](https://colab.research.google.com/drive/1gR3pY-JzDZahzpVdbGBtrNGDBmzUNJaJ?usp=sharing)

Stable version
```
pip install litellm==0.1.345
```

## Streaming Queries
liteLLM supports streaming the model response back, pass `stream=True` to get a streaming iterator in response.
```python
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for chunk in response:
    print(chunk['choices'][0]['delta'])
```

# hosted version
- [Grab time if you want access 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)

# why did we build this 
- **Need for simplicity**: Our code started to get extremely complicated managing & translating calls between Azure, OpenAI, Cohere

# Support
Contact us at ishaan@berri.ai / krrish@berri.ai
