# *🚅 litellm*
[![PyPI Version](https://img.shields.io/pypi/v/litellm.svg)](https://pypi.org/project/litellm/)
[![PyPI Version](https://img.shields.io/badge/stable%20version-v0.1.345-blue?color=green&link=https://pypi.org/project/litellm/0.1.1/)](https://pypi.org/project/litellm/0.1.1/)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/BerriAI/litellm/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/BerriAI/litellm/tree/main)
![Downloads](https://img.shields.io/pypi/dm/litellm)
[![litellm](https://img.shields.io/badge/%20%F0%9F%9A%85%20liteLLM-OpenAI%7CAzure%7CAnthropic%7CPalm%7CCohere%7CReplicate%7CHugging%20Face-blue?color=green)](https://github.com/BerriAI/litellm)

[![](https://dcbadge.vercel.app/api/server/wuPM9dRgDw)](https://discord.gg/wuPM9dRgDw)

a light package to simplify calling OpenAI, Azure, Cohere, Anthropic, Huggingface API Endpoints. It manages: 
- translating inputs to the provider's completion and embedding endpoints
- guarantees [consistent output](https://litellm.readthedocs.io/en/latest/output/), text responses will always be available at `['choices'][0]['message']['content']`
- exception mapping - common exceptions across providers are mapped to the [OpenAI exception types](https://help.openai.com/en/articles/6897213-openai-library-error-types-guidance)
# usage
<a href='https://docs.litellm.ai/docs/completion/supported' target="_blank"><img alt='None' src='https://img.shields.io/badge/Supported_LLMs-100000?style=for-the-badge&logo=None&logoColor=000000&labelColor=000000&color=8400EA'/></a>

Demo - https://litellm.ai/playground \
Read the docs - https://docs.litellm.ai/docs/

## quick start
```
pip install litellm
```

```python
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "openai key"
os.environ["COHERE_API_KEY"] = "cohere key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)
```
Code Sample: [Getting Started Notebook](https://colab.research.google.com/drive/1gR3pY-JzDZahzpVdbGBtrNGDBmzUNJaJ?usp=sharing)

Stable version
```
pip install litellm==0.1.345
```

## Streaming Queries
liteLLM supports streaming the model response back, pass `stream=True` to get a streaming iterator in response.
Streaming is supported for OpenAI, Azure, Anthropic, Huggingface models
```python
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for chunk in response:
    print(chunk['choices'][0]['delta'])

# claude 2
result = completion('claude-2', messages, stream=True)
for chunk in result:
  print(chunk['choices'][0]['delta'])
```

# support / talk with founders
- [Our calendar 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

# why did we build this 
- **Need for simplicity**: Our code started to get extremely complicated managing & translating calls between Azure, OpenAI, Cohere
