# *🚅 litellm*
[![liteLLM Dev Tests](https://github.com/BerriAI/litellm/actions/workflows/tests.yml/badge.svg)](https://github.com/BerriAI/litellm/actions/workflows/tests.yml)
[![Publish to PyPI](https://github.com/BerriAI/litellm/actions/workflows/publish_pypi.yml/badge.svg?branch=main)](https://github.com/BerriAI/litellm/actions/workflows/publish_pypi.yml) 
[![PyPI Version](https://img.shields.io/pypi/v/litellm.svg)](https://pypi.org/project/litellm/)

[![](https://dcbadge.vercel.app/api/server/wuPM9dRgDw)](https://discord.gg/wuPM9dRgDw)

a simple & light 100 line package to call OpenAI, Azure, Cohere, Anthropic API Endpoints 

litellm manages:
- translating inputs to completion and embedding endpoints
- guarantees consistent output, text responses will always be available at `['choices'][0]['message']['content']`

Read the docs - https://litellm.readthedocs.io/en/latest/
# usage
## installation
```
pip install litellm
```

Stable version
```
pip install litellm==0.1.1
```

* Code Sample: [Getting Started Notebook](https://colab.research.google.com/drive/1gR3pY-JzDZahzpVdbGBtrNGDBmzUNJaJ?usp=sharing)
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

# hosted version
- [Grab time if you want access 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)

# why did I build this 
- **Need for simplicity**: My code started to get extremely complicated managing & translating calls between Azure, OpenAI, Cohere

# Support
Contact us at ishaan@berri.ai / krrish@berri.ai
