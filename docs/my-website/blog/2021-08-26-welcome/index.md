# 🚅 litellm
A light 100 line package to simplify calling OpenAI, Azure, Cohere, Anthropic APIs 

###### litellm manages:
* Calling all LLM APIs using the OpenAI format - `completion(model, messages)`
* Consistent output for all LLM APIs, text responses will always be available at `['choices'][0]['message']['content']`
* Consistent Exceptions for all LLM APIs, we map RateLimit, Context Window, and Authentication Error exceptions across all providers to their OpenAI equivalents. [see Code](https://github.com/BerriAI/litellm/blob/ba1079ff6698ef238c5c7f771dd2b698ec76f8d9/litellm/utils.py#L250)

###### observability:
* Logging - see exactly what the raw model request/response is by plugging in your own function `completion(.., logger_fn=your_logging_fn)` and/or print statements from the package `litellm.set_verbose=True`
* Callbacks - automatically send your data to Helicone, Sentry, Posthog, Slack - `litellm.success_callbacks`, `litellm.failure_callbacks` [see Callbacks](https://litellm.readthedocs.io/en/latest/advanced/)

## Quick Start
Go directly to code: [Getting Started Notebook](https://colab.research.google.com/drive/1gR3pY-JzDZahzpVdbGBtrNGDBmzUNJaJ?usp=sharing)
### Installation
```
pip install litellm
```

### Usage
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
Need Help / Support : [see troubleshooting](https://litellm.readthedocs.io/en/latest/troubleshoot)

## Why did we build liteLLM 
- **Need for simplicity**: Our code started to get extremely complicated managing & translating calls between Azure, OpenAI, Cohere

## Support
* [Meet with us 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
* Contact us at ishaan@berri.ai / krrish@berri.ai
