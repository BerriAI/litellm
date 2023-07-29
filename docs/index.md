# *🚅 litellm*
a light 100 line package to simplify calling OpenAI, Azure, Cohere, Anthropic APIs 

###### litellm manages:
* Calling all LLM APIs using the OpenAI format - `completion(model, messages)`
* Consistent output for all LLM APIs, text responses will always be available at `['choices'][0]['message']['content']`
* **[Advanced]** Automatically logging your output to Sentry, Posthog, Slack [see liteLLM Client](/docs/advanced.md)

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
Need Help / Support : [see troubleshooting](/docs/troubleshoot.md)

## Why did we build liteLLM 
- **Need for simplicity**: Our code started to get extremely complicated managing & translating calls between Azure, OpenAI, Cohere

## Support
* [Meet with us 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
* Contact us at ishaan@berri.ai / krrish@berri.ai
