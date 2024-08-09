# 🧊 Keywords AI - LLM monitoring platform

:::tip

This is community maintained. Please make an issue if you run into a bug:
https://github.com/BerriAI/litellm

:::

[Keywords AI](https://keywordsai.co/) makes it easy for developers to build LLM applications. With 2 lines of code, developers get a complete monitoring platform that speeds up deploying & monitoring AI apps in production.

## Using Keywords AI with LiteLLM

LiteLLM provides `callbacks`, allowing you to easily log data to Keywords AI based on the status of your responses.

### Supported LLM Providers

Keywords AI can log requests across [various LLM providers](https://docs.keywordsai.co/integration/supported-models)

### Integration Methods

There are two main approaches to integrate Keywords AI with LiteLLM:

1. Using callbacks
2. Using Keywords AI as a proxy

### Approach 1: Use Callbacks

Use just 2 line of code to log your responses with Keywords AI:

```python
from litellm.integrations.keywordsai import KeywordsAILogger
litellm.callbacks = [KeywordsAILogger]
```

Complete Code

```python
import os
from litellm import completion
from litellm.integrations.keywordsai import KeywordsAILogger
litellm.callbacks = [KeywordsAILogger]
## Set env variables
os.environ["KEYWORDS_AI_API_KEY"] = "KEYWORDS_AI_API_KEY"

# OpenAI call
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hi, Keywords AI!!"}],
)

print(response)
```

### Approach 2: Use Keywords AI as a proxy

To use Keywords AI as a proxy for your LLM requests:

Complete Code:

```python
litellm.api_base = "https://api.keywordsai.co/api/"
KEYWORDS_AI_API_KEY = os.getenv("KEYWORDS_AI_API_KEY")

response = litellm.completion(
    api_key=KEYWORDS_AI_API_KEY, # !!!!!!! Use the keyowrdsai api key in your completion call !!!!!!!
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "How does a court case get to the Supreme Court?"}]
)

print(response)
# Go to https://platform.keywordsai.co/ to see the log
```
