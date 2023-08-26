# Promptlayer Tutorial

Promptlayer is a platform for prompt engineers. Log OpenAI requests. Search usage history. Track performance. Visually manage prompt templates.

<video controls width='900' >
  <source src='https://llmonitor.com/videos/demo-annotated.mp4'/>
</video>

## Use Promptlayer to log requests across all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)

liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

### Using Callbacks

# Get your PromptLayer API Key from https://promptlayer.com/

Use just 2 lines of code, to instantly log your responses **across all providers** with promptlayer:

```python
litellm.success_callback = ["promptlayer"]

```

Complete code

```python
from litellm import completion

## set env variables
os.environ["PROMPTLAYER_API_KEY"] = "your"

os.environ["OPENAI_API_KEY"], os.environ["COHERE_API_KEY"] = "", ""

# set callbacks
litellm.success_callback = ["promptlayer"]
litellm.failure_callback = ["llmonitor"]

#openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])

#cohere call
response = completion(model="command-nightly", messages=[{"role": "user", "content": "Hi 👋 - i'm cohere"}])
```
