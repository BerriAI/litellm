# Promptlayer Tutorial

Promptlayer is a platform for prompt engineers. Log OpenAI requests. Search usage history. Track performance. Visually manage prompt templates.

<Image img={require('../../img/promptlayer.png')} />

## Use Promptlayer to log requests across all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)

liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

### Using Callbacks

Get your PromptLayer API Key from https://promptlayer.com/

Use just 2 lines of code, to instantly log your responses **across all providers** with promptlayer:

```python
litellm.success_callback = ["promptlayer"]

```

Complete code

```python
from litellm import completion

## set env variables
os.environ["PROMPTLAYER_API_KEY"] = "your-promptlayer-key"

os.environ["OPENAI_API_KEY"], os.environ["COHERE_API_KEY"] = "", ""

# set callbacks
litellm.success_callback = ["promptlayer"]

#openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])

#cohere call
response = completion(model="command-nightly", messages=[{"role": "user", "content": "Hi 👋 - i'm cohere"}])
```

### Logging Metadata 

You can also log completion call metadata to Promptlayer. 

You can add metadata to a completion call through the metadata param: 
```python 
completion(model,messages, metadata={"model": "ai21"})
```

**Complete Code**
```python
from litellm import completion

## set env variables
os.environ["PROMPTLAYER_API_KEY"] = "your-promptlayer-key"

os.environ["OPENAI_API_KEY"], os.environ["COHERE_API_KEY"] = "", ""

# set callbacks
litellm.success_callback = ["promptlayer"]

#openai call - log llm provider is openai
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}], metadata={"provider": "openai"})

#cohere call - log llm provider is cohere
response = completion(model="command-nightly", messages=[{"role": "user", "content": "Hi 👋 - i'm cohere"}], metadata={"provider": "cohere"})
```

Credits to [Nick Bradford](https://github.com/nsbradford), from [Vim-GPT](https://github.com/nsbradford/VimGPT), for the suggestion. 