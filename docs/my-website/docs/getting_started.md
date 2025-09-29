# Getting Started

import QuickStart from '../src/components/QuickStart.js'

LiteLLM simplifies LLM API calls by mapping them all to the [OpenAI ChatCompletion format](https://platform.openai.com/docs/api-reference/chat).

## basic usage

By default we provide a free $10 community-key to try all providers supported on LiteLLM.

```python
from litellm import completion

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["COHERE_API_KEY"] = "your-api-key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion("command-nightly", messages)
```

**Need a dedicated key?**
Email us @ krrish@berri.ai

Next Steps 👉 [Call all supported models - e.g. Claude-2, Llama2-70b, etc.](./proxy_api.md#supported-models)

More details 👉

- [Completion() function details](./completion/)
- [Overview of supported models / providers on LiteLLM](./providers/)
- [Search all models / providers](https://models.litellm.ai/)
- [Build your own OpenAI proxy](https://github.com/BerriAI/liteLLM-proxy/tree/main)

## streaming

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

- [streaming + async](./completion/stream.md)
- [tutorial for streaming Llama2 on TogetherAI](./tutorials/TogetherAI_liteLLM.md)

## exception handling

LiteLLM maps exceptions across all supported providers to the OpenAI exceptions. All our exceptions inherit from OpenAI's exception types, so any error-handling you have for that, should work out of the box with LiteLLM.

```python
from openai.error import OpenAIError
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "bad-key"
try:
    # some code
    completion(model="claude-instant-1", messages=[{"role": "user", "content": "Hey, how's it going?"}])
except OpenAIError as e:
    print(e)
```

## Logging Observability - Log LLM Input/Output ([Docs](https://docs.litellm.ai/docs/observability/callbacks))

LiteLLM exposes pre defined callbacks to send data to MLflow, Lunary, Langfuse, Helicone, Promptlayer, Traceloop, Slack

```python
from litellm import completion

## set env variables for logging tools (API key set up is not required when using MLflow)
os.environ["LUNARY_PUBLIC_KEY"] = "your-lunary-public-key" # get your public key at https://app.lunary.ai/settings
os.environ["HELICONE_API_KEY"] = "your-helicone-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

os.environ["OPENAI_API_KEY"]

# set callbacks
litellm.success_callback = ["lunary", "mlflow", "langfuse", "helicone"] # log input/output to MLflow, langfuse, lunary, helicone

#openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

More details 👉

- [exception mapping](./exception_mapping.md)
- [retries + model fallbacks for completion()](./completion/reliable_completions.md)
- [tutorial for model fallbacks with completion()](./tutorials/fallbacks.md)
