# LangFuse Tutorial

LangFuse is open Source Observability & Analytics for LLM Apps
Detailed production traces and a granular view on quality, cost and latency

<Image img={require('../../img/langfuse.gif')} />

## Use Langfuse to log requests across all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)

liteLLM provides `callbacks`, making it easy for you to log data depending on the status of your responses.

### Using Callbacks

Get your Langfuse API Keys from https://cloud.langfuse.com/

Use just 2 lines of code, to instantly log your responses **across all providers** with langfuse:

```python
litellm.success_callback = ["langfuse"]

```

Complete code

```python
from litellm import completion

## set env variables
os.environ["LANGFUSE_PUBLIC_KEY"] = "your key"
os.environ["LANGFUSE_SECRET_KEY"] = "your key"

os.environ["OPENAI_API_KEY"], os.environ["COHERE_API_KEY"] = "", ""

# set callbacks
litellm.success_callback = ["langfuse"]

#openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])

#cohere call
response = completion(model="command-nightly", messages=[{"role": "user", "content": "Hi 👋 - i'm cohere"}])
```
