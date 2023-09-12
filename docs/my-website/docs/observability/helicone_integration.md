# Helicone Tutorial 
[Helicone](https://helicone.ai/) is an open source observability platform that proxies your OpenAI traffic and provides you key insights into your spend, latency and usage.

## Use Helicone to log requests across all LLM Providers (OpenAI, Azure, Anthropic, Cohere, Replicate, PaLM)
liteLLM provides `success_callbacks` and `failure_callbacks`, making it easy for you to send data to a particular provider depending on the status of your responses. 

In this case, we want to log requests to Helicone when a request succeeds. 

### Approach 1: Use Callbacks 
Use just 1 line of code, to instantly log your responses **across all providers** with helicone: 
```python
litellm.success_callback=["helicone"]
```

Complete code
```python
from litellm import completion

## set env variables
os.environ["HELICONE_API_KEY"] = "your-helicone-key" 
os.environ["OPENAI_API_KEY"], os.environ["COHERE_API_KEY"] = "", ""

# set callbacks
litellm.success_callback=["helicone"]

#openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}]) 

#cohere call
response = completion(model="command-nightly", messages=[{"role": "user", "content": "Hi 👋 - i'm cohere"}]) 
```

### Approach 2: [OpenAI + Azure only] Use Helicone as a proxy
Helicone provides advanced functionality like caching, etc. Helicone currently supports this for Azure and OpenAI.

If you want to use Helicone to proxy your OpenAI/Azure requests, then you can - 

- Set helicone as your base url via: `litellm.api_url` 
- Pass in helicone request headers via: `litellm.headers` 

Complete Code
```python
import litellm
from litellm import completion

litellm.api_base = "https://oai.hconeai.com/v1"
litellm.headers = {"Helicone-Auth": f"Bearer {os.getenv('HELICONE_API_KEY')}"}

response = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "how does a court case get to the Supreme Court?"}]
)

print(response)
```
