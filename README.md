<h1 align="center">
        🚅 LiteLLM
    </h1>
    <p align="center">
        <p align="center">Call all LLM APIs using the OpenAI format [Bedrock, Huggingface, VertexAI, TogetherAI, Azure, OpenAI, etc.]
        <br>
    </p>
<h4 align="center"><a href="https://docs.litellm.ai/docs/simple_proxy" target="_blank">OpenAI Proxy Server</a> | <a href="https://docs.litellm.ai/docs/enterprise"target="_blank">Enterprise Tier</a></h4>
<h4 align="center">
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
    </a>
    <a href="https://dl.circleci.com/status-badge/redirect/gh/BerriAI/litellm/tree/main" target="_blank">
        <img src="https://dl.circleci.com/status-badge/img/gh/BerriAI/litellm/tree/main.svg?style=svg" alt="CircleCI">
    </a>
    <a href="https://www.ycombinator.com/companies/berriai">
        <img src="https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square" alt="Y Combinator W23">
    </a>
    <a href="https://wa.link/huol9n">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square" alt="Whatsapp">
    </a>
    <a href="https://discord.gg/wuPM9dRgDw">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
</h4>

LiteLLM manages:
- Translate inputs to provider's `completion`, `embedding`, and `image_generation` endpoints
- [Consistent output](https://docs.litellm.ai/docs/completion/output), text responses will always be available at `['choices'][0]['message']['content']`
- Retry/fallback logic across multiple deployments (e.g. Azure/OpenAI) - [Router](https://docs.litellm.ai/docs/routing)
- Set Budgets & Rate limits per project, api key, model [OpenAI Proxy Server](https://docs.litellm.ai/docs/simple_proxy)


[**Jump to OpenAI Proxy Docs**](https://github.com/BerriAI/litellm?tab=readme-ov-file#openai-proxy---docs) <br>
[**Jump to Supported LLM Providers**](https://github.com/BerriAI/litellm?tab=readme-ov-file#supported-provider-docs)

Support for more providers. Missing a provider or LLM Platform, raise a [feature request](https://github.com/BerriAI/litellm/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+).

# Usage ([**Docs**](https://docs.litellm.ai/docs/))
> [!IMPORTANT]
> LiteLLM v1.0.0 now requires `openai>=1.0.0`. Migration guide [here](https://docs.litellm.ai/docs/migration)


<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_Getting_Started.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


```shell
pip install litellm
```

```python
from litellm import completion
import os

## set ENV variables 
os.environ["OPENAI_API_KEY"] = "your-openai-key" 
os.environ["COHERE_API_KEY"] = "your-cohere-key" 

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="gpt-3.5-turbo", messages=messages)

# cohere call
response = completion(model="command-nightly", messages=messages)
print(response)
```

Call any model supported by a provider, with `model=<provider_name>/<model_name>`. There might be provider-specific details here, so refer to [provider docs for more information](https://docs.litellm.ai/docs/providers)

## Async ([Docs](https://docs.litellm.ai/docs/completion/stream#async-completion))

```python
from litellm import acompletion
import asyncio

async def test_get_response():
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    response = await acompletion(model="gpt-3.5-turbo", messages=messages)
    return response

response = asyncio.run(test_get_response())
print(response)
```

## Streaming ([Docs](https://docs.litellm.ai/docs/completion/stream))
liteLLM supports streaming the model response back, pass `stream=True` to get a streaming iterator in response.  
Streaming is supported for all models (Bedrock, Huggingface, TogetherAI, Azure, OpenAI, etc.)
```python
from litellm import completion
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "")

# claude 2
response = completion('claude-2', messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "")
```

## Logging Observability ([Docs](https://docs.litellm.ai/docs/observability/callbacks))
LiteLLM exposes pre defined callbacks to send data to Langfuse, DynamoDB, s3 Buckets, LLMonitor, Helicone, Promptlayer, Traceloop, Athina, Slack
```python
from litellm import completion

## set env variables for logging tools
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LLMONITOR_APP_ID"] = "your-llmonitor-app-id"
os.environ["ATHINA_API_KEY"] = "your-athina-api-key"

os.environ["OPENAI_API_KEY"]

# set callbacks
litellm.success_callback = ["langfuse", "llmonitor", "athina"] # log input/output to langfuse, llmonitor, supabase, athina etc

#openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

# OpenAI Proxy - ([Docs](https://docs.litellm.ai/docs/simple_proxy))

Set Budgets & Rate limits across multiple projects

The proxy provides: 
1. [Hooks for auth](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth)
2. [Hooks for logging](https://docs.litellm.ai/docs/proxy/logging#step-1---create-your-custom-litellm-callback-class)
3. [Cost tracking](https://docs.litellm.ai/docs/proxy/virtual_keys#tracking-spend)
4. [Rate Limiting](https://docs.litellm.ai/docs/proxy/users#set-rate-limits)

## 📖 Proxy Endpoints - [Swagger Docs](https://litellm-api.up.railway.app/)

## Quick Start Proxy - CLI 

```shell
pip install 'litellm[proxy]'
```

### Step 1: Start litellm proxy
```shell
$ litellm --model huggingface/bigcode/starcoder

#INFO: Proxy running on http://0.0.0.0:4000
```

### Step 2: Make ChatCompletions Request to Proxy
```python
import openai # openai v1.0.0+
client = openai.OpenAI(api_key="anything",base_url="http://0.0.0.0:4000") # set proxy to base_url
# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)
```

## Proxy Key Management ([Docs](https://docs.litellm.ai/docs/proxy/virtual_keys))
UI on `/ui` on your proxy server 
![ui_3](https://github.com/BerriAI/litellm/assets/29436595/47c97d5e-b9be-4839-b28c-43d7f4f10033)

Set budgets and rate limits across multiple projects
`POST /key/generate`

### Request
```shell
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4", "claude-2"], "duration": "20m","metadata": {"user": "ishaan@berri.ai", "team": "core-infra"}}'
```

### Expected Response
```shell
{
    "key": "sk-kdEXbIqZRwEeEiHwdg7sFA", # Bearer token
    "expires": "2023-11-19T01:38:25.838000+00:00" # datetime object
}
```

## Supported Providers ([Docs](https://docs.litellm.ai/docs/providers))
| Provider      | [Completion](https://docs.litellm.ai/docs/#basic-usage) | [Streaming](https://docs.litellm.ai/docs/completion/stream#streaming-responses)  | [Async Completion](https://docs.litellm.ai/docs/completion/stream#async-completion)  | [Async Streaming](https://docs.litellm.ai/docs/completion/stream#async-streaming)  | [Async Embedding](https://docs.litellm.ai/docs/embedding/supported_embedding)  | [Async Image Generation](https://docs.litellm.ai/docs/image_generation)  | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [openai](https://docs.litellm.ai/docs/providers/openai)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [azure](https://docs.litellm.ai/docs/providers/azure)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)  | ✅ | ✅ | ✅ | ✅ | ✅ |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)  | ✅ | ✅ | ✅ | ✅ |✅ |
| [google - vertex_ai [Gemini]](https://docs.litellm.ai/docs/providers/vertex)  | ✅ | ✅ | ✅ | ✅ |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)  | ✅ | ✅ | ✅ | ✅ |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini)  | ✅ |  | ✅ |  | |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral)  | ✅ | ✅ | ✅ | ✅ | ✅ |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers)  | ✅ | ✅ | ✅ | ✅ |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)  | ✅ | ✅ | ✅ | ✅ | ✅ |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)  | ✅ | ✅ | ✅ | ✅ |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)  | ✅ | ✅ | ✅ | ✅ | ✅ |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)  | ✅ | ✅ | ✅ | ✅ |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)  | ✅ | ✅ | ✅ | ✅ |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)  | ✅ | ✅ | ✅ | ✅ |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)  | ✅ | ✅ | ✅ | ✅ |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)  | ✅ | ✅ | ✅ | ✅ |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)  | ✅ | ✅ | ✅ | ✅ |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)  | ✅ | ✅ | ✅ | ✅ |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)  | ✅ | ✅ | ✅ | ✅ |
| [petals](https://docs.litellm.ai/docs/providers/petals)  | ✅ | ✅ | ✅ | ✅ |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)  | ✅ | ✅ | ✅ | ✅ |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)  | ✅ | ✅ | ✅ | ✅ |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)  | ✅ | ✅ | ✅ | ✅ |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq)  | ✅ | ✅ | ✅ | ✅ |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)  | ✅ | ✅ | ✅ | ✅ |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage)  |  |  |  |  | ✅ |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference)  |  |  |  |  | ✅ |


[**Read the Docs**](https://docs.litellm.ai/docs/)

## Contributing
To contribute: Clone the repo locally -> Make a change -> Submit a PR with the change. 

Here's how to modify the repo locally: 
Step 1: Clone the repo 
```
git clone https://github.com/BerriAI/litellm.git
```

Step 2: Navigate into the project, and install dependencies: 
```
cd litellm
poetry install
```

Step 3: Test your change:
```
cd litellm/tests # pwd: Documents/litellm/litellm/tests
poetry run flake8
poetry run pytest .
```

Step 4: Submit a PR with your changes! 🚀
- push your fork to your GitHub repo 
- submit a PR from there 

# Enterprise
For companies that need better security, user management and professional support

[Talk to founders](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

This covers: 
- ✅ **Features under the [LiteLLM Commercial License](https://docs.litellm.ai/docs/proxy/enterprise):**
- ✅ **Feature Prioritization**
- ✅ **Custom Integrations**
- ✅ **Professional Support - Dedicated discord + slack**
- ✅ **Custom SLAs**
- ✅ **Secure access with Single Sign-On**

# Support / talk with founders
- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

# Why did we build this 
- **Need for simplicity**: Our code started to get extremely complicated managing & translating calls between Azure, OpenAI and Cohere.

# Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/BerriAI/litellm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BerriAI/litellm" />
</a>

