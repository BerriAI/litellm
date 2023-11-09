<h1 align="center">
        🚅 LiteLLM
    </h1>
    <p align="center">
        <p align="center">Call all LLM APIs using the OpenAI format [Bedrock, Huggingface, Cohere, TogetherAI, Azure, OpenAI, etc.]
        <br>
    </p>
<h4 align="center"><a href="https://github.com/BerriAI/litellm/tree/main/litellm/proxy" target="_blank">Evaluate LLMs → OpenAI-Compatible Server</a></h4>
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

LiteLLM manages
- Translating inputs to the provider's `completion` and `embedding` endpoints
- Guarantees [consistent output](https://docs.litellm.ai/docs/completion/output), text responses will always be available at `['choices'][0]['message']['content']`
- Exception mapping - common exceptions across providers are mapped to the OpenAI exception types.

**10/05/2023:** LiteLLM is adopting Semantic Versioning for all commits. [Learn more](https://github.com/BerriAI/litellm/issues/532)  
**10/16/2023:** **Self-hosted OpenAI-proxy server** [Learn more](https://docs.litellm.ai/docs/simple_proxy)

# Usage ([**Docs**](https://docs.litellm.ai/docs/))

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_Getting_Started.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```
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

## Streaming ([Docs](https://docs.litellm.ai/docs/completion/stream))
liteLLM supports streaming the model response back, pass `stream=True` to get a streaming iterator in response.  
Streaming is supported for all models (Bedrock, Huggingface, TogetherAI, Azure, OpenAI, etc.)
```python
from litellm import completion
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for chunk in response:
    print(chunk['choices'][0]['delta'])

# claude 2
result = completion('claude-2', messages, stream=True)
for chunk in result:
  print(chunk['choices'][0]['delta'])
```

## Logging Observability ([Docs](https://docs.litellm.ai/docs/observability/callbacks))
LiteLLM exposes pre defined callbacks to send data to LLMonitor, Langfuse, Helicone, Promptlayer, Traceloop, Slack
```python
from litellm import completion

## set env variables for logging tools
os.environ["PROMPTLAYER_API_KEY"] = "your-promptlayer-key"
os.environ["LLMONITOR_APP_ID"] = "your-llmonitor-app-id"

os.environ["OPENAI_API_KEY"]

# set callbacks
litellm.success_callback = ["promptlayer", "llmonitor"] # log input/output to promptlayer, llmonitor, supabase

#openai call
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

## OpenAI Proxy - ([Docs](https://docs.litellm.ai/docs/simple_proxy))
**If you don't want to make code changes to add the litellm package to your code base**, you can use litellm proxy. Create a server to call 100+ LLMs (Huggingface/Bedrock/TogetherAI/etc) in the OpenAI ChatCompletions & Completions format

### Step 1: Start litellm proxy
```shell
$ litellm --model huggingface/bigcode/starcoder

#INFO: Proxy running on http://0.0.0.0:8000
```

### Step 2: Replace openai base
```python
import openai 

openai.api_base = "http://0.0.0.0:8000"

print(openai.ChatCompletion.create(model="test", messages=[{"role":"user", "content":"Hey!"}]))
```

## Supported Provider ([Docs](https://docs.litellm.ai/docs/providers))
| Provider      | [Completion](https://docs.litellm.ai/docs/#basic-usage) | [Streaming](https://docs.litellm.ai/docs/completion/stream#streaming-responses)  | [Async Completion](https://docs.litellm.ai/docs/completion/stream#async-completion)  | [Async Streaming](https://docs.litellm.ai/docs/completion/stream#async-streaming)  |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| [openai](https://docs.litellm.ai/docs/providers/openai)  | ✅ | ✅ | ✅ | ✅ |
| [azure](https://docs.litellm.ai/docs/providers/azure)  | ✅ | ✅ | ✅ | ✅ |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)  | ✅ | ✅ | ✅ | ✅ |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)  | ✅ | ✅ | ✅ | ✅ |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)  | ✅ | ✅ | ✅ | ✅ |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)  | ✅ | ✅ | ✅ | ✅ |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)  | ✅ | ✅ | ✅ | ✅ |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)  | ✅ | ✅ | ✅ | ✅ |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)  | ✅ | ✅ | ✅ | ✅ |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)  | ✅ | ✅ | ✅ | ✅ |
| [google - vertex_ai](https://docs.litellm.ai/docs/providers/vertex)  | ✅ | ✅ | ✅ | ✅ |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)  | ✅ | ✅ | ✅ | ✅ |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)  | ✅ | ✅ | ✅ | ✅ |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)  | ✅ | ✅ | ✅ | ✅ |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)  | ✅ | ✅ | ✅ | ✅ |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)  | ✅ | ✅ | ✅ | ✅ |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)  | ✅ | ✅ | ✅ | ✅ |
| [petals](https://docs.litellm.ai/docs/providers/petals)  | ✅ | ✅ | ✅ | ✅ |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)  | ✅ | ✅ | ✅ | ✅ |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)  | ✅ | ✅ | ✅ | ✅ |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)  | ✅ | ✅ | ✅ | ✅ |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)  | ✅ | ✅ | ✅ | ✅ |

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
pytest .
```

Step 4: Submit a PR with your changes! 🚀
- push your fork to your GitHub repo 
- submit a PR from there 

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

