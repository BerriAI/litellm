<h1 align="center">
        🚅 LiteLLM
    </h1>
    <p align="center">
        <p align="center">Call all LLM APIs using the OpenAI format [Anthropic, Huggingface, Cohere, TogetherAI, Azure, OpenAI, etc.]
        <br>
        <br>
        <a href="https://github.com/BerriAI/litellm/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml&title=%5BBug%5D%3A+">Bug Report</a>
        ·
        <a href="https://github.com/BerriAI/litellm/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+">Feature Request</a>
    </p>
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
</h4>
<table align="center" style="border:0; padding-top: 20px; border-collapse: separate; border-spacing: 10px;">
<tr>
    <td style="border:none;"><a href="https://docs.litellm.ai/docs/" target="_blank">Docs</a></td>
    <td style="border:none;"><a href="https://docs.litellm.ai/docs/providers" target="_blank">100+ Supported Models</a></td>
    <td style="border:none;"><a href="https://www.loom.com/share/bd4e9029a1da4fd4a3e40e41c87ebb3a?sid=d6c82c80-1e57-4519-a9df-0c5cc8c4c40d" target="_blank" rel="noopener noreferrer">Demo Video</a></td>
        
</tr>
</table>

LiteLLM manages
- Translating inputs to the provider's completion and embedding endpoints
- Guarantees [consistent output](https://docs.litellm.ai/docs/completion/output), text responses will always be available at `['choices'][0]['message']['content']`
- Exception mapping - common exceptions across providers are mapped to the OpenAI exception types

**🚨 Seeing errors?** [![Chat on WhatsApp](https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square)](https://wa.link/huol9n) [![Chat on Discord](https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square)](https://discord.gg/wuPM9dRgDw) 

**05/10/2023:** LiteLLM is adopting Semantic Versioning for all commits. [Learn more](https://github.com/BerriAI/litellm/issues/532)

# Usage

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_OpenAI.ipynb">
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
Streaming is supported for OpenAI, Azure, Anthropic, Huggingface models
```python
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for chunk in response:
    print(chunk['choices'][0]['delta'])

# claude 2
result = completion('claude-2', messages, stream=True)
for chunk in result:
  print(chunk['choices'][0]['delta'])
```


## Caching ([Docs](https://docs.litellm.ai/docs/caching/))

LiteLLM supports caching `completion()` and `embedding()` calls for all LLMs. [Hosted Cache LiteLLM API](https://docs.litellm.ai/docs/caching/caching_api)
```python
import litellm
from litellm.caching import Cache
import os

litellm.cache = Cache()
os.environ['OPENAI_API_KEY'] = ""
# add to cache
response1 = litellm.completion(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "why is LiteLLM amazing?"}], 
    caching=True
)
# returns cached response
response2 = litellm.completion(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": "why is LiteLLM amazing?"}], 
    caching=True
)

print(f"response1: {response1}")
print(f"response2: {response2}")

```

## OpenAI Proxy Server ([Docs](https://docs.litellm.ai/docs/proxy_server))
Spin up a local server to translate openai api calls to any non-openai model (e.g. Huggingface, TogetherAI, Ollama, etc.)

This works for async + streaming as well. 
```python
litellm --model <model_name>
```
Running your model locally or on a custom endpoint ? Set the `--api-base` parameter [see how](https://docs.litellm.ai/docs/proxy_server)

## Supported Provider ([Docs](https://docs.litellm.ai/docs/providers))
| Provider      | [Completion](https://docs.litellm.ai/docs/#basic-usage) | [Streaming](https://docs.litellm.ai/docs/completion/stream#streaming-responses)  | [Async Completion](https://docs.litellm.ai/docs/completion/stream#async-completion)  | [Async Streaming](https://docs.litellm.ai/docs/completion/stream#async-streaming)  |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| [openai](https://docs.litellm.ai/docs/providers/openai)  | ✅ | ✅ | ✅ | ✅ |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)  | ✅ | ✅ | ✅ | ✅ |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)  | ✅ | ✅ | ✅ | ✅ |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)  | ✅ | ✅ | ✅ | ✅ |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)  | ✅ | ✅ | ✅ | ✅ |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)  | ✅ | ✅ | ✅ | ✅ |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)  | ✅ | ✅ | ✅ | ✅ |
| [vertex_ai](https://docs.litellm.ai/docs/providers/vertex)  | ✅ | ✅ | ✅ | ✅ |
| [palm](https://docs.litellm.ai/docs/providers/palm)  | ✅ | ✅ | ✅ | ✅ |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)  | ✅ | ✅ | ✅ | ✅ |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)  | ✅ | ✅ | ✅ | ✅ |
| [azure](https://docs.litellm.ai/docs/providers/azure)  | ✅ | ✅ | ✅ | ✅ |
| [sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)  | ✅ | ✅ | ✅ | ✅ |
| [bedrock](https://docs.litellm.ai/docs/providers/bedrock)  | ✅ | ✅ | ✅ | ✅ |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)  | ✅ | ✅ | ✅ | ✅ |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)  | ✅ | ✅ | ✅ | ✅ |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)  | ✅ | ✅ | ✅ | ✅ |
| [petals](https://docs.litellm.ai/docs/providers/petals)  | ✅ | ✅ | ✅ | ✅ |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)  | ✅ | ✅ | ✅ | ✅ |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)  | ✅ | ✅ | ✅ | ✅ |

[**Read the Docs**](https://docs.litellm.ai/docs/)
# Contributing
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

[Learn more on how to make a PR](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)


# Support / talk with founders
- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

# Why did we build this 
- **Need for simplicity**: Our code started to get extremely complicated managing & translating calls between Azure, OpenAI, Cohere

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

