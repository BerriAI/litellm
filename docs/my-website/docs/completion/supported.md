# Supported Providers + Models

## API Keys 
liteLLM reads key naming, all keys should be named in the following format:
`<PROVIDER>_API_KEY` for example
* `OPENAI_API_KEY`      Provider = OpenAI
* `TOGETHERAI_API_KEY`  Provider = TogetherAI
* `HUGGINGFACE_API_KEY` Provider = HuggingFace


### OpenAI Chat Completion Models

| Model Name       | Function Call                          | Required OS Variables                |
|------------------|----------------------------------------|--------------------------------------|
| gpt-3.5-turbo    | `completion('gpt-3.5-turbo', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-0301    | `completion('gpt-3.5-turbo-0301', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-0613    | `completion('gpt-3.5-turbo-0613', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-16k    | `completion('gpt-3.5-turbo-16k', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-3.5-turbo-16k-0613    | `completion('gpt-3.5-turbo-16k-0613', messages)` | `os.environ['OPENAI_API_KEY']`       |
| gpt-4            | `completion('gpt-4', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-0314            | `completion('gpt-4-0314', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-0613            | `completion('gpt-4-0613', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-32k            | `completion('gpt-4-32k', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-32k-0314            | `completion('gpt-4-32k-0314', messages)`         | `os.environ['OPENAI_API_KEY']`       |
| gpt-4-32k-0613            | `completion('gpt-4-32k-0613', messages)`         | `os.environ['OPENAI_API_KEY']`       |

These also support the `OPENAI_API_BASE` environment variable, which can be used to specify a custom API endpoint.

### Azure OpenAI Chat Completion Models

| Model Name       | Function Call                           | Required OS Variables                     |
|------------------|-----------------------------------------|-------------------------------------------|
| gpt-3.5-turbo    | `completion('gpt-3.5-turbo', messages, custom_llm_provider="azure")` | `os.environ['AZURE_API_KEY']`,`os.environ['AZURE_API_BASE']`,`os.environ['AZURE_API_VERSION']` |
| gpt-4            | `completion('gpt-4', messages, custom_llm_provider="azure")`         | `os.environ['AZURE_API_KEY']`,`os.environ['AZURE_API_BASE']`,`os.environ['AZURE_API_VERSION']` |

### OpenAI Text Completion Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| text-davinci-003 | `completion('text-davinci-003', messages)` | `os.environ['OPENAI_API_KEY']`       |
| ada-001 | `completion('ada-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| curie-001 | `completion('curie-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| babbage-001 | `completion('babbage-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| babbage-002 | `completion('ada-001', messages)` | `os.environ['OPENAI_API_KEY']`       |
| davinci-002 | `completion('davinci-002', messages)` | `os.environ['OPENAI_API_KEY']`       |


### Google VertexAI Models
Sample notebook for calling VertexAI models: https://github.com/BerriAI/litellm/blob/main/cookbook/liteLLM_VertextAI_Example.ipynb

All calls using Vertex AI require the following parameters:
* Your Project ID
`litellm.vertex_project` = "hardy-device-38811" Your Project ID
* Your Project Location
`litellm.vertex_location` = "us-central1" 

Authentication:
VertexAI uses Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information on setting this up

VertexAI requires you to set `application_default_credentials.json`, this can be set by running `gcloud auth application-default login` in your terminal

| Model Name       | Function Call                                            |
|------------------|----------------------------------------------------------|
| chat-bison       | `completion('chat-bison', messages)`                    |
| chat-bison@001   | `completion('chat-bison@001', messages)`                |
| text-bison       | `completion('text-bison', messages)`                    |
| text-bison@001   | `completion('text-bison@001', messages)`                |

### Anthropic Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| claude-instant-1  | `completion('claude-instant-1', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-instant-1.2  | `completion('claude-instant-1.2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-2  | `completion('claude-2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |

### Hugging Face Inference API

All [`text2text-generation`](https://huggingface.co/models?library=transformers&pipeline_tag=text2text-generation&sort=downloads) and [`text-generation`](https://huggingface.co/models?library=transformers&pipeline_tag=text-generation&sort=downloads) models are supported by liteLLM. You can use any text model from Hugging Face with the following steps:

* Copy the `model repo` URL from Hugging Face and set it as the `model` parameter in the completion call.
* Set `hugging_face` parameter to `True`.
* Make sure to set the hugging face API key

Here are some examples of supported models:
**Note that the models mentioned in the table are examples, and you can use any text model available on Hugging Face by following the steps above.**

| Model Name       | Function Call                                                                       | Required OS Variables                |
|------------------|-------------------------------------------------------------------------------------|--------------------------------------|
| [stabilityai/stablecode-completion-alpha-3b-4k](https://huggingface.co/stabilityai/stablecode-completion-alpha-3b-4k)  | `completion(model="stabilityai/stablecode-completion-alpha-3b-4k", messages=messages, custom_llm_provider="huggingface")` | `os.environ['HUGGINGFACE_API_KEY']`       |
| [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)                           | `completion(model="bigcode/starcoder", messages=messages, custom_llm_provider="huggingface")`          | `os.environ['HUGGINGFACE_API_KEY']`       |
| [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)                         | `completion(model="google/flan-t5-xxl", messages=messages, custom_llm_provider="huggingface")`         | `os.environ['HUGGINGFACE_API_KEY']`       |
| [google/flan-t5-large](https://huggingface.co/google/flan-t5-large)                     | `completion(model="google/flan-t5-large", messages=messages, custom_llm_provider="huggingface")`       | `os.environ['HUGGINGFACE_API_KEY']`       |

### Replicate Models
liteLLM supports all replicate LLMs. For replicate models ensure to add a `replicate` prefix to the `model` arg. liteLLM detects it using this arg. 
Below are examples on how to call replicate LLMs using liteLLM 

Model Name                  | Function Call                                                  | Required OS Variables                |
-----------------------------|----------------------------------------------------------------|--------------------------------------|
 replicate/llama-2-70b-chat | `completion('replicate/replicate/llama-2-70b-chat', messages)` | `os.environ['REPLICATE_API_KEY']`    |
 a16z-infra/llama-2-13b-chat| `completion('replicate/a16z-infra/llama-2-13b-chat', messages)`| `os.environ['REPLICATE_API_KEY']`    |
 joehoover/instructblip-vicuna13b | `completion('replicate/joehoover/instructblip-vicuna13b', messages)` | `os.environ['REPLICATE_API_KEY']` |
 replicate/dolly-v2-12b     | `completion('replicate/replicate/dolly-v2-12b', messages)`     | `os.environ['REPLICATE_API_KEY']`    |
 a16z-infra/llama-2-7b-chat | `completion('replicate/a16z-infra/llama-2-7b-chat', messages)` | `os.environ['REPLICATE_API_KEY']`    |
 replicate/vicuna-13b       | `completion('replicate/replicate/vicuna-13b', messages)`       | `os.environ['REPLICATE_API_KEY']`    |
 daanelson/flan-t5-large    | `completion('replicate/daanelson/flan-t5-large', messages)`    | `os.environ['REPLICATE_API_KEY']`    |
 replit/replit-code-v1-3b   | `completion('replicate/replit/replit-code-v1-3b', messages)`   | `os.environ['REPLICATE_API_KEY']`    |

### AI21 Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| j2-light         | `completion('j2-light', messages)`         | `os.environ['AI21_API_KEY']`         |
| j2-mid           | `completion('j2-mid', messages)`           | `os.environ['AI21_API_KEY']`         |
| j2-ultra         | `completion('j2-ultra', messages)`         | `os.environ['AI21_API_KEY']`         |

### Cohere Models

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| command-nightly  | `completion('command-nightly', messages)`  | `os.environ['COHERE_API_KEY']`       |

### Together AI Models
liteLLM supports `non-streaming` and `streaming` requests to all models on https://api.together.xyz/

Example TogetherAI Usage - Note: liteLLM supports all models deployed on TogetherAI

| Model Name                        | Function Call                                                          | Required OS Variables           |
|-----------------------------------|------------------------------------------------------------------------|---------------------------------|
| togethercomputer/llama-2-70b-chat  | `completion('togethercomputer/llama-2-70b-chat', messages)`            | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/LLaMA-2-13b-chat  | `completion('togethercomputer/LLaMA-2-13b-chat', messages)`            | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/code-and-talk-v1 | `completion('togethercomputer/code-and-talk-v1', messages)`           | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/creative-v1      | `completion('togethercomputer/creative-v1', messages)`                | `os.environ['TOGETHERAI_API_KEY']` |
| togethercomputer/yourmodel        | `completion('togethercomputer/yourmodel', messages)`                  | `os.environ['TOGETHERAI_API_KEY']` |


### Baseten Models
Baseten provides infrastructure to deploy and serve ML models https://www.baseten.co/. Use liteLLM to easily call models deployed on Baseten.

Example Baseten Usage - Note: liteLLM supports all models deployed on Basten

| Model Name       | Function Call                                  | Required OS Variables              |
|------------------|--------------------------------------------|------------------------------------|
| Falcon 7B        | `completion(model='baseten/qvv0xeq', messages=messages)`         | `os.environ['BASETEN_API_KEY']`     |
| Wizard LM        | `completion(model='baseten/q841o8w', messages=messages)`         | `os.environ['BASETEN_API_KEY']`     |
| MPT 7B Base      | `completion(model='baseten/31dxrj3', messages=messages)`         | `os.environ['BASETEN_API_KEY']`     |


### OpenRouter Completion Models

All the text models from [OpenRouter](https://openrouter.ai/docs) are supported by liteLLM.

| Model Name       | Function Call                              | Required OS Variables                |
|------------------|--------------------------------------------|--------------------------------------|
| openai/gpt-3.5-turbo | `completion('openai/gpt-3.5-turbo', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| openai/gpt-3.5-turbo-16k | `completion('openai/gpt-3.5-turbo-16k', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| openai/gpt-4 | `completion('openai/gpt-4', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| openai/gpt-4-32k | `completion('openai/gpt-4-32k', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| anthropic/claude-2 | `completion('anthropic/claude-2', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| anthropic/claude-instant-v1 | `completion('anthropic/claude-instant-v1', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| google/palm-2-chat-bison | `completion('google/palm-2-chat-bison', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| google/palm-2-codechat-bison | `completion('google/palm-2-codechat-bison', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| meta-llama/llama-2-13b-chat | `completion('meta-llama/llama-2-13b-chat', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |
| meta-llama/llama-2-70b-chat | `completion('meta-llama/llama-2-70b-chat', messages)` | `os.environ['OR_SITE_URL']`,`os.environ['OR_APP_NAME']`,`os.environ['OPENROUTER_API_KEY']`       |

### Petals Models
Supported models on https://chat.petals.dev/

| Model Name           | Function Call                                                          | Required OS Variables          |
|----------------------|------------------------------------------------------------------------|--------------------------------|
| stabilityai/StableBeluga2 | `completion(model='stabilityai/StableBeluga2', messages, custom_llm_provider="petals")` | No API Key required          |
| enoch/llama-65b-hf   | `completion(model='enoch/llama-65b-hf', messages, custom_llm_provider="petals")` | No API Key required          |
| bigscience/bloomz    | `completion(model='bigscience/bloomz', messages, custom_llm_provider="petals")` | No API Key required          |

### Ollama Models
Ollama supported models: https://github.com/jmorganca/ollama

| Model Name           | Function Call                                                                     | Required OS Variables          |
|----------------------|-----------------------------------------------------------------------------------|--------------------------------|
| Llama2 7B            | `completion(model='llama2', messages, custom_api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Llama2 13B           | `completion(model='llama2:13b', messages, custom_api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Llama2 70B           | `completion(model='llama2:70b', messages, custom_api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Llama2 Uncensored    | `completion(model='llama2-uncensored', messages, custom_api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Orca Mini            | `completion(model='orca-mini', messages, custom_api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Vicuna               | `completion(model='vicuna', messages, custom_api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Nous-Hermes          | `completion(model='nous-hermes', messages, custom_api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Nous-Hermes 13B      | `completion(model='nous-hermes:13b', messages, custom_api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
| Wizard Vicuna Uncensored | `completion(model='wizard-vicuna', messages, custom_api_base="http://localhost:11434", custom_llm_provider="ollama", stream=True)` | No API Key required |
