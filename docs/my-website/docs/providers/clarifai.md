# Clarifai
Anthropic, OpenAI, Mistral, Llama and Gemini LLMs are Supported on Clarifai. 

:::warning 

Streaming is not yet supported on using clarifai and litellm. Tracking support here: https://github.com/BerriAI/litellm/issues/4162

:::

## Pre-Requisites
`pip install litellm`

## Required Environment Variables
To obtain your Clarifai Personal access token follow this [link](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/). Optionally the PAT can also be passed in `completion` function.

```python
os.environ["CLARIFAI_API_KEY"] = "YOUR_CLARIFAI_PAT"  # CLARIFAI_PAT

```

## Usage

```python
import os
from litellm import completion

os.environ["CLARIFAI_API_KEY"] = ""

response = completion(
  model="clarifai/mistralai.completion.mistral-large",
  messages=[{ "content": "Tell me a joke about physics?","role": "user"}]
)
```

**Output**
```json
{
    "id": "chatcmpl-572701ee-9ab2-411c-ac75-46c1ba18e781",
    "choices": [
      {
        "finish_reason": "stop",
        "index": 1,
        "message": {
          "content": "Sure, here's a physics joke for you:\n\nWhy can't you trust an atom?\n\nBecause they make up everything!",
          "role": "assistant"
        }
      }
    ],
    "created": 1714410197,
    "model": "https://api.clarifai.com/v2/users/mistralai/apps/completion/models/mistral-large/outputs",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {
      "prompt_tokens": 14,
      "completion_tokens": 24,
      "total_tokens": 38
    }
  }
```

## Clarifai models
liteLLM supports all models on [Clarifai community](https://clarifai.com/explore/models?filterData=%5B%7B%22field%22%3A%22use_cases%22%2C%22value%22%3A%5B%22llm%22%5D%7D%5D&page=1&perPage=24)

Example  Usage - Note: liteLLM supports all models deployed on Clarifai

## Llama LLMs
| Model Name                        | Function Call |
---------------------------|---------------------------------|
| clarifai/meta.Llama-2.llama2-7b-chat    | `completion('clarifai/meta.Llama-2.llama2-7b-chat', messages)`
| clarifai/meta.Llama-2.llama2-13b-chat   | `completion('clarifai/meta.Llama-2.llama2-13b-chat', messages)`
| clarifai/meta.Llama-2.llama2-70b-chat   | `completion('clarifai/meta.Llama-2.llama2-70b-chat', messages)` |
| clarifai/meta.Llama-2.codeLlama-70b-Python   | `completion('clarifai/meta.Llama-2.codeLlama-70b-Python', messages)`| 
| clarifai/meta.Llama-2.codeLlama-70b-Instruct | `completion('clarifai/meta.Llama-2.codeLlama-70b-Instruct', messages)` |   

## Mistral LLMs
| Model Name                                  | Function Call                                                         |
|---------------------------------------------|------------------------------------------------------------------------|
| clarifai/mistralai.completion.mixtral-8x22B            | `completion('clarifai/mistralai.completion.mixtral-8x22B', messages)`               |
| clarifai/mistralai.completion.mistral-large           | `completion('clarifai/mistralai.completion.mistral-large', messages)`              |
| clarifai/mistralai.completion.mistral-medium          | `completion('clarifai/mistralai.completion.mistral-medium', messages)`             |
| clarifai/mistralai.completion.mistral-small           | `completion('clarifai/mistralai.completion.mistral-small', messages)`              |
| clarifai/mistralai.completion.mixtral-8x7B-Instruct-v0_1 | `completion('clarifai/mistralai.completion.mixtral-8x7B-Instruct-v0_1', messages)`
| clarifai/mistralai.completion.mistral-7B-OpenOrca  | `completion('clarifai/mistralai.completion.mistral-7B-OpenOrca', messages)`          |
| clarifai/mistralai.completion.openHermes-2-mistral-7B | `completion('clarifai/mistralai.completion.openHermes-2-mistral-7B', messages)`      |


## Jurassic LLMs 
| Model Name                                    | Function Call                                                      |
|-----------------------------------------------|---------------------------------------------------------------------|
| clarifai/ai21.complete.Jurassic2-Grande       | `completion('clarifai/ai21.complete.Jurassic2-Grande', messages)`       |
| clarifai/ai21.complete.Jurassic2-Grande-Instruct | `completion('clarifai/ai21.complete.Jurassic2-Grande-Instruct', messages)` |
| clarifai/ai21.complete.Jurassic2-Jumbo-Instruct  | `completion('clarifai/ai21.complete.Jurassic2-Jumbo-Instruct', messages)`  |
| clarifai/ai21.complete.Jurassic2-Jumbo         | `completion('clarifai/ai21.complete.Jurassic2-Jumbo', messages)`          |
| clarifai/ai21.complete.Jurassic2-Large         | `completion('clarifai/ai21.complete.Jurassic2-Large', messages)`          |

## Wizard LLMs

| Model Name                                    | Function Call                                                      |
|-----------------------------------------------|---------------------------------------------------------------------|
| clarifai/wizardlm.generate.wizardCoder-Python-34B | `completion('clarifai/wizardlm.generate.wizardCoder-Python-34B', messages)`    |
| clarifai/wizardlm.generate.wizardLM-70B          | `completion('clarifai/wizardlm.generate.wizardLM-70B', messages)`             | 
| clarifai/wizardlm.generate.wizardLM-13B          | `completion('clarifai/wizardlm.generate.wizardLM-13B', messages)`           |
| clarifai/wizardlm.generate.wizardCoder-15B       | `completion('clarifai/wizardlm.generate.wizardCoder-15B', messages)`          |

## Anthropic models

| Model Name                                    | Function Call                                                      |
|-----------------------------------------------|---------------------------------------------------------------------|
| clarifai/anthropic.completion.claude-v1       | `completion('clarifai/anthropic.completion.claude-v1', messages)`       |
| clarifai/anthropic.completion.claude-instant-1_2 | `completion('clarifai/anthropic.completion.claude-instant-1_2', messages)` |
| clarifai/anthropic.completion.claude-instant  | `completion('clarifai/anthropic.completion.claude-instant', messages)`  |
| clarifai/anthropic.completion.claude-v2       | `completion('clarifai/anthropic.completion.claude-v2', messages)`       |
| clarifai/anthropic.completion.claude-2_1      | `completion('clarifai/anthropic.completion.claude-2_1', messages)`      |
| clarifai/anthropic.completion.claude-3-opus   | `completion('clarifai/anthropic.completion.claude-3-opus', messages)`   |
| clarifai/anthropic.completion.claude-3-sonnet | `completion('clarifai/anthropic.completion.claude-3-sonnet', messages)` |

## OpenAI GPT LLMs

| Model Name                                    | Function Call                                                      |
|-----------------------------------------------|---------------------------------------------------------------------|
| clarifai/openai.chat-completion.GPT-4         | `completion('clarifai/openai.chat-completion.GPT-4', messages)`          |
| clarifai/openai.chat-completion.GPT-3_5-turbo | `completion('clarifai/openai.chat-completion.GPT-3_5-turbo', messages)`  |
| clarifai/openai.chat-completion.gpt-4-turbo   | `completion('clarifai/openai.chat-completion.gpt-4-turbo', messages)`    |
| clarifai/openai.completion.gpt-3_5-turbo-instruct | `completion('clarifai/openai.completion.gpt-3_5-turbo-instruct', messages)` |

## GCP LLMs

| Model Name                                    | Function Call                                                      |
|-----------------------------------------------|---------------------------------------------------------------------|
| clarifai/gcp.generate.gemini-1_5-pro         | `completion('clarifai/gcp.generate.gemini-1_5-pro', messages)`          |
| clarifai/gcp.generate.imagen-2               | `completion('clarifai/gcp.generate.imagen-2', messages)`                |
| clarifai/gcp.generate.code-gecko             | `completion('clarifai/gcp.generate.code-gecko', messages)`              |
| clarifai/gcp.generate.code-bison             | `completion('clarifai/gcp.generate.code-bison', messages)`              |
| clarifai/gcp.generate.text-bison            | `completion('clarifai/gcp.generate.text-bison', messages)`               |
| clarifai/gcp.generate.gemma-2b-it            | `completion('clarifai/gcp.generate.gemma-2b-it', messages)`              |
| clarifai/gcp.generate.gemma-7b-it            | `completion('clarifai/gcp.generate.gemma-7b-it', messages)`              |
| clarifai/gcp.generate.gemini-pro            | `completion('clarifai/gcp.generate.gemini-pro', messages)`               |
| clarifai/gcp.generate.gemma-1_1-7b-it       | `completion('clarifai/gcp.generate.gemma-1_1-7b-it', messages)`          |

## Cohere LLMs
| Model Name                                    | Function Call                                                      |
|-----------------------------------------------|---------------------------------------------------------------------|
| clarifai/cohere.generate.cohere-generate-command | `completion('clarifai/cohere.generate.cohere-generate-command', messages)` |
 clarifai/cohere.generate.command-r-plus' | `completion('clarifai/clarifai/cohere.generate.command-r-plus', messages)`|

## Databricks LLMs

| Model Name                                        | Function Call                                                      |
|---------------------------------------------------|---------------------------------------------------------------------|
| clarifai/databricks.drbx.dbrx-instruct           | `completion('clarifai/databricks.drbx.dbrx-instruct', messages)`   |
| clarifai/databricks.Dolly-v2.dolly-v2-12b        | `completion('clarifai/databricks.Dolly-v2.dolly-v2-12b', messages)`|

## Microsoft LLMs

| Model Name                                        | Function Call                                                      |
|---------------------------------------------------|---------------------------------------------------------------------|
| clarifai/microsoft.text-generation.phi-2          | `completion('clarifai/microsoft.text-generation.phi-2', messages)`  |
| clarifai/microsoft.text-generation.phi-1_5        | `completion('clarifai/microsoft.text-generation.phi-1_5', messages)`|

## Salesforce models

| Model Name                                                | Function Call                                                                |
|-----------------------------------------------------------|-------------------------------------------------------------------------------|
| clarifai/salesforce.blip.general-english-image-caption-blip-2 | `completion('clarifai/salesforce.blip.general-english-image-caption-blip-2', messages)` |
| clarifai/salesforce.xgen.xgen-7b-8k-instruct             | `completion('clarifai/salesforce.xgen.xgen-7b-8k-instruct', messages)`         |


## Other Top performing LLMs

| Model Name                                        | Function Call                                                      |
|---------------------------------------------------|---------------------------------------------------------------------|
| clarifai/deci.decilm.deciLM-7B-instruct          | `completion('clarifai/deci.decilm.deciLM-7B-instruct', messages)`  |
| clarifai/upstage.solar.solar-10_7b-instruct      | `completion('clarifai/upstage.solar.solar-10_7b-instruct', messages)` |
| clarifai/openchat.openchat.openchat-3_5-1210     | `completion('clarifai/openchat.openchat.openchat-3_5-1210', messages)` |
| clarifai/togethercomputer.stripedHyena.stripedHyena-Nous-7B | `completion('clarifai/togethercomputer.stripedHyena.stripedHyena-Nous-7B', messages)` |
| clarifai/fblgit.una-cybertron.una-cybertron-7b-v2 | `completion('clarifai/fblgit.una-cybertron.una-cybertron-7b-v2', messages)` |
| clarifai/tiiuae.falcon.falcon-40b-instruct       | `completion('clarifai/tiiuae.falcon.falcon-40b-instruct', messages)` |
| clarifai/togethercomputer.RedPajama.RedPajama-INCITE-7B-Chat | `completion('clarifai/togethercomputer.RedPajama.RedPajama-INCITE-7B-Chat', messages)` |
| clarifai/bigcode.code.StarCoder                  | `completion('clarifai/bigcode.code.StarCoder', messages)`           |
| clarifai/mosaicml.mpt.mpt-7b-instruct            | `completion('clarifai/mosaicml.mpt.mpt-7b-instruct', messages)`     |
