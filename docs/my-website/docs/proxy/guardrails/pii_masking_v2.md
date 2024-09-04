import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PII Masking - Presidio

## Quick Start

LiteLLM supports [Microsoft Presidio](https://github.com/microsoft/presidio/) for PII masking. 

### 1. Define Guardrails on your LiteLLM config.yaml 

Define your guardrails under the `guardrails` section
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "presidio-pre-guard"
    litellm_params:
      guardrail: presidio  # supported values: "aporia", "bedrock", "lakera", "presidio"
      mode: "pre_call"
```

Set the following env vars 

```bash
export PRESIDIO_ANALYZER_API_BASE="http://localhost:5002"
export PRESIDIO_ANONYMIZER_API_BASE="http://localhost:5001"
```

#### Supported values for `mode`

- `pre_call` Run **before** LLM call, on **input**
- `post_call` Run **after** LLM call, on **input & output**
- `logging_only` Run **after** LLM call, only apply PII Masking before logging to Langfuse, etc. Not on the actual llm api request / response.


### 2. Start LiteLLM Gateway 


```shell
litellm --config config.yaml --detailed_debug
```

### 3. Test request 

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Masked PII call" value = "not-allowed">

Expect this to mask `Jane Doe` since it's PII

```shell
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello my name is Jane Doe"}
    ],
    "guardrails": ["presidio-pre-guard"],
  }'
```

Expected response on failure

```shell
{
 "id": "chatcmpl-A3qSC39K7imjGbZ8xCDacGJZBoTJQ",
 "choices": [
   {
     "finish_reason": "stop",
     "index": 0,
     "message": {
       "content": "Hello, <PERSON>! How can I assist you today?",
       "role": "assistant",
       "tool_calls": null,
       "function_call": null
     }
   }
 ],
 "created": 1725479980,
 "model": "gpt-3.5-turbo-2024-07-18",
 "object": "chat.completion",
 "system_fingerprint": "fp_5bd87c427a",
 "usage": {
   "completion_tokens": 13,
   "prompt_tokens": 14,
   "total_tokens": 27
 },
 "service_tier": null
}
```

</TabItem>

<TabItem label="No PII Call " value = "allowed">

```shell
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello good morning"}
    ],
    "guardrails": ["presidio-pre-guard"],
  }'
```

</TabItem>


</Tabs>

## Advanced

###  Set `language` per request

The Presidio API [supports passing the `language` param](https://microsoft.github.io/presidio/api-docs/api-docs.html#tag/Analyzer/paths/~1analyze/post). Here is how to set the `language` per request

<Tabs>
<TabItem label="curl" value = "curl">

```shell
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "is this credit card number 9283833 correct?"}
    ],
    "guardrails": ["presidio-pre-guard"],
    "guardrail_config": {"language": "es"}
  }'
```

</TabItem>


<TabItem label="OpenAI Python SDK" value = "python">

```python

import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    extra_body={ 
        "metadata": {
            "guardrails": ["presidio-pre-guard"],
            "guardrail_config": {"language": "es"}
        }
    }
)
print(response)
```

</TabItem>

</Tabs>


### Output parsing 


LLM responses can sometimes contain the masked tokens. 

For presidio 'replace' operations, LiteLLM can check the LLM response and replace the masked token with the user-submitted values. 

Define your guardrails under the `guardrails` section
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "presidio-pre-guard"
    litellm_params:
      guardrail: presidio  # supported values: "aporia", "bedrock", "lakera", "presidio"
      mode: "pre_call"
      output_parse_pii: True
```

**Expected Flow: **

1. User Input: "hello world, my name is Jane Doe. My number is: 034453334"

2. LLM Input: "hello world, my name is [PERSON]. My number is: [PHONE_NUMBER]"

3. LLM Response: "Hey [PERSON], nice to meet you!"

4. User Response: "Hey Jane Doe, nice to meet you!"

### Ad Hoc Recognizers


Send ad-hoc recognizers to presidio `/analyze` by passing a json file to the proxy 

[**Example** ad-hoc recognizer](../../../../litellm/proxy/hooks/example_presidio_ad_hoc_recognize)

#### Define ad-hoc recognizer on your LiteLLM config.yaml 

Define your guardrails under the `guardrails` section
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "presidio-pre-guard"
    litellm_params:
      guardrail: presidio  # supported values: "aporia", "bedrock", "lakera", "presidio"
      mode: "pre_call"
      presidio_ad_hoc_recognizers: "./hooks/example_presidio_ad_hoc_recognizer.json"
```

Set the following env vars 

```bash
export PRESIDIO_ANALYZER_API_BASE="http://localhost:5002"
export PRESIDIO_ANONYMIZER_API_BASE="http://localhost:5001"
```


You can see this working, when you run the proxy: 

```bash
litellm --config /path/to/config.yaml --debug
```

Make a chat completions request, example:

```
{
  "model": "azure-gpt-3.5",
  "messages": [{"role": "user", "content": "John Smith AHV number is 756.3026.0705.92. Zip code: 1334023"}]
}
```

And search for any log starting with `Presidio PII Masking`, example:
```
Presidio PII Masking: Redacted pii message: <PERSON> AHV number is <AHV_NUMBER>. Zip code: <US_DRIVER_LICENSE>
```

### Logging Only


Only apply PII Masking before logging to Langfuse, etc.

Not on the actual llm api request / response.

:::note
This is currently only applied for 
- `/chat/completion` requests
- on 'success' logging

:::

1. Define mode: `logging_only` on your LiteLLM config.yaml 

Define your guardrails under the `guardrails` section
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "presidio-pre-guard"
    litellm_params:
      guardrail: presidio  # supported values: "aporia", "bedrock", "lakera", "presidio"
      mode: "logging_only"
```

Set the following env vars 

```bash
export PRESIDIO_ANALYZER_API_BASE="http://localhost:5002"
export PRESIDIO_ANONYMIZER_API_BASE="http://localhost:5001"
```


2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "Hi, my name is Jane!"
    }
  ]
  }'
```


**Expected Logged Response**

```
Hi, my name is <PERSON>!
```


