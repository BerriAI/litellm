import Image from '@theme/IdealImage';

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
    "model": "gpt-4o-mini",
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
 "model": "gpt-4o-mini-2024-07-18",
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
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Hello good morning"}
    ],
    "guardrails": ["presidio-pre-guard"],
  }'
```

</TabItem>


</Tabs>


## Set `language` per request

## Output parsing 

## Ad Hoc Recognizers




