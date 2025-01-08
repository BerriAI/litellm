import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Lakera AI

## Quick Start
### 1. Define Guardrails on your LiteLLM config.yaml 

Define your guardrails under the `guardrails` section
```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "lakera-guard"
    litellm_params:
      guardrail: lakera  # supported values: "aporia", "bedrock", "lakera"
      mode: "during_call"
      api_key: os.environ/LAKERA_API_KEY
      api_base: os.environ/LAKERA_API_BASE
  - guardrail_name: "lakera-pre-guard"
    litellm_params:
      guardrail: lakera  # supported values: "aporia", "bedrock", "lakera"
      mode: "pre_call"
      api_key: os.environ/LAKERA_API_KEY
      api_base: os.environ/LAKERA_API_BASE
  
```

#### Supported values for `mode`

- `pre_call` Run **before** LLM call, on **input**
- `post_call` Run **after** LLM call, on **input & output**
- `during_call` Run **during** LLM call, on **input** Same as `pre_call` but runs in parallel as LLM call.  Response not returned until guardrail check completes

### 2. Start LiteLLM Gateway 


```shell
litellm --config config.yaml --detailed_debug
```

### 3. Test request 

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Unsuccessful call" value = "not-allowed">

Expect this to fail since since `ishaan@berri.ai` in the request is PII

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi my email is ishaan@berri.ai"}
    ],
    "guardrails": ["lakera-guard"]
  }'
```

Expected response on failure

```shell
{
 "error": {
   "message": {
     "error": "Violated content safety policy",
     "lakera_ai_response": {
       "model": "lakera-guard-1",
       "results": [
         {
           "categories": {
             "prompt_injection": true,
             "jailbreak": false
           },
           "category_scores": {
             "prompt_injection": 0.999,
             "jailbreak": 0.0
           },
           "flagged": true,
           "payload": {}
         }
       ],
       "dev_info": {
         "git_revision": "cb163444",
         "git_timestamp": "2024-08-19T16:00:28+02:00",
         "version": "1.3.53"
       }
     }
   },
   "type": "None",
   "param": "None",
   "code": "400"
 }
}

```

</TabItem>

<TabItem label="Successful Call " value = "allowed">

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-npnwjPQciVRok5yNZgKmFQ" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "hi what is the weather"}
    ],
    "guardrails": ["lakera-guard"]
  }'
```

</TabItem>


</Tabs>

## Advanced 
### Set category-based thresholds.

Lakera has 2 categories for prompt_injection attacks:
- jailbreak
- prompt_injection

```yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

guardrails:
  - guardrail_name: "lakera-guard"
    litellm_params:
      guardrail: lakera  # supported values: "aporia", "bedrock", "lakera"
      mode: "during_call"
      api_key: os.environ/LAKERA_API_KEY
      api_base: os.environ/LAKERA_API_BASE
      category_thresholds:
        prompt_injection: 0.1
        jailbreak: 0.1
  
```