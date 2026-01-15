import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Lakera AI

## Quick Start
### 1. Define Guardrails on your LiteLLM config.yaml 

Define your guardrails under the `guardrails` section

```yaml showLineNumbers title="litellm config.yaml"
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "lakera-guard"
    litellm_params:
      guardrail: lakera_v2  # supported values: "aporia", "bedrock", "lakera"
      mode: "during_call"
      api_key: os.environ/LAKERA_API_KEY
      api_base: os.environ/LAKERA_API_BASE
  - guardrail_name: "lakera-pre-guard"
    litellm_params:
      guardrail: lakera_v2  # supported values: "aporia", "bedrock", "lakera"
      mode: "pre_call"
      api_key: os.environ/LAKERA_API_KEY
      api_base: os.environ/LAKERA_API_BASE
  - guardrail_name: "lakera-monitor"
    litellm_params:
      guardrail: lakera_v2
      mode: "pre_call"
      on_flagged: "monitor"  # Log violations but don't block
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

```shell showLineNumbers title="Curl Request"
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

```shell showLineNumbers title="Curl Request"
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


## Supported Params 

```yaml
guardrails:
  - guardrail_name: "lakera-guard"
    litellm_params:
      guardrail: lakera_v2  # supported values: "aporia", "bedrock", "lakera"
      mode: "during_call"
      api_key: os.environ/LAKERA_API_KEY
      api_base: os.environ/LAKERA_API_BASE
      ### OPTIONAL ### 
      # project_id: Optional[str] = None,
      # payload: Optional[bool] = True,
      # breakdown: Optional[bool] = True,
      # metadata: Optional[Dict] = None,
      # dev_info: Optional[bool] = True,
      # on_flagged: Optional[str] = "block",  # "block" or "monitor"
```

- `api_base`: (Optional[str]) The base of the Lakera integration. Defaults to `https://api.lakera.ai` 
- `api_key`: (str) The API Key for the Lakera integration.
- `project_id`: (Optional[str]) ID of the relevant project
- `payload`: (Optional[bool]) When true the response will return a payload object containing any PII, profanity or custom detector regex matches detected, along with their location within the contents. 
- `breakdown`: (Optional[bool]) When true the response will return a breakdown list of the detectors that were run, as defined in the policy, and whether each of them detected something or not.
- `metadata`: (Optional[Dict]) Metadata tags can be attached to screening requests as an object that can contain any arbitrary key-value pairs. 
- `dev_info`: (Optional[bool]) When true the response will return an object with developer information about the build of Lakera Guard.
- `on_flagged`: (Optional[str]) Action to take when content is flagged. Defaults to `"block"`. 
  - `"block"`: Raises an HTTP 400 exception when violations are detected (default behavior)
  - `"monitor"`: Logs violations but allows the request to proceed. Useful for tuning security policies without blocking legitimate requests.
