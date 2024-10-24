import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bedrock

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
  - guardrail_name: "bedrock-pre-guard"
    litellm_params:
      guardrail: bedrock  # supported values: "aporia", "bedrock", "lakera"
      mode: "during_call"
      guardrailIdentifier: ff6ujrregl1q # your guardrail ID on bedrock
      guardrailVersion: "DRAFT"         # your guardrail version on bedrock
  
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
    "guardrails": ["bedrock-guard"]
  }'
```

Expected response on failure

```shell
{
  "error": {
    "message": {
      "error": "Violated guardrail policy",
      "bedrock_guardrail_response": {
        "action": "GUARDRAIL_INTERVENED",
        "assessments": [
          {
            "topicPolicy": {
              "topics": [
                {
                  "action": "BLOCKED",
                  "name": "Coffee",
                  "type": "DENY"
                }
              ]
            }
          }
        ],
        "blockedResponse": "Sorry, the model cannot answer this question. coffee guardrail applied ",
        "output": [
          {
            "text": "Sorry, the model cannot answer this question. coffee guardrail applied "
          }
        ],
        "outputs": [
          {
            "text": "Sorry, the model cannot answer this question. coffee guardrail applied "
          }
        ],
        "usage": {
          "contentPolicyUnits": 0,
          "contextualGroundingPolicyUnits": 0,
          "sensitiveInformationPolicyFreeUnits": 0,
          "sensitiveInformationPolicyUnits": 0,
          "topicPolicyUnits": 1,
          "wordPolicyUnits": 0
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
    "guardrails": ["bedrock-guard"]
  }'
```

</TabItem>


</Tabs>

