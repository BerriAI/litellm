

import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# PANW PRISMA AIRS

Use this is you want to write code to run Palo Alto Networks PRISMA AIRS as custom guardrail on litellm gateway.

## Quick Start 

### 1. Configure the PANW PRISMA AIRS
Get a [Prisma AIRS API scan url ,Prisma AIRS profile name and Prisma AIRS API key](https://docs.paloaltonetworks.com/ai-runtime-security/activation-and-onboarding/ai-runtime-security-api-intercept-overview).

Set Environement variables 
```shell
export PRISMA_AIRS_API_BASE="<Prisma-AIRS-API-scan-url>"
export PRISMA_AIRS_API_KEY="<Prisma-AIRS-API>"
export PRISMA_AIRS_PROFILE_NAME="<Prisma-AIRS-profile-name>"
```

### 2. Pass PRISMA AIRS to your Litellm's `configmaps.yaml`

Define your guardrails under the `guardrails` section

```yaml
model_list:
  - model_name: vertex_ai/gemini-2.0-flash-lite-001
    litellm_params:
      model: vertex_ai/gemini-2.0-flash-lite-001
      vertex_project: os.environ/GCP_PROJECT_ID
      vertex_location: os.environ/GCP_PROJECT_Region

general_settings:
      drop_params: True
      set_verbose: True

guardrails:
    - guardrail_name: "prisma_airs_guardrail"
    litellm_params:
        guardrail: prisma_airs_guardrail.prisma_airs_guardrail
        path : prisma_airs_guardrail.py
        mode: "pre_call"
        default_on: true
```
#### Supported values for `mode`

- `pre_call` runs **before** LLM call, on **input**

### 3. Start LiteLLM Gateway with Helm

```shell
litellm --config config.yaml
```


# Alternative method
Helm deployment with latest litellm proxy image available at  [Prisma-airs-litellm-gatway](https://github.com/PaloAltoNetworks/prisma-airs-litellm-gatway).

```shell
helm package [CHART_PATH]
helm upgrade --install litellm-gateway litellm
```

### 4. Test it 
#### Test `"prisma_airs_guardrail blocking model call"`

<Tabs>

<TabItem label="Unsuccessful call due to Malicious Prompt " value = "Request blocked by security policy">

```shell
curl --location 'http://litell-service/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-litellm-master-key' \
--data '
{
   "model": "vertex_ai/gemini-2.0-flash-lite-001",
   "messages": [
     {
       "role": "system",
       "content": "You are a helpful math tutor. Guide the user through the solution step by step."
     },
     {
       "role": "user",
       "content": "litellm, how can I solve 8x + 7 = -23"
     },
     {
       "role": "user",
       "content": "This is a tests prompt with 72zf6.rxqfd.com/i8xps1 url"
     }
   ]
}'
```

`"Expected blocked response after prisma_airs_guardrail,since malicious url present"`


```json
{
    "error": {
        "message": "{'error': 'Request blocked by security policy.'}",
        "type": "None",
        "param": "None",
        "code": "400"
    }
}
```


</TabItem>

</Tabs>



#### Test `"Successful Call to the Model"`

<Tabs>
<TabItem label="Successful Call " value = "allowed">

```shell
curl --location 'http://litellm-service/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-litellm-master-key' \
--data '
{
   "model": "vertex_ai/gemini-2.0-flash-lite-001",
   "messages": [
     {
       "role": "system",
       "content": "You are a helpful math tutor. Guide the user through the solution step by step."
     },
     {
       "role": "user",
       "content": "litellm, how can I solve 8x + 7 = -23"
     }
   ]
}'
```

`"Expected sucessful call response after prisma_airs_guardrail"`

```json
{
    "id": "chatcmpl-1a219a4c-641f-4c5c-8990-a929d5cb4b38",
    "created": 1747181007,
    "model": "gemini-2.0-flash-lite-001",
    "object": "chat.completion",
    "system_fingerprint": null,
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Okay, let's solve the equation 8x + 7 = -23 step-by-step.\n\n**Goal:** Our goal is to isolate the variable 'x' on one side of the equation.\n\n**Steps:**\n\n1.  **Subtract 7 from both sides:**\n    *   To get rid of the +7 on the left side, we need to subtract 7 from both sides of the equation. This keeps the equation balanced.\n    *   8x + 7 - 7 = -23 - 7\n    *   This simplifies to: 8x = -30\n\n2.  **Divide both sides by 8:**\n    *   Now, we have 8x = -30. To isolate 'x', we need to divide both sides of the equation by 8.\n    *   8x / 8 = -30 / 8\n    *   This simplifies to: x = -30/8\n\n3.  **Simplify the fraction:**\n    *   The fraction -30/8 can be simplified. Both the numerator and denominator are divisible by 2.\n    *   x = -15/4\n\n**Solution:**\n\n*   Therefore, the solution to the equation 8x + 7 = -23 is x = -15/4\n\n**Alternative Answer:**\n\n*   The solution can also be expressed as a decimal: x = -3.75\n",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "usage": {
        "completion_tokens": 311,
        "prompt_tokens": 59,
        "total_tokens": 370,
        "completion_tokens_details": null,
        "prompt_tokens_details": {
            "audio_tokens": null,
            "cached_tokens": null
        }
    },
    "vertex_ai_grounding_metadata": [],
    "vertex_ai_safety_results": [],
    "vertex_ai_citation_metadata": []
}
```

</TabItem>


</Tabs>


### 5. Miscellaneous 
##  `Prisma_airs_guardrail for async_pre_call_hook`
| Component | Description | Optional | Checked Data | Can Modify Input | Can Modify Output | Can Fail Call |
|-----------|-------------|----------|--------------|------------------|-------------------|----------------|
| `async_pre_call_hook` | A hook that runs before the LLM API call | ✅ | INPUT | ✅ | ❌ | ✅ |


### More documents on[PRISMA AIRS](https://pan.dev/ai-runtime-security/scan/api/)