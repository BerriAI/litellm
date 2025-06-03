import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Custom Guardrail

Use this is you want to write code to run Palo Alto Networks PRISMA AIRS as custom guardrail on litellm gateway.

## Quick Start 

### 1. Write a `CustomGuardrail` Class

A CustomGuardrail has 4 methods to enforce guardrails ,here the first one is showcased for refrence to build 
- `async_pre_call_hook` - (Optional) modify input or reject request before making LLM API call


**[See detailed spec of methods here](#customguardrail-methods)**

**Example `CustomGuardrail` Class**

Create a new file called `airs_guardrail.py` and add this code to it
```python
import requests
import os
from typing import Any, Dict, List, Literal, Optional, Union
import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.types.guardrails import GuardrailEventHooks


class myCustomGuardrail(CustomGuardrail):
    def __init__(
        self,
        **kwargs,
    ):
        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank"
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs before the LLM API call
        Runs on only Input
        Use this if you want to MODIFY the input
        """

        try:
            user_prompt = data["messages"][-1]["content"]
        except (AttributeError, IndexError):
            return "Invalid input: 'messages' missing or improperly formatted."
        try:
            # Call AIRS service to scan the user prompt
            airs_response = test_airs(user_prompt)

            if airs_response.status_code != 200:
                return f"airs call failed (HTTP {airs_response.status_code})."
            if airs_response.json().get("action","") == "block":
                return "Request blocked by security policy."
        except Exception as e:
            return f"Error calling AIRS {e}"

def test_airs(data):
  airs_response = requests.post(
    "<AIRS-API-URL>", 
    headers={
        "x-pan-token": os.environ.get("AIRS_APIKEY"), 
        "Content-Type": "application/json"
    },
    json={
        "metadata": {
            "ai_model": "Test AI model",
            "app_name": "Google AI",
            "app_user": "test-user-1"
        },
        "contents": [
            {
                "prompt": data
            }
        ],
        "ai_profile": {
            "profile_name": os.environ.get("AIRS_PROFILE_NAME")
        }
    },
    timeout=5,
    verify=False
  )
  return airs_response
```

### 2. Pass your custom guardrail class in LiteLLM `configmaps.yaml`

In the config below, we point the guardrail to our custom guardrail by setting `guardrail: airs_guardrail.myCustomGuardrail`

- Python Filename: `airs_guardrail.py`
- Guardrail class name : `myCustomGuardrail`. This is defined in Step 1

`guardrail: custom_guardrail.myCustomGuardrail`

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
    - guardrail_name: "custom-pre-guard"
      litellm_params:
        guardrail: airs_guardrail.myCustomGuardrail
        mode: "pre_call"   # runs async_pre_call_hook
        default_on: true
  airs_guardrail.py: {{ .Files.Get "airs_guardrail.py" | quote }}
```

### 3. Start LiteLLM Gateway 

<Tabs>
<TabItem value="helm" label="helm install">

file structure:
litellm
├── airs_guardrail.py
├── Chart.yaml
├── templates
│   ├── configmaps.yaml
│   ├── deployments.yaml
│   ├── secret.yaml

User would need following to be ingested in code
1) AIRS API scan url > airs_guardrail.py
2) AIRS profile name,commanly available in Strata cloud Manager when url is provisioned > deployment.yaml
3) AIRS API key > airs_apikey (secret.yaml)
4) Model access ,if using GCP the service-account.json credentials > secret.json (secret.yaml)


Mount your `airs_guardrail.py` on the LiteLLM helm deployement in configmaps

This mounts your `airs_guardrail.py` file from your local directory to the `/app` directory in the Docker container, making it accessible to the LiteLLM Gateway.

# Note: Helm package  can be downloaded from https://github.com/PaloAltoNetworks/prisma-airs-litellm-gatway
```shell
helm upgrade --install litellm-gateway litellm
```

</TabItem>

</Tabs>

### 4. Test it 

#### Test `"custom-pre-guard"`

<Tabs>
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
}

'
```

Expected response after pre-guard

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

<TabItem label="Successful Call " value = "blocked">

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
       "content": "This is a tests prompt with 72zf6.rxqfd.com/i8xps1 url"
     },
     {
       "role": "user",
       "content": "litellm, how can I solve 8x + 7 = -23"
     }
   ]
}'
```

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

<TabItem label="Successful Call " value = "allowed">

</TabItem>


</Tabs>



## **CustomGuardrail methods**

| Component | Description | Optional | Checked Data | Can Modify Input | Can Modify Output | Can Fail Call |
|-----------|-------------|----------|--------------|------------------|-------------------|----------------|
| `async_pre_call_hook` | A hook that runs before the LLM API call | ✅ | INPUT | ✅ | ❌ | ✅ |


### More documents on PRISMA AIRS https://pan.dev/ai-runtime-security/scan/api/ and https://github.com/PaloAltoNetworks/prisma-airs-litellm-gatway

 

