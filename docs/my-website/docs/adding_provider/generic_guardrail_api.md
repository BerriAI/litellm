# [BETA] Generic Guardrail API - Integrate Without a PR

## The Problem

As a guardrail provider, integrating with LiteLLM traditionally requires:
- Making a PR to the LiteLLM repository
- Waiting for review and merge
- Maintaining provider-specific code in LiteLLM's codebase
- Updating the integration for changes to your API

## The Solution

The **Generic Guardrail API** lets you integrate with LiteLLM **instantly** by implementing a simple API endpoint. No PR required.

### Key Benefits

1. **No PR Needed** - Deploy and integrate immediately
2. **Universal Support** - Works across ALL LiteLLM endpoints (chat, embeddings, image generation, etc.)
3. **Simple Contract** - One endpoint, three response types
4. **Multi-Modal Support** - Handle both text and images in requests/responses
5. **Custom Parameters** - Pass provider-specific params via config
6. **Full Control** - You own and maintain your guardrail API

## How It Works

1. LiteLLM extracts text and images from any request (chat messages, embeddings, image prompts, etc.)
2. Sends extracted content + metadata to your API endpoint
3. Your API responds with: `BLOCKED`, `NONE`, or `GUARDRAIL_INTERVENED`
4. LiteLLM enforces the decision and applies any modifications

## API Contract

### Endpoint

Implement `POST /beta/litellm_basic_guardrail_api`

### Request Format

```json
{
  "texts": ["extracted text from the request"],  // array of text strings
  "images": ["base64_encoded_image_data"],  // optional array of images
  "request_data": {
    "user_api_key_hash": "hash of the litellm virtual key used",
    "user_api_key_alias": "alias of the litellm virtual key used",
    "user_api_key_user_id": "user id associated with the litellm virtual key used",
    "user_api_key_user_email": "user email associated with the litellm virtual key used",
    "user_api_key_team_id": "team id associated with the litellm virtual key used",
    "user_api_key_team_alias": "team alias associated with the litellm virtual key used",
    "user_api_key_end_user_id": "end user id associated with the litellm virtual key used",
    "user_api_key_org_id": "org id associated with the litellm virtual key used"
  },
  "input_type": "request",  // "request" or "response"
  "litellm_call_id": "unique_call_id",  // the call id of the individual LLM call
  "litellm_trace_id": "trace_id",  // the trace id of the LLM call - useful if there are multiple LLM calls for the same conversation
  "additional_provider_specific_params": {
    // your custom params from config
  }
}
```

### Response Format

```json
{
  "action": "BLOCKED" | "NONE" | "GUARDRAIL_INTERVENED",
  "blocked_reason": "why content was blocked",  // required if action=BLOCKED
  "texts": ["modified text"],  // optional array of modified text strings
  "images": ["modified_base64_image"]  // optional array of modified images
}
```

**Actions:**
- `BLOCKED` - LiteLLM raises error and blocks request
- `NONE` - Request proceeds unchanged  
- `GUARDRAIL_INTERVENED` - Request proceeds with modified texts/images (provide `texts` and/or `images` fields)

## LiteLLM Configuration

Add to `config.yaml`:

```yaml
litellm_settings:
  guardrails:
    - guardrail_name: "my-guardrail"
      litellm_params:
        guardrail: generic_guardrail_api
        mode: pre_call  # or post_call, during_call
        api_base: https://your-guardrail-api.com
        api_key: os.environ/YOUR_GUARDRAIL_API_KEY  # optional
        additional_provider_specific_params:
          # your custom parameters
          threshold: 0.8
          language: "en"
```

## Usage

Users apply your guardrail by name:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "hello"}],
    guardrails=["my-guardrail"]
)
```

Or with dynamic parameters:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "hello"}],
    guardrails=[{
        "my-guardrail": {
            "extra_body": {
                "custom_threshold": 0.9
            }
        }
    }]
)
```

## Implementation Example

See [mock_bedrock_guardrail_server.py](https://github.com/BerriAI/litellm/blob/main/cookbook/mock_guardrail_server/mock_bedrock_guardrail_server.py) for a complete reference implementation.

**Minimal FastAPI example:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI()

class GuardrailRequest(BaseModel):
    texts: List[str]
    images: Optional[List[str]] = None
    request_data: Dict[str, Any]
    input_type: str  # "request" or "response"
    litellm_call_id: Optional[str] = None
    litellm_trace_id: Optional[str] = None
    additional_provider_specific_params: Dict[str, Any]

class GuardrailResponse(BaseModel):
    action: str  # BLOCKED, NONE, or GUARDRAIL_INTERVENED
    blocked_reason: Optional[str] = None
    texts: Optional[List[str]] = None
    images: Optional[List[str]] = None

@app.post("/beta/litellm_basic_guardrail_api")
async def apply_guardrail(request: GuardrailRequest):
    # Your guardrail logic here
    for text in request.texts:
        if "badword" in text.lower():
            return GuardrailResponse(
                action="BLOCKED",
                blocked_reason="Content contains prohibited terms"
            )
    
    return GuardrailResponse(action="NONE")
```

## When to Use This

✅ **Use Generic Guardrail API when:**
- You want instant integration without waiting for PRs
- You maintain your own guardrail service
- You need full control over updates and features
- You want to support all LiteLLM endpoints automatically

❌ **Make a PR when:**
- You want deeper integration with LiteLLM internals
- Your guardrail requires complex LiteLLM-specific logic
- You want to be featured as a built-in provider

## Questions?

This is a **beta API**. We're actively improving it based on feedback. Open an issue or PR if you need additional capabilities.

