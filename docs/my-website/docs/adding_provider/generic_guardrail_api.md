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

## Supported Endpoints

The Generic Guardrail API works with the following LiteLLM endpoints:

- `/v1/chat/completions` - OpenAI Chat Completions
- `/v1/completions` - OpenAI Text Completions
- `/v1/responses` - OpenAI Responses API
- `/v1/images/generations` - OpenAI Image Generation
- `/v1/audio/transcriptions` - OpenAI Audio Transcriptions
- `/v1/audio/speech` - OpenAI Text-to-Speech
- `/v1/messages` - Anthropic Messages
- `/v1/rerank` - Cohere Rerank
- Pass-through endpoints

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
  "tools": [  // tool calls sent to the LLM (in the OpenAI Chat Completions spec)
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }
  ],
  "tool_calls": [  // tool calls received from the LLM (in the OpenAI Chat Completions spec)
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"San Francisco\"}"
      }
    }
  ],
  "structured_messages": [  // optional, full messages in OpenAI format (for chat endpoints)
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello"}
  ],
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

## Parameters

### `tools` Parameter

The `tools` parameter provides information about available function/tool definitions in the request.

**Format:** OpenAI `ChatCompletionToolParam` format (see [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools))

**Example:**
```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather in a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City and state, e.g. San Francisco, CA"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"]
        }
      },
      "required": ["location"]
    }
  }
}
```

**Availability:**
- **Input only:** Tools are only passed for `input_type="request"` (pre-call guardrails). Output/response guardrails do not currently receive tool definitions.
- **Supported endpoints:** The `tools` parameter is supported on: `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`. Other endpoints do not have tool support.

**Use cases:**
- Enforce tool permission policies (e.g., only allow certain users/teams to access specific tools)
- Validate tool schemas before sending to LLM
- Log tool usage for audit purposes
- Block sensitive tools based on user context

### `tool_calls` Parameter

The `tool_calls` parameter contains actual function/tool invocations being made in the request or response.

**Format:** OpenAI `ChatCompletionMessageToolCall` format (see [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/object#chat/object-tool_calls))

**Example:**
```json
{
  "id": "call_abc123",
  "type": "function",
  "function": {
    "name": "get_weather",
    "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}"
  }
}
```

**Key Difference from `tools`:**
- **`tools`** = Tool definitions/schemas (what tools are *available*)
- **`tool_calls`** = Tool invocations/executions (what tools are *being called* with what arguments)

**Availability:**
- **Both input and output:** Tool calls can be present in both `input_type="request"` (assistant messages requesting tool calls) and `input_type="response"` (LLM responses with tool calls).
- **Supported endpoints:** The `tool_calls` parameter is supported on: `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`.

**Use cases:**
- Validate tool call arguments before execution
- Redact sensitive data from tool call arguments (e.g., PII)
- Log tool invocations for audit/debugging
- Block tool calls with dangerous parameters
- Modify tool call arguments (e.g., enforce constraints, sanitize inputs)
- Monitor tool usage patterns across users/teams

### `structured_messages` Parameter

The `structured_messages` parameter provides the full input in OpenAI chat completion spec format, useful for distinguishing between system and user messages.

**Format:** Array of OpenAI chat completion messages (see [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages))

**Example:**
```json
[
  {"role": "system", "content": "You are a helpful assistant"},
  {"role": "user", "content": "Hello"}
]
```

**Availability:**
- **Supported endpoints:** `/v1/chat/completions`, `/v1/messages`, `/v1/responses`
- **Input only:** Only passed for `input_type="request"` (pre-call guardrails)

**Use cases:**
- Apply different policies for system vs user messages
- Enforce role-based content restrictions
- Log structured conversation context

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
    tools: Optional[List[Dict[str, Any]]] = None  # OpenAI ChatCompletionToolParam format (tool definitions)
    tool_calls: Optional[List[Dict[str, Any]]] = None  # OpenAI ChatCompletionMessageToolCall format (tool invocations)
    structured_messages: Optional[List[Dict[str, Any]]] = None  # OpenAI messages format (for chat endpoints)
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
    
    # Example: Check text content
    for text in request.texts:
        if "badword" in text.lower():
            return GuardrailResponse(
                action="BLOCKED",
                blocked_reason="Content contains prohibited terms"
            )
    
    # Example: Check tool definitions (if present in request)
    if request.tools:
        for tool in request.tools:
            if tool.get("type") == "function":
                function_name = tool.get("function", {}).get("name", "")
                # Block sensitive tool definitions
                if function_name in ["delete_data", "access_admin_panel"]:
                    return GuardrailResponse(
                        action="BLOCKED",
                        blocked_reason=f"Tool '{function_name}' is not allowed"
                    )
    
    # Example: Check tool calls (if present in request or response)
    if request.tool_calls:
        for tool_call in request.tool_calls:
            if tool_call.get("type") == "function":
                function_name = tool_call.get("function", {}).get("name", "")
                arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                
                # Parse arguments and validate
                import json
                try:
                    arguments = json.loads(arguments_str)
                    # Block dangerous arguments
                    if "file_path" in arguments and ".." in str(arguments["file_path"]):
                        return GuardrailResponse(
                            action="BLOCKED",
                            blocked_reason="Tool call contains path traversal attempt"
                        )
                except json.JSONDecodeError:
                    pass
    
    # Example: Check structured messages (if present in request)
    if request.structured_messages:
        for message in request.structured_messages:
            if message.get("role") == "system":
                # Apply stricter policies to system messages
                if "admin" in message.get("content", "").lower():
                    return GuardrailResponse(
                        action="BLOCKED",
                        blocked_reason="System message contains restricted terms"
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

