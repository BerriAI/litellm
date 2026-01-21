# Fix for Structured Outputs in `/v1/messages` Endpoint

## Issue Description
The `/v1/messages` endpoint for Claude Sonnet 4.5 deployed in Azure Foundry and Amazon Bedrock was not properly handling the `output_format` parameter for structured outputs. When users sent requests with `output_format`, the response was Markdown text instead of JSON, even though direct calls to the provider APIs worked correctly.

## Root Cause
The `output_format` parameter was not recognized as a valid parameter in the `/v1/messages` endpoint implementation. Specifically:

1. **Missing from TypedDict**: `output_format` was not included in the `AnthropicMessagesRequestOptionalParams` TypedDict, causing it to be stripped from requests.

2. **Missing from supported params**: The `get_supported_anthropic_messages_params()` method in `AnthropicMessagesConfig` did not include `"output_format"` in its list of supported parameters.

3. **Missing beta header injection**: The `_update_headers_with_anthropic_beta()` method did not automatically add the `structured-outputs-2025-11-13` beta header when `output_format` was present.

## Files Changed

### 1. `/home/user/litellm/litellm/types/llms/anthropic.py`
**Change**: Added `output_format` field to `AnthropicMessagesRequestOptionalParams` TypedDict

```python
class AnthropicMessagesRequestOptionalParams(TypedDict, total=False):
    # ... existing fields ...
    output_format: Optional[AnthropicOutputSchema]  # Structured outputs support
```

### 2. `/home/user/litellm/litellm/llms/anthropic/experimental_pass_through/messages/transformation.py`
**Changes**:
a) Added `"output_format"` to supported parameters list:
```python
def get_supported_anthropic_messages_params(self, model: str) -> list:
    return [
        # ... existing params ...
        "output_format",
        # ...
    ]
```

b) Updated `_update_headers_with_anthropic_beta()` to inject structured-outputs beta header:
```python
# Check for structured outputs
if optional_params.get("output_format") is not None:
    beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.STRUCTURED_OUTPUT_2025_09_25.value)
```

### 3. `/home/user/litellm/tests/test_litellm/llms/anthropic/experimental_pass_through/messages/test_anthropic_messages_structured_outputs.py`
**Change**: Created comprehensive test suite to verify structured outputs support

Tests include:
- Verification that `output_format` is in supported parameters
- Request transformation preserves `output_format`
- Beta header is automatically added
- Beta headers merge correctly with existing headers
- Integration test for full request flow
- Specific tests for Bedrock and Azure Foundry models

## Impact

This fix applies to **all** providers that use the `/v1/messages` endpoint, including:
- **Anthropic** (direct API calls)
- **Amazon Bedrock** (via `AmazonAnthropicClaudeMessagesConfig` which inherits from `AnthropicMessagesConfig`)
- **Azure Foundry** (via `AzureAnthropicMessagesConfig` which inherits from `AnthropicMessagesConfig`)
- **Vertex AI** (via `VertexAIPartnerModelsAnthropicMessagesConfig` which inherits from `AnthropicMessagesConfig`)

All these implementations inherit from `AnthropicMessagesConfig`, so the fix automatically propagates to all of them.

## Verification

The fix ensures that:
1. The `output_format` parameter is preserved throughout the request pipeline
2. The `anthropic-beta: structured-outputs-2025-11-13` header is automatically injected
3. The complete request (including `output_format` in the body and the beta header) is sent to the provider API
4. Structured outputs work correctly for Claude Sonnet 4.5 and Opus 4.1 models on all supported providers

## Example Usage

After this fix, users can use structured outputs with the `/v1/messages` endpoint:

```bash
curl --request POST \
  --url https://litellm.example.com/v1/messages \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  --header "X-API-KEY: <api-key>" \
  -d '{
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan."
      }
    ],
    "output_format": {
      "type": "json_schema",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "email": {"type": "string"},
          "plan_interest": {"type": "string"}
        },
        "required": ["name", "email", "plan_interest"],
        "additionalProperties": false
      }
    }
  }'
```

This will now correctly return JSON output instead of Markdown text.
