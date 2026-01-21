# Testing Guide for Structured Outputs Fix

This document explains how to test the structured outputs fix for the `/v1/messages` endpoint.

## Quick Summary of the Fix

The fix adds support for the `output_format` parameter in the `/v1/messages` endpoint, which enables structured JSON outputs for Claude Sonnet 4.5 and Opus 4.1 models.

## What Was Fixed

1. Added `output_format` to the supported parameters list
2. Added `output_format` to the TypedDict to prevent it from being stripped
3. Auto-injection of the `anthropic-beta: structured-outputs-2025-11-13` header

## Testing Methods

### Method 1: Unit Tests (No API Key Required)

The test suite validates the transformation logic without making actual API calls:

```bash
# Run the specific test file
poetry run pytest tests/test_litellm/llms/anthropic/experimental_pass_through/messages/test_anthropic_messages_structured_outputs.py -v
```

These tests verify:
- ✅ `output_format` is in supported parameters
- ✅ Request transformation preserves `output_format`
- ✅ Beta header is automatically added
- ✅ Headers merge correctly with existing beta headers
- ✅ Works for Bedrock and Azure Foundry models

### Method 2: Manual Verification Script

Run the verification script to inspect the transformation logic:

```bash
poetry run python verify_request_transformation.py
```

This will show you:
- The transformed request body
- The injected headers
- Validation that all pieces are in place

### Method 3: Integration Test Against Real API (Requires API Key)

#### For Anthropic Direct API:

```bash
# Set your API key
export ANTHROPIC_API_KEY=your-key-here

# Run the manual test
poetry run python test_structured_outputs_manual.py
```

#### For Amazon Bedrock:

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
export AWS_REGION_NAME=us-east-1

# Run the manual test
poetry run python test_structured_outputs_manual.py
```

#### For Azure Foundry:

```bash
# Test via LiteLLM proxy or use the Python client
curl --request POST \
  --url https://your-litellm-proxy/v1/messages \
  --header 'X-API-KEY: your-litellm-key' \
  --header 'content-type: application/json' \
  -d '{
    "model": "azure_ai/claude-sonnet-4-5",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "Extract info from: John Smith (john@example.com) wants Enterprise plan."
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
        "required": ["name", "email", "plan_interest"]
      }
    }
  }'
```

### Method 4: Via LiteLLM Proxy

1. Start the proxy:
```bash
litellm --config your_config.yaml
```

2. Make a request with `output_format`:
```bash
curl --request POST \
  --url http://localhost:4000/v1/messages \
  --header 'Authorization: Bearer your-api-key' \
  --header 'content-type: application/json' \
  -d '{
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Say hello"}],
    "output_format": {
      "type": "json_schema",
      "schema": {
        "type": "object",
        "properties": {
          "greeting": {"type": "string"}
        }
      }
    }
  }'
```

## Expected Results

### ✅ With `output_format` (FIXED)

The response should contain **JSON**:
```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "{\"name\": \"John Smith\", \"email\": \"john@example.com\", \"plan_interest\": \"Enterprise plan\"}"
    }
  ],
  "model": "claude-sonnet-4-5-20250929",
  "stop_reason": "end_turn",
  "usage": {...}
}
```

### ❌ Without `output_format` (Expected behavior)

The response contains **Markdown**:
```json
{
  "content": [
    {
      "type": "text",
      "text": "# Key Information\n\n- Name: John Smith\n- Email: john@example.com\n..."
    }
  ]
}
```

## Verification Checklist

When testing, verify:

- [ ] Request body includes `output_format` field
- [ ] Request headers include `anthropic-beta: structured-outputs-2025-11-13`
- [ ] Response content is valid JSON (can be parsed)
- [ ] Response JSON matches the provided schema
- [ ] Works with Anthropic direct API
- [ ] Works with Amazon Bedrock
- [ ] Works with Azure Foundry
- [ ] Works with Vertex AI (if applicable)

## Debugging

If structured outputs don't work:

1. **Check the request is reaching the provider**:
   - Set `LITELLM_LOG=DEBUG` to see full request details
   - Verify `output_format` is in the logged request body
   - Verify `anthropic-beta` header includes `structured-outputs-2025-11-13`

2. **Check the model supports structured outputs**:
   - Only Claude Sonnet 4.5 and Opus 4.1 support native structured outputs
   - Other models will fall back to tool-based JSON mode

3. **Check provider-specific issues**:
   - Bedrock: Ensure the model ARN is correct
   - Azure Foundry: Ensure the deployment supports the feature
   - Vertex AI: May need additional configuration

## Code Changes to Review

The fix involves these files:
1. `litellm/types/llms/anthropic.py` - Added `output_format` to TypedDict
2. `litellm/llms/anthropic/experimental_pass_through/messages/transformation.py` - Added to supported params and beta header injection
3. `tests/.../test_anthropic_messages_structured_outputs.py` - Comprehensive test coverage

## Additional Resources

- [Anthropic Structured Outputs Documentation](https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs)
- [LiteLLM Issue Discussion](https://github.com/BerriAI/litellm/issues/)
- Claude models that support structured outputs: `claude-sonnet-4-5`, `claude-opus-4-1`
