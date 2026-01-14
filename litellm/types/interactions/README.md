# Interactions API Types

This directory contains type definitions for the Google Interactions API.

## Generated Types

The `generated.py` file is auto-generated from the official OpenAPI spec:
https://ai.google.dev/static/api/interactions.openapi.json

### How to Regenerate

When the API spec changes, regenerate the types with:

```bash
pip install datamodel-code-generator

datamodel-codegen \
    --url "https://ai.google.dev/static/api/interactions.openapi.json" \
    --output litellm/types/interactions/generated.py \
    --output-model-type pydantic_v2.BaseModel \
    --target-python-version 3.9
```

Then add the LiteLLM-specific types at the bottom of the generated file:
- `InteractionsAPIResponse`
- `InteractionsAPIStreamingResponse`
- `DeleteInteractionResult`
- `CancelInteractionResult`

### Key Types

**Request Types:**
- `CreateModelInteractionParams` - For model interactions
- `CreateAgentInteractionParams` - For agent interactions

**Content Types:**
- `Content` - Union of all content types (text, image, audio, etc.)
- `TextContent` - Text content with `type: "text"`
- `Turn` - A turn in multi-turn conversation with `role` and `content`

**Tool Types:**
- `Tool` - Union of all tool types
- `Function` - Function tool declaration

**Response Types:**
- `InteractionsAPIResponse` - LiteLLM response wrapper
- `InteractionsAPIStreamingResponse` - Streaming response chunk

