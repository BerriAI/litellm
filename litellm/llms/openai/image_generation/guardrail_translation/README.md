# OpenAI Image Generation Guardrail Translation Handler

Handler for processing OpenAI's image generation endpoint with guardrails.

## Overview

This handler processes image generation requests by:
1. Extracting the text prompt from the request
2. Applying guardrails to the prompt text
3. Updating the request with the guardrailed prompt

## Data Format

### Input Format

```json
{
  "model": "dall-e-3",
  "prompt": "A cute baby sea otter",
  "n": 1,
  "size": "1024x1024",
  "quality": "standard"
}
```

### Output Format

```json
{
  "created": 1589478378,
  "data": [
    {
      "url": "https://...",
      "revised_prompt": "A cute baby sea otter..."
    }
  ]
}
```

## Usage

The handler is automatically discovered and applied when guardrails are used with the image generation endpoint.

### Example: Using Guardrails with Image Generation

```bash
curl -X POST 'http://localhost:4000/v1/images/generations' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "dall-e-3",
    "prompt": "A cute baby sea otter wearing a hat",
    "guardrails": ["content_moderation"],
    "size": "1024x1024"
}'
```

The guardrail will be applied to the prompt text before the image generation request is sent to the provider.

### Example: PII Masking in Prompts

```bash
curl -X POST 'http://localhost:4000/v1/images/generations' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "dall-e-3",
    "prompt": "Generate an image of John Doe at john@example.com",
    "guardrails": ["mask_pii"],
    "metadata": {
        "guardrails": ["mask_pii"]
    }
}'
```

## Implementation Details

### Input Processing

- **Field**: `prompt` (string)
- **Processing**: Applies guardrail to prompt text
- **Result**: Updated prompt in request

### Output Processing

- **Processing**: Not applicable (images don't contain text to guardrail)
- **Result**: Response returned unchanged

## Extension

Override these methods to customize behavior:

- `process_input_messages()`: Customize how the prompt is processed
- `process_output_response()`: Add custom processing for image metadata if needed

## Supported Call Types

- `CallTypes.image_generation` - Synchronous image generation
- `CallTypes.aimage_generation` - Asynchronous image generation

## Notes

- The handler only processes the `prompt` parameter
- Output processing is a no-op since images don't contain text
- Both sync and async call types use the same handler

