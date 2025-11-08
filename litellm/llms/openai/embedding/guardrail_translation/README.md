# OpenAI Embedding Guardrail Translation Handler

Handler for processing OpenAI's embedding endpoint with guardrails.

## Overview

This handler processes embedding requests by:
1. Extracting the text input(s) from the request
2. Applying guardrails to the input text(s)
3. Updating the request with the guardrailed input(s)

## Data Format

### Input Format

Single string input:
```json
{
  "model": "text-embedding-3-small",
  "input": "The food was delicious and the waiter..."
}
```

Multiple string inputs:
```json
{
  "model": "text-embedding-3-small",
  "input": [
    "The food was delicious",
    "The service was excellent"
  ]
}
```

Token array input (guardrails skipped for token arrays):
```json
{
  "model": "text-embedding-3-small",
  "input": [[1234, 5678, 9101]]
}
```

### Output Format

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0023, -0.009, ...],
      "index": 0
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

## Usage

The handler is automatically discovered and applied when guardrails are used with the embedding endpoint.

### Example: Using Guardrails with Embeddings

```bash
curl -X POST 'http://localhost:4000/v1/embeddings' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "text-embedding-3-small",
    "input": "The food was delicious and the waiter was friendly",
    "guardrails": ["content_moderation"]
}'
```

The guardrail will be applied to the input text before the embedding request is sent to the provider.

### Example: PII Masking in Embeddings

```bash
curl -X POST 'http://localhost:4000/v1/embeddings' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "text-embedding-3-small",
    "input": "Contact John Doe at john@example.com for details",
    "guardrails": ["mask_pii"],
    "metadata": {
        "guardrails": ["mask_pii"]
    }
}'
```

### Example: Multiple Inputs with Guardrails

```bash
curl -X POST 'http://localhost:4000/v1/embeddings' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "text-embedding-3-small",
    "input": [
        "First text to embed",
        "Second text to embed",
        "Third text to embed"
    ],
    "guardrails": ["content_moderation"]
}'
```

## Implementation Details

### Input Processing

- **Field**: `input` (string | string[] | number[][])
- **Processing**: 
  - Single string: Applies guardrail directly (if length ≤ 4096 characters)
  - Array of strings: Applies guardrail to each string (if length ≤ 4096 characters)
  - Array of token arrays: Skips guardrail (tokens cannot be meaningfully guardrailed)
- **Safety Limit**: Inputs exceeding 4096 characters are automatically skipped to prevent guardrail API failures
- **Result**: Updated input in request

### Output Processing

- **Processing**: Not applicable (embeddings are numeric vectors, not text)
- **Result**: Response returned unchanged

## Extension

Override these methods to customize behavior:

- `process_input_messages()`: Customize how the input is processed
- `process_output_response()`: Add custom processing for embedding metadata if needed

## Supported Call Types

- `CallTypes.embedding` - Synchronous embedding
- `CallTypes.aembedding` - Asynchronous embedding

## Notes

- The handler processes the `input` parameter for string and string array inputs
- Token array inputs are skipped (guardrails cannot be applied to tokens)
- Output processing is a no-op since embeddings are numeric vectors
- Both sync and async call types use the same handler
- Guardrails are applied to each individual string when processing arrays
- **Safety Limit**: Inputs exceeding 4096 characters are automatically skipped to prevent guardrail API failures due to size limits


