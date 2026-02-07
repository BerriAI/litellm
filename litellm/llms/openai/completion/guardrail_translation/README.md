# OpenAI Text Completion Guardrail Translation Handler

Handler for processing OpenAI's text completion endpoint (`/v1/completions`) with guardrails.

## Overview

This handler processes text completion requests by:
1. Extracting the text prompt(s) from the request
2. Applying guardrails to the prompt text(s)
3. Updating the request with the guardrailed prompt(s)
4. Applying guardrails to the completion output text

## Data Format

### Input Format

**Single Prompt:**
```json
{
  "model": "gpt-3.5-turbo-instruct",
  "prompt": "Say this is a test",
  "max_tokens": 7,
  "temperature": 0
}
```

**Multiple Prompts (Batch):**
```json
{
  "model": "gpt-3.5-turbo-instruct",
  "prompt": [
    "Tell me a joke",
    "Write a poem"
  ],
  "max_tokens": 50
}
```

### Output Format

```json
{
  "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
  "object": "text_completion",
  "created": 1589478378,
  "model": "gpt-3.5-turbo-instruct",
  "choices": [
    {
      "text": "\n\nThis is indeed a test",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}
```

## Usage

The handler is automatically discovered and applied when guardrails are used with the text completion endpoint.

### Example: Using Guardrails with Text Completion

```bash
curl -X POST 'http://localhost:4000/v1/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "gpt-3.5-turbo-instruct",
    "prompt": "Say this is a test",
    "guardrails": ["content_moderation"],
    "max_tokens": 7
}'
```

The guardrail will be applied to both:
- **Input**: The prompt text before sending to the LLM
- **Output**: The completion text in the response

### Example: PII Masking in Prompts and Completions

```bash
curl -X POST 'http://localhost:4000/v1/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "gpt-3.5-turbo-instruct",
    "prompt": "My name is John Doe and my email is john@example.com",
    "guardrails": ["mask_pii"],
    "metadata": {
        "guardrails": ["mask_pii"]
    }
}'
```

### Example: Batch Prompts with Guardrails

```bash
curl -X POST 'http://localhost:4000/v1/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "gpt-3.5-turbo-instruct",
    "prompt": [
        "Tell me about AI",
        "What is machine learning?"
    ],
    "guardrails": ["content_filter"],
    "max_tokens": 100
}'
```

## Implementation Details

### Input Processing

- **Field**: `prompt` (string or list of strings)
- **Processing**: 
  - String prompts: Apply guardrail directly
  - List prompts: Apply guardrail to each string in the list
- **Result**: Updated prompt(s) in request

### Output Processing

- **Field**: `choices[*].text` (string)
- **Processing**: Applies guardrail to each completion text
- **Result**: Updated completion texts in response

### Supported Prompt Types

1. **String**: Single prompt as a string
2. **List of Strings**: Multiple prompts for batch completion
3. **List of Lists**: Token-based prompts (passed through unchanged)

## Extension

Override these methods to customize behavior:

- `process_input_messages()`: Customize how prompts are processed
- `process_output_response()`: Customize how completion texts are processed

## Supported Call Types

- `CallTypes.text_completion` - Synchronous text completion
- `CallTypes.atext_completion` - Asynchronous text completion

## Notes

- The handler processes both input prompts and output completion texts
- List prompts are processed individually (each string in the list)
- Non-string prompt items (e.g., token lists) are passed through unchanged
- Both sync and async call types use the same handler

