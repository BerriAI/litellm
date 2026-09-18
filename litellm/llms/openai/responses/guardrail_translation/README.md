# OpenAI Responses API Guardrail Translation Handler

This module provides guardrail translation support for the OpenAI Responses API format.

## Overview

The `OpenAIResponsesHandler` class handles the translation of guardrail operations for both input and output of the Responses API. It follows the same pattern as the Chat Completions handler but is adapted for the Responses API's specific data structures.

## Responses API Format

### Input Format
The Responses API accepts input in two formats:

1. **String input**: Simple text string
   ```python
   {"input": "Hello world", "model": "gpt-4"}
   ```

2. **List input**: Array of message objects (ResponseInputParam)
   ```python
   {
       "input": [
           {
               "role": "user",
               "content": "Hello",  # Can be string or list of content items
               "type": "message"
           }
       ],
       "model": "gpt-4"
   }
   ```

### Output Format
The Responses API returns a `ResponsesAPIResponse` object with:

```python
{
    "id": "resp_123",
    "output": [
        {
            "type": "message",
            "id": "msg_123",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Assistant response",
                    "annotations": []
                }
            ]
        }
    ]
}
```

## Usage

The handler is automatically discovered and registered for `CallTypes.responses` and `CallTypes.aresponses`.

### Example

```python
from litellm.llms import get_guardrail_translation_mapping
from litellm.types.utils import CallTypes

# Get the handler
handler_class = get_guardrail_translation_mapping(CallTypes.responses)
handler = handler_class()

# Process input
data = {"input": "User message", "model": "gpt-4"}
processed_data = await handler.process_input_messages(data, guardrail_instance)

# Process output
response = await litellm.aresponses(**processed_data)
processed_response = await handler.process_output_response(response, guardrail_instance)
```

## Key Methods

### `process_input_messages(data, guardrail_to_apply)`
Processes input data by:
1. Handling both string and list input formats
2. Extracting text content from messages
3. Applying guardrails to text content in parallel
4. Mapping guardrail responses back to the original structure

### `process_output_response(response, guardrail_to_apply)`
Processes output response by:
1. Extracting text from output items' content
2. Applying guardrails to all text content in parallel
3. Replacing original text with guardrailed versions

## Extending the Handler

The handler can be customized by overriding these methods:

- `_extract_input_text_and_create_tasks()`: Customize input text extraction logic
- `_apply_guardrail_responses_to_input()`: Customize how guardrail responses are applied to input
- `_extract_output_text_and_create_tasks()`: Customize output text extraction logic
- `_apply_guardrail_responses_to_output()`: Customize how guardrail responses are applied to output
- `_has_text_content()`: Customize text content detection

## Testing

Comprehensive tests are available in `tests/llm_translation/test_openai_responses_guardrail_handler.py`:

```bash
pytest tests/llm_translation/test_openai_responses_guardrail_handler.py -v
```

## Implementation Details

- **Parallel Processing**: All text content is processed in parallel using `asyncio.gather()`
- **Mapping Tracking**: Uses tuples to track the location of each text segment for accurate replacement
- **Type Safety**: Handles both Pydantic objects and dict representations
- **Multimodal Support**: Properly handles mixed content with text and other media types

