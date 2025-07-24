# LiteLLM Google GenAI Interface

Interface to interact with Google GenAI Functions in the native Google interface format.

## Overview

This module provides a native interface to Google's Generative AI API, allowing you to use Google's content generation capabilities with both streaming and non-streaming modes, in both synchronous and asynchronous contexts.

## Available Functions

### Non-Streaming Functions

- `generate_content()` - Synchronous content generation
- `agenerate_content()` - Asynchronous content generation

### Streaming Functions

- `generate_content_stream()` - Synchronous streaming content generation
- `agenerate_content_stream()` - Asynchronous streaming content generation

## Usage Examples

### Basic Non-Streaming Usage

```python
from litellm.google_genai import generate_content, agenerate_content
from google.genai.types import ContentDict, PartDict

# Synchronous usage
contents = ContentDict(
    parts=[
        PartDict(text="Hello, can you tell me a short joke?")
    ],
)

response = generate_content(
    contents=contents,
    model="gemini-pro",  # or your preferred model
    # Add other model-specific parameters as needed
)

print(response)
```

### Async Non-Streaming Usage

```python
import asyncio
from litellm.google_genai import agenerate_content
from google.genai.types import ContentDict, PartDict

async def main():
    contents = ContentDict(
        parts=[
            PartDict(text="Hello, can you tell me a short joke?")
        ],
    )
    
    response = await agenerate_content(
        contents=contents,
        model="gemini-pro",
        # Add other model-specific parameters as needed
    )
    
    print(response)

# Run the async function
asyncio.run(main())
```

### Streaming Usage

```python
from litellm.google_genai import generate_content_stream
from google.genai.types import ContentDict, PartDict

# Synchronous streaming
contents = ContentDict(
    parts=[
        PartDict(text="Tell me a story about space exploration")
    ],
)

for chunk in generate_content_stream(
    contents=contents,
    model="gemini-pro",
):
    print(f"Chunk: {chunk}")
```

### Async Streaming Usage

```python
import asyncio
from litellm.google_genai import agenerate_content_stream
from google.genai.types import ContentDict, PartDict

async def main():
    contents = ContentDict(
        parts=[
            PartDict(text="Tell me a story about space exploration")
        ],
    )
    
    async for chunk in agenerate_content_stream(
        contents=contents,
        model="gemini-pro",
    ):
        print(f"Async chunk: {chunk}")

asyncio.run(main())
```


## Testing

This module includes comprehensive tests covering:
- Sync and async non-streaming requests
- Sync and async streaming requests
- Response validation
- Error handling scenarios

See `tests/unified_google_tests/base_google_test.py` for test implementation examples.