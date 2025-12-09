# Adding Guardrail Support to Endpoints

This guide explains how to add guardrail translation support to new LiteLLM endpoints (e.g., Chat Completions, Responses API, etc.).

## When to Add Guardrail Support

Add guardrail support when:
- You're creating a new LiteLLM endpoint (e.g., a new API format)
- You want to enable guardrails for an existing endpoint that doesn't support them
- You need custom text extraction logic for a specific message format

## Directory Structure

Guardrail handlers follow this structure:

```
litellm/llms/{provider}/{endpoint}/guardrail_translation/
├── __init__.py          # Exports handler and registers call types
├── handler.py           # Main handler implementation
└── README.md            # Documentation (optional but recommended)
```

### Example Structures

**OpenAI Chat Completions:**
```
litellm/llms/openai/chat/guardrail_translation/
├── __init__.py
├── handler.py
└── README.md
```

**OpenAI Responses API:**
```
litellm/llms/openai/responses/guardrail_translation/
├── __init__.py
├── handler.py
└── README.md
```

**Anthropic Messages:**
```
litellm/llms/anthropic/chat/guardrail_translation/
├── __init__.py
└── handler.py
```

## Step-by-Step Implementation

### Step 1: Create the Handler Class

Create `handler.py` that inherits from `BaseTranslation`:

```python
"""
{Provider} {Endpoint} Handler for Unified Guardrails

This module provides guardrail translation support for {Provider}'s {Endpoint} format.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.utils import ModelResponse  # Or appropriate response type


class MyEndpointHandler(BaseTranslation):
    """
    Handler for processing {Endpoint} with guardrails.

    This class provides methods to:
    1. Process input (pre-call hook)
    2. Process output response (post-call hook)
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process input by applying guardrails to text content.
        
        Args:
            data: Request data dictionary
            guardrail_to_apply: The guardrail instance to apply
            
        Returns:
            Modified data with guardrails applied
        """
        # Your implementation here
        pass

    async def process_output_response(
        self,
        response: Any,  # Use appropriate response type
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process output response by applying guardrails to text content.
        
        Args:
            response: API response object
            guardrail_to_apply: The guardrail instance to apply
            
        Returns:
            Modified response with guardrails applied
        """
        # Your implementation here
        pass
```

### Step 2: Implement Core Methods

#### A. Process Input Messages

Extract text from input, apply guardrails, and map back:

```python
async def process_input_messages(
    self,
    data: dict,
    guardrail_to_apply: "CustomGuardrail",
) -> Any:
    """Process input messages by applying guardrails to text content."""
    # 1. Get input data from request
    messages = data.get("messages")  # or appropriate field
    if messages is None:
        return data

    # 2. Extract text and create tasks
    tasks = []
    task_mappings: List[Tuple[int, Optional[int]]] = []
    
    for msg_idx, message in enumerate(messages):
        await self._extract_input_text_and_create_tasks(
            message=message,
            msg_idx=msg_idx,
            tasks=tasks,
            task_mappings=task_mappings,
            guardrail_to_apply=guardrail_to_apply,
        )

    # 3. Run all guardrail tasks in parallel
    if tasks:
        responses = await asyncio.gather(*tasks)

        # 4. Map responses back to original structure
        await self._apply_guardrail_responses_to_input(
            messages=messages,
            responses=responses,
            task_mappings=task_mappings,
        )

    return data
```

#### B. Process Output Response

Extract text from response, apply guardrails, and update:

```python
async def process_output_response(
    self,
    response: "ModelResponse",
    guardrail_to_apply: "CustomGuardrail",
) -> Any:
    """Process output response by applying guardrails to text content."""
    # 1. Check if response has text to process
    if not self._has_text_content(response):
        return response

    # 2. Extract text and create tasks
    tasks = []
    task_mappings: List[Tuple[int, Optional[int]]] = []
    
    for idx, item in enumerate(response.choices):  # or appropriate field
        await self._extract_output_text_and_create_tasks(
            item=item,
            idx=idx,
            tasks=tasks,
            task_mappings=task_mappings,
            guardrail_to_apply=guardrail_to_apply,
        )

    # 3. Run all guardrail tasks in parallel
    if tasks:
        responses = await asyncio.gather(*tasks)

        # 4. Update response with guardrailed text
        await self._apply_guardrail_responses_to_output(
            response=response,
            responses=responses,
            task_mappings=task_mappings,
        )

    return response
```

### Step 3: Create Helper Methods

Implement helper methods for text extraction and mapping:

```python
async def _extract_input_text_and_create_tasks(
    self,
    message: Dict[str, Any],
    msg_idx: int,
    tasks: List,
    task_mappings: List[Tuple[int, Optional[int]]],
    guardrail_to_apply: "CustomGuardrail",
) -> None:
    """Extract text content from a message and create guardrail tasks."""
    content = message.get("content")
    if content is None:
        return

    if isinstance(content, str):
        # Simple string content
        tasks.append(guardrail_to_apply.apply_guardrail(text=content))
        task_mappings.append((msg_idx, None))
    elif isinstance(content, list):
        # List content (e.g., multimodal)
        for content_idx, content_item in enumerate(content):
            if isinstance(content_item, dict):
                text_str = content_item.get("text")
                if text_str:
                    tasks.append(guardrail_to_apply.apply_guardrail(text=text_str))
                    task_mappings.append((msg_idx, int(content_idx)))

async def _apply_guardrail_responses_to_input(
    self,
    messages: List[Dict[str, Any]],
    responses: List[str],
    task_mappings: List[Tuple[int, Optional[int]]],
) -> None:
    """Apply guardrail responses back to input messages."""
    for task_idx, guardrail_response in enumerate(responses):
        msg_idx, content_idx = task_mappings[task_idx]
        
        if content_idx is None:
            # String content
            messages[msg_idx]["content"] = guardrail_response
        else:
            # List content
            messages[msg_idx]["content"][content_idx]["text"] = guardrail_response

def _has_text_content(self, response: Any) -> bool:
    """Check if response has any text content to process."""
    # Implement based on your response structure
    return True  # or appropriate logic
```

### Step 4: Register the Handler

Create `__init__.py` to register the handler with call types:

```python
"""My Endpoint handler for Unified Guardrails."""

from litellm.llms.{provider}/{endpoint}/guardrail_translation.handler import (
    MyEndpointHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.my_endpoint: MyEndpointHandler,
    CallTypes.amy_endpoint: MyEndpointHandler,  # async version if applicable
}

__all__ = ["guardrail_translation_mappings"]
```

**Important:** Make sure your `CallTypes` are defined in `litellm/types/utils.py`.

### Step 5: Add Documentation

Create `README.md` with usage examples and format details:

```markdown
# {Provider} {Endpoint} Guardrail Translation Handler

Handler for processing {Provider}'s {Endpoint} with guardrails.

## Overview

This handler processes {Endpoint} input/output by:
1. Extracting text from messages/responses
2. Applying guardrails to text content
3. Mapping guardrailed text back to original structure

## Data Format

### Input Format
```json
{
  "field": "value",
  "messages": [...]
}
```

### Output Format
```json
{
  "field": "value",
  "output": [...]
}
```

## Usage

The handler is automatically discovered and applied when guardrails are used with this endpoint.

```bash
curl -X POST 'http://localhost:4000/{my_endpoint}' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer your-api-key' \
-d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "guardrails": ["test"]
}'

```
## Extension

Override these methods to customize behavior:
- `_extract_input_text_and_create_tasks()`: Custom text extraction
- `_apply_guardrail_responses_to_input()`: Custom response mapping
- `_has_text_content()`: Custom content detection
```

### Step 6: Add Unit Tests

Create comprehensive tests in `tests/test_litellm/llms/{provider}/{endpoint}/`:

```python
"""
Unit tests for {Provider} {Endpoint} Guardrail Translation Handler
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.{provider}.{endpoint}.guardrail_translation.handler import (
    MyEndpointHandler,
)
from litellm.types.utils import CallTypes


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing"""
    
    async def apply_guardrail(self, text: str) -> str:
        return f"{text} [GUARDRAILED]"


class TestHandlerDiscovery:
    """Test that the handler is properly discovered"""
    
    def test_handler_discovered(self):
        handler_class = get_guardrail_translation_mapping(CallTypes.my_endpoint)
        assert handler_class == MyEndpointHandler


class TestInputProcessing:
    """Test input processing functionality"""
    
    @pytest.mark.asyncio
    async def test_process_simple_input(self):
        handler = MyEndpointHandler()
        guardrail = MockGuardrail(guardrail_name="test")
        
        data = {"messages": [{"role": "user", "content": "Hello"}]}
        result = await handler.process_input_messages(data, guardrail)
        
        assert result["messages"][0]["content"] == "Hello [GUARDRAILED]"


class TestOutputProcessing:
    """Test output processing functionality"""
    
    @pytest.mark.asyncio
    async def test_process_simple_output(self):
        handler = MyEndpointHandler()
        guardrail = MockGuardrail(guardrail_name="test")
        
        # Create mock response
        response = create_mock_response()
        result = await handler.process_output_response(response, guardrail)
        
        # Assert guardrail was applied
        assert "GUARDRAILED" in get_response_text(result)
```

## Support

For questions or issues:
- Check existing handler implementations for examples
- Review the base translation class documentation
- Create an issue on GitHub with the `guardrails` label

