import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Message Sanitization for Tool Calling for anthropic models

**Automatically fix common message formatting issues when using tool calling with `modify_params=True`**

LiteLLM can automatically sanitize messages to handle common issues that occur during tool calling workflows, especially when using OpenAI-compatible clients with providers that have strict message format requirements (like Anthropic Claude).

## Overview

When `litellm.modify_params = True` is enabled, LiteLLM automatically sanitizes messages to fix three common issues:

1. **Orphaned Tool Calls** - Assistant messages with tool_calls but missing tool results
2. **Orphaned Tool Results** - Tool messages that reference non-existent tool_call_ids
3. **Empty Message Content** - Messages with empty or whitespace-only text content

This ensures your tool calling workflows work seamlessly across different LLM providers without manual message validation.

## Why Message Sanitization?

Different LLM providers have varying requirements for message formats, especially during tool calling:

- **Anthropic Claude** requires every tool_call to have a corresponding tool result
- Some providers reject messages with empty content
- OpenAI-compatible clients may not always maintain perfect message consistency

Without sanitization, these issues cause API errors that interrupt your workflows. With `modify_params=True`, LiteLLM handles these edge cases automatically.

## Quick Start

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

# Enable automatic message sanitization
litellm.modify_params = True

# This will work even if messages have formatting issues
response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "What's the weather in Boston?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Boston"}'}
                }
            ]
            # Missing tool result - LiteLLM will add a dummy result automatically
        },
        {"role": "user", "content": "Thanks!"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
litellm_settings:
  modify_params: true  # Enable automatic message sanitization

model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
```

</TabItem>
</Tabs>

## Sanitization Cases

### Case A: Orphaned Tool Calls (Missing Tool Results)

**Problem:** An assistant message contains `tool_calls`, but no corresponding tool result messages follow.

**Solution:** LiteLLM automatically adds dummy tool result messages for any missing tool results.

**Example:**

```python
import litellm
litellm.modify_params = True

# Messages with orphaned tool calls
messages = [
    {"role": "user", "content": "Search for Python tutorials"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "Python tutorials"}'}
            }
        ]
    },
    # Missing tool result here!
    {"role": "user", "content": "What about JavaScript?"}
]

# LiteLLM automatically adds:
# {
#     "role": "tool",
#     "tool_call_id": "call_abc123",
#     "content": "[System: Tool execution skipped/interrupted by user. No result provided for tool 'web_search'.]"
# }

response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=messages,
    tools=[...]
)
```

**When this happens:**
- User interrupts tool execution
- Client loses tool results due to network issues
- Conversation flow changes before tool completes
- Multi-turn conversations where tools are optional

### Case B: Orphaned Tool Results (Invalid tool_call_id)

**Problem:** A tool message references a `tool_call_id` that doesn't exist in any previous assistant message.

**Solution:** LiteLLM automatically removes these orphaned tool result messages.

**Example:**

```python
import litellm
litellm.modify_params = True

# Messages with orphaned tool result
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {
        "role": "tool",
        "tool_call_id": "call_nonexistent",  # This tool_call_id doesn't exist!
        "content": "Some result"
    }
]

# LiteLLM automatically removes the orphaned tool message

response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=messages
)
```

**When this happens:**
- Message history is manually edited
- Tool results are duplicated or mismatched
- Conversation state is restored incorrectly
- Messages are merged from different conversations

### Case C: Empty Message Content

**Problem:** User or assistant messages have empty or whitespace-only content.

**Solution:** LiteLLM replaces empty content with a system placeholder message.

**Example:**

```python
import litellm
litellm.modify_params = True

# Messages with empty content
messages = [
    {"role": "user", "content": ""},  # Empty content
    {"role": "assistant", "content": "   "},  # Whitespace only
]

# LiteLLM automatically replaces with:
# {"role": "user", "content": "[System: Empty message content sanitised to satisfy protocol]"}
# {"role": "assistant", "content": "[System: Empty message content sanitised to satisfy protocol]"}

response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=messages
)
```

**When this happens:**
- UI sends empty messages
- Content is stripped during preprocessing
- Placeholder messages in conversation history
- Edge cases in message construction

## Configuration

### Enable Globally

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

# Enable for all completion calls
litellm.modify_params = True
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
litellm_settings:
  modify_params: true
```

</TabItem>
<TabItem value="env" label="Environment Variable">

```bash
export LITELLM_MODIFY_PARAMS=True
```

</TabItem>
</Tabs>

### Enable Per-Request

```python
import litellm

# Enable only for specific requests
response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=messages,
    modify_params=True  # Override global setting
)
```

## Supported Providers

Message sanitization currently works with:

- ✅ Anthropic (Claude)

**Note:** While the sanitization logic is provider-agnostic, it is currently only applied in the Anthropic message transformation pipeline. Support for additional providers may be added in future releases.

## Implementation Details

### How It Works

The message sanitization process runs **before** messages are converted to provider-specific formats:

1. **Input:** OpenAI-format messages with potential issues
2. **Sanitization:** Three helper functions process the messages:
   - `_sanitize_empty_text_content()` - Fixes empty content
   - `_add_missing_tool_results()` - Adds dummy tool results
   - `_is_orphaned_tool_result()` - Identifies orphaned results
3. **Output:** Clean, provider-compatible messages

### Code Reference

The sanitization logic is implemented in:
- `litellm/litellm_core_utils/prompt_templates/factory.py`
- Function: `sanitize_messages_for_tool_calling()`

### Logging

When sanitization occurs, LiteLLM logs debug messages:

```python
import litellm
litellm.set_verbose = True  # Enable debug logging

# You'll see logs like:
# "_add_missing_tool_results: Found 1 orphaned tool calls. Adding dummy tool results."
# "_is_orphaned_tool_result: Found orphaned tool result with tool_call_id=call_123"
# "_sanitize_empty_text_content: Replaced empty text content in user message"
```

## Best Practices

### 1. Enable for Production Workflows

```python
# Recommended for production
litellm.modify_params = True

# Ensures robust handling of edge cases
response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=messages,
    tools=tools
)
```

### 2. Preserve Tool Results When Possible

While sanitization handles missing tool results, it's better to provide actual results:

```python
# Good: Provide actual tool results
messages = [
    {"role": "user", "content": "Search for Python"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_123", "content": "Actual search results"}
]

# Fallback: Sanitization adds dummy result if missing
messages = [
    {"role": "user", "content": "Search for Python"},
    {"role": "assistant", "tool_calls": [...]},
    # Missing tool result - sanitization adds dummy
]
```

### 3. Monitor Sanitization Events

Use logging to track when sanitization occurs:

```python
import litellm
import logging

# Enable debug logging
litellm.set_verbose = True
logging.basicConfig(level=logging.DEBUG)

# Track sanitization events in your application
response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=messages
)
```

### 4. Test Edge Cases

Ensure your application handles sanitized messages correctly:

```python
import litellm
litellm.modify_params = True

# Test orphaned tool calls
test_messages = [
    {"role": "user", "content": "Test"},
    {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]},
    {"role": "user", "content": "Continue"}  # No tool result
]

response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=test_messages,
    tools=[...]
)

# Verify the response handles the dummy tool result appropriately
```

## Related Features

- **[Drop Params](./drop_params.md)** - Drop unsupported parameters for specific providers
- **[Message Trimming](./message_trimming.md)** - Trim messages to fit token limits
- **[Function Calling](./function_call.md)** - Complete guide to tool/function calling
- **[Reasoning Content](../reasoning_content.md)** - Extended thinking with tool calling

## Troubleshooting

### Sanitization Not Working

**Issue:** Messages still cause errors despite `modify_params=True`

**Solution:**
1. Verify `modify_params` is enabled:
   ```python
   import litellm
   print(litellm.modify_params)  # Should be True
   ```

2. Check if the issue is provider-specific:
   ```python
   litellm.set_verbose = True  # Enable debug logging
   ```

3. Ensure you're using a recent version of LiteLLM:
   ```bash
   pip install --upgrade litellm
   ```

### Unexpected Dummy Tool Results

**Issue:** Dummy tool results appear when you expect actual results

**Cause:** Tool result messages are missing or have incorrect `tool_call_id`

**Solution:**
1. Verify tool result messages have correct `tool_call_id`:
   ```python
   # Correct
   {"role": "tool", "tool_call_id": "call_123", "content": "result"}
   
   # Incorrect - will be treated as orphaned
   {"role": "tool", "tool_call_id": "wrong_id", "content": "result"}
   ```

2. Ensure tool results immediately follow assistant messages with tool_calls

### Performance Impact

**Issue:** Concerned about performance overhead

**Details:** Message sanitization has minimal performance impact:
- Runs in O(n) time where n = number of messages
- Only processes messages when `modify_params=True`
- Typically adds < 1ms to request processing time

## FAQ

**Q: Does sanitization modify my original messages?**

A: No, sanitization creates a new list of messages. Your original messages remain unchanged.

**Q: Can I disable specific sanitization cases?**

A: Currently, all three cases are handled together when `modify_params=True`. To disable sanitization entirely, set `modify_params=False`.

**Q: What happens to the dummy tool results?**

A: Dummy tool results are sent to the LLM provider along with other messages. The model sees them as regular tool results with informative error messages.

**Q: Does this work with streaming?**

A: Yes, message sanitization works with both streaming and non-streaming requests.

**Q: Is this related to `drop_params`?**

A: No, they're separate features:
- `modify_params` - Modifies/fixes message content and structure
- `drop_params` - Removes unsupported API parameters

Both can be enabled simultaneously.

## See Also

- [Reasoning Content with Tool Calling](../reasoning_content.md)
- [Function Calling Guide](./function_call.md)
- [Bedrock Provider Documentation](../providers/bedrock.md)
- [Anthropic Provider Documentation](../providers/anthropic.md)
