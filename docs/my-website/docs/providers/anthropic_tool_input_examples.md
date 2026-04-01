# Anthropic Tool Input Examples

Provide concrete examples of valid tool inputs to help Claude understand how to use your tools more effectively. This is particularly useful for complex tools with nested objects, optional parameters, or format-sensitive inputs.

:::info
Tool input examples is a beta feature. LiteLLM automatically detects tools with the `input_examples` field and adds the appropriate beta header based on your provider:

- **Anthropic API & Microsoft Foundry**: `advanced-tool-use-2025-11-20`
- **Amazon Bedrock**: `advanced-tool-use-2025-11-20` (Claude Opus 4.5 only)
- **Google Cloud Vertex AI**: Not supported

You don't need to manually specify beta headers—LiteLLM handles this automatically.
:::

## When to Use Input Examples

Input examples are most helpful for:

- **Complex nested objects**: Tools with deeply nested parameter structures
- **Optional parameters**: Showing when optional parameters should be included
- **Format-sensitive inputs**: Demonstrating expected formats (dates, addresses, etc.)
- **Enum values**: Illustrating valid enum choices in context
- **Edge cases**: Showing how to handle special cases

:::tip
**Prioritize descriptions first!** Clear, detailed tool descriptions are more important than examples. Use `input_examples` as a supplement for complex tools where descriptions alone may not be sufficient.
:::

## Quick Start

Add an `input_examples` field to your tool definition with an array of example input objects:

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        }
                    },
                    "required": ["location"]
                }
            },
            "input_examples": [
                {
                    "location": "San Francisco, CA",
                    "unit": "fahrenheit"
                },
                {
                    "location": "Tokyo, Japan",
                    "unit": "celsius"
                },
                {
                    "location": "New York, NY"  # 'unit' is optional
                }
            ]
        }
    ]
)

print(response)
```

## How It Works

When you provide `input_examples`:

1. **LiteLLM detects** the `input_examples` field in your tool definition
2. **Beta header added automatically**: The `advanced-tool-use-2025-11-20` header is injected
3. **Examples included in prompt**: Anthropic includes the examples alongside your tool schema
4. **Claude learns patterns**: The model uses examples to understand proper tool usage
5. **Better tool calls**: Claude makes more accurate tool calls with correct parameter formats

## Example Formats

### Simple Tool with Examples

```python
{
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Email address"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["to", "subject", "body"]
        }
    },
    "input_examples": [
        {
            "to": "user@example.com",
            "subject": "Meeting Reminder",
            "body": "Don't forget our meeting tomorrow at 2 PM."
        },
        {
            "to": "team@company.com",
            "subject": "Weekly Update",
            "body": "Here's this week's progress report..."
        }
    ]
}
```

### Complex Nested Objects

```python
{
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Create a new calendar event",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "time": {"type": "string"}
                    }
                },
                "attendees": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                            "optional": {"type": "boolean"}
                        }
                    }
                }
            },
            "required": ["title", "start"]
        }
    },
    "input_examples": [
        {
            "title": "Team Standup",
            "start": {
                "date": "2025-01-15",
                "time": "09:00"
            },
            "attendees": [
                {"email": "alice@example.com", "optional": False},
                {"email": "bob@example.com", "optional": True}
            ]
        },
        {
            "title": "Lunch Break",
            "start": {
                "date": "2025-01-15",
                "time": "12:00"
            }
            # No attendees - showing optional field
        }
    ]
}
```

### Format-Sensitive Parameters

```python
{
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Search for available flights",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Airport code"},
                "destination": {"type": "string", "description": "Airport code"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "passengers": {"type": "integer"}
            },
            "required": ["origin", "destination", "date"]
        }
    },
    "input_examples": [
        {
            "origin": "SFO",
            "destination": "JFK",
            "date": "2025-03-15",
            "passengers": 2
        },
        {
            "origin": "LAX",
            "destination": "ORD",
            "date": "2025-04-20",
            "passengers": 1
        }
    ]
}
```

## Requirements and Limitations

### Schema Validation

- Each example **must be valid** according to the tool's `input_schema`
- Invalid examples will return a **400 error** from Anthropic
- Validation happens server-side (LiteLLM passes examples through)

### Server-Side Tools Not Supported

Input examples are **only supported for user-defined tools**. The following server-side tools do NOT support `input_examples`:

- `web_search` (web search tool)
- `code_execution` (code execution tool)
- `computer_use` (computer use tool)
- `bash_tool` (bash execution tool)
- `text_editor` (text editor tool)

### Token Costs

Examples add to your prompt tokens:

- **Simple examples**: ~20-50 tokens per example
- **Complex nested objects**: ~100-200 tokens per example
- **Trade-off**: Higher token cost for better tool call accuracy

### Model Compatibility

Input examples work with all Claude models that support the `advanced-tool-use-2025-11-20` beta header:

- Claude Opus 4.5 (`claude-opus-4-5-20251101`)
- Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`)
- Claude Opus 4.1 (`claude-opus-4-1-20250805`)

:::note
On Google Cloud's Vertex AI and Amazon Bedrock, only Claude Opus 4.5 supports tool input examples.
:::

## Best Practices

### 1. Show Diverse Examples

Include examples that demonstrate different use cases:

```python
"input_examples": [
    {"location": "San Francisco, CA", "unit": "fahrenheit"},  # US city
    {"location": "Tokyo, Japan", "unit": "celsius"},          # International
    {"location": "New York, NY"}                              # Optional param omitted
]
```

### 2. Demonstrate Optional Parameters

Show when optional parameters should and shouldn't be included:

```python
"input_examples": [
    {
        "query": "machine learning",
        "filters": {"year": 2024, "category": "research"}  # With optional filters
    },
    {
        "query": "artificial intelligence"  # Without optional filters
    }
]
```

### 3. Illustrate Format Requirements

Make format expectations clear through examples:

```python
"input_examples": [
    {
        "phone": "+1-555-123-4567",  # Shows expected phone format
        "date": "2025-01-15",         # Shows date format (YYYY-MM-DD)
        "time": "14:30"               # Shows time format (HH:MM)
    }
]
```

### 4. Keep Examples Realistic

Use realistic, production-like examples rather than placeholder data:

```python
# ✅ Good - realistic examples
"input_examples": [
    {"email": "alice@company.com", "role": "admin"},
    {"email": "bob@company.com", "role": "user"}
]

# ❌ Bad - placeholder examples
"input_examples": [
    {"email": "test@test.com", "role": "role1"},
    {"email": "example@example.com", "role": "role2"}
]
```

### 5. Limit Example Count

Provide 2-5 examples per tool:

- **Too few** (1): May not show enough variation
- **Just right** (2-5): Demonstrates patterns without bloating tokens
- **Too many** (10+): Wastes tokens, diminishing returns

## Integration with Other Features

Input examples work seamlessly with other Anthropic tool features:

### With Tool Search

```python
{
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Execute a SQL query",
        "parameters": {...}
    },
    "defer_loading": True,  # Tool search
    "input_examples": [     # Input examples
        {"sql": "SELECT * FROM users WHERE id = 1"}
    ]
}
```

### With Programmatic Tool Calling

```python
{
    "type": "function",
    "function": {
        "name": "fetch_data",
        "description": "Fetch data from API",
        "parameters": {...}
    },
    "allowed_callers": ["code_execution_20250825"],  # Programmatic calling
    "input_examples": [                               # Input examples
        {"endpoint": "/api/users", "method": "GET"}
    ]
}
```

### All Features Combined

```python
{
    "type": "function",
    "function": {
        "name": "advanced_tool",
        "description": "A complex tool",
        "parameters": {...}
    },
    "defer_loading": True,                            # Tool search
    "allowed_callers": ["code_execution_20250825"],  # Programmatic calling
    "input_examples": [                               # Input examples
        {"param1": "value1", "param2": "value2"}
    ]
}
```

## Provider Support

LiteLLM supports input examples across the following Anthropic-compatible providers:

- **Standard Anthropic API** (`anthropic/claude-sonnet-4-5-20250929`) ✅
- **Azure Anthropic / Microsoft Foundry** (`azure/claude-sonnet-4-5-20250929`) ✅
- **Amazon Bedrock** (`bedrock/invoke/anthropic.claude-opus-4-5-20251101-v1:0`) ✅ (Opus 4.5 only)
- **Google Cloud Vertex AI** (`vertex_ai/claude-sonnet-4-5-20250929`) ❌ Not supported

The beta header (`advanced-tool-use-2025-11-20`) is automatically added when LiteLLM detects tools with the `input_examples` field.

## Troubleshooting

### "Invalid request" error with examples

**Problem**: Receiving 400 error when using input examples

**Solution**: Ensure each example is valid according to your `input_schema`:

```python
# Check that:
# 1. All required fields are present in examples
# 2. Field types match the schema
# 3. Enum values are valid
# 4. Nested objects follow the schema structure
```

### Examples not improving tool calls

**Problem**: Adding examples doesn't seem to help

**Solution**:
1. **Check descriptions first**: Ensure tool descriptions are detailed and clear
2. **Review example quality**: Make sure examples are realistic and diverse
3. **Verify schema**: Confirm examples actually match your schema
4. **Add more variation**: Include examples showing different use cases

### Token usage too high

**Problem**: Input examples consuming too many tokens

**Solution**:
1. **Reduce example count**: Use 2-3 examples instead of 5+
2. **Simplify examples**: Remove unnecessary fields from examples
3. **Consider descriptions**: If descriptions are clear, examples may not be needed

## When NOT to Use Input Examples

Skip input examples if:

- **Tool is simple**: Single parameter tools with clear descriptions
- **Schema is self-explanatory**: Well-structured schema with good descriptions
- **Token budget is tight**: Examples add 20-200 tokens each
- **Server-side tools**: web_search, code_execution, etc. don't support examples

## Related Features

- [Anthropic Tool Search](./anthropic_tool_search.md) - Dynamically discover and load tools on-demand
- [Anthropic Programmatic Tool Calling](./anthropic_programmatic_tool_calling.md) - Call tools from code execution
- [Anthropic Provider](./anthropic.md) - General Anthropic provider documentation

