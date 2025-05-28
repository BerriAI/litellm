# Chat Provider for OpenAI Responses API

The `chat` provider enables legacy tools and applications to seamlessly use OpenAI's new responses-only models through the familiar `/chat/completions` endpoint. This provider automatically handles the bidirectional transformation between Chat Completion API and Responses API formats.

## Overview

OpenAI has started releasing models that are only available through their Responses API (e.g., `gpt-4o` with enhanced reasoning). However, many existing tools and applications are built around the Chat Completions API. The `chat` provider bridges this gap by:

1. **Converting** Chat Completion requests to Responses API format
2. **Proxying** to OpenAI's `/responses` endpoint  
3. **Transforming** responses back to Chat Completion format
4. **Supporting** all advanced features like streaming, function calling, and session management

## Configuration

### Basic Setup

```yaml
model_list:
  - model_name: gpt-4o-via-responses
    litellm_params:
      model: chat/gpt-4o  # Use "chat/" prefix
      api_base: https://api.openai.com/v1  # OpenAI base URL
      api_key: os.environ/OPENAI_API_KEY
      custom_llm_provider: chat  # Explicitly specify chat provider
```

### Advanced Configuration

```yaml
model_list:
  - model_name: gpt-4o-responses-advanced
    litellm_params:
      model: chat/gpt-4o
      api_base: https://api.openai.com/v1
      api_key: os.environ/OPENAI_API_KEY
      custom_llm_provider: chat
      # Optional: Enable verbose logging
      set_verbose: true
      
  # Alternative model for comparison
  - model_name: gpt-4o-regular
    litellm_params:
      model: openai/gpt-4o  # Direct chat completions
      api_key: os.environ/OPENAI_API_KEY
```

## Usage Examples

### Basic Chat Completion

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:4000",  # LiteLLM proxy
    api_key="your-key"
)

response = client.chat.completions.create(
    model="gpt-4o-via-responses",
    messages=[
        {"role": "user", "content": "Hello! How are you?"}
    ]
)

print(response.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o-via-responses",
    messages=[
        {"role": "user", "content": "Write a short poem"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o-via-responses",
    messages=[
        {"role": "user", "content": "What's the weather in NYC?"}
    ],
    tools=tools,
    tool_choice="auto"
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

### System Messages (Instructions)

```python
response = client.chat.completions.create(
    model="gpt-4o-via-responses",
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful coding assistant."
        },
        {
            "role": "user", 
            "content": "Explain Python list comprehensions"
        }
    ]
)
```

### Session Management

For conversational flows with session context:

```python
# First request
response1 = client.chat.completions.create(
    model="gpt-4o-via-responses",
    messages=[
        {"role": "user", "content": "Remember that my name is Alice"}
    ]
)

# Continue session using previous_response_id
response2 = client.chat.completions.create(
    model="gpt-4o-via-responses",
    messages=[
        {"role": "user", "content": "What's my name?"}
    ],
    previous_response_id=response1.id  # Link to previous response
)
```

## Supported Parameters

### Standard Chat Completion Parameters

All standard OpenAI Chat Completion parameters are supported:

- `messages` - Conversation messages
- `model` - Model name (use `chat/` prefix)
- `max_tokens` - Maximum response tokens
- `temperature` - Randomness (0-2)
- `top_p` - Nucleus sampling
- `stream` - Enable streaming
- `tools` - Function definitions
- `tool_choice` - Tool selection strategy
- `user` - User identifier
- `metadata` - Request metadata

### Responses API Specific Parameters

Additional parameters unique to the Responses API:

- `previous_response_id` - Session management
- `instructions` - System-level instructions (auto-extracted from system messages)

## Feature Support

| Feature | Status | Notes |
|---------|--------|-------|
| ✅ Basic Chat | Supported | Full bidirectional transformation |
| ✅ Streaming | Supported | Real-time response streaming |
| ✅ Function Calling | Supported | Complete tool integration |
| ✅ System Messages | Supported | Converted to `instructions` |
| ✅ Multimodal | Supported | Text, images, etc. |
| ✅ Session Management | Supported | Via `previous_response_id` |
| ✅ Token Counting | Supported | Accurate usage tracking |
| ✅ Error Handling | Supported | Proper error mapping |

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Legacy Tool   │───▶│ Chat Provider │───▶│ OpenAI Responses│
│ /chat/completions│    │ (Transform)   │    │      API        │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Transform Back   │
                    │ to Chat Format   │
                    └──────────────────┘
```

The chat provider:
1. **Receives** standard Chat Completion requests
2. **Transforms** messages, tools, and parameters to Responses API format
3. **Sends** request to OpenAI's `/responses` endpoint
4. **Converts** Responses API response back to Chat Completion format
5. **Returns** familiar Chat Completion response

## Testing

### Start LiteLLM Proxy

```bash
# Using the provided test configuration
litellm --config test_chat_provider_config.yaml --port 4000
```

### Run Test Suite

```bash
# Run the comprehensive test script
python test_chat_provider.py
```

### Manual Testing

```bash
# Basic test
curl -X POST "http://localhost:4000/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o-via-responses",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming test
curl -X POST "http://localhost:4000/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4o-via-responses",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure you're using the `chat/` prefix: `chat/gpt-4o`
   - Check your configuration file

2. **API Key Errors**
   - Verify your OpenAI API key has access to the Responses API
   - Some models may require special access

3. **Transformation Errors**
   - Check the logs for detailed error messages
   - All exceptions are now logged with full stack traces

4. **Session Management Issues**
   - Ensure `previous_response_id` is valid
   - Session handling will fail gracefully if there are issues

### Debug Mode

Enable verbose logging for detailed debugging:

```yaml
litellm_settings:
  set_verbose: true
  json_logs: true
  log_level: DEBUG
```

### Error Logging

All exceptions are logged with full details:
- JSON parsing errors from responses API
- Session management failures  
- Transformation errors
- API communication issues

## Migration Guide

### From Direct Responses API

If you're currently using the Responses API directly:

**Before:**
```python
# Direct responses API call
response = openai_client.responses.create(
    model="gpt-4o",
    input=[{"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]}],
    instructions="You are helpful"
)
```

**After:**
```python
# Via chat provider
response = client.chat.completions.create(
    model="chat/gpt-4o",  # Use chat provider
    messages=[
        {"role": "system", "content": "You are helpful"},  # Auto-converted to instructions
        {"role": "user", "content": "Hello"}
    ]
)
```

### From Regular Chat Completions

Minimal changes required:

**Before:**
```python
response = client.chat.completions.create(
    model="gpt-4o",  # Regular model
    messages=[...]
)
```

**After:**
```python
response = client.chat.completions.create(
    model="chat/gpt-4o",  # Chat provider model
    messages=[...]  # Same messages
)
```

## Benefits

1. **Legacy Compatibility** - Existing tools work without modification
2. **Advanced Features** - Access to responses-only models
3. **Seamless Migration** - Minimal code changes required
4. **Full Feature Parity** - Streaming, functions, sessions all supported
5. **Transparent Operation** - Works exactly like regular chat completions