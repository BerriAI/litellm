# LiteLLM Bug Fix: Qwen3 Tool Calls Dropped

## Issue
https://github.com/BerriAI/litellm/issues/18922

## Problem Summary
When using qwen3 models through LiteLLM's Ollama provider, `tool_calls` are dropped from the response. The response contains only `content: "{}"` while valid `tool_calls` from Ollama are lost.

**Root Cause**: Qwen3 includes a `thinking` field in its responses that qwen2.5 does not. The Ollama response handler doesn't properly handle responses that have both `thinking` and `tool_calls`.

## Files to Investigate

1. **`litellm/llms/ollama/completion/transformation.py`** - Ollama response transformation
2. **`litellm/llms/ollama_chat.py`** - Ollama chat handler (legacy)
3. **`litellm/llms/ollama/chat/transformation.py`** - Ollama chat transformation

## Expected Fix Location

Look for where the Ollama response message is parsed. The code likely does something like:
```python
content = message.get("content", "")
```

But doesn't extract:
```python
tool_calls = message.get("tool_calls", [])
thinking = message.get("thinking", "")  # qwen3 specific
```

## Test Cases

### Test 1: Qwen3 with tool_calls should work

```python
def test_ollama_qwen3_tool_calls():
    """Test that qwen3 tool_calls are properly forwarded."""
    import litellm

    response = litellm.completion(
        model="ollama/qwen3:14b",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }],
        api_base="http://localhost:11434"
    )

    # This currently fails - tool_calls is None
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) > 0
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
```

### Test 2: Mock Ollama response with thinking field

```python
def test_ollama_response_with_thinking_field():
    """Test that responses with 'thinking' field preserve tool_calls."""
    from litellm.llms.ollama.chat.transformation import OllamaChatConfig

    # Simulated Ollama response (what qwen3 returns)
    mock_ollama_response = {
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "Let me check the weather function...",
            "tool_calls": [{
                "id": "call_abc123",
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "Tokyo"}
                }
            }]
        },
        "done": True
    }

    # Transform to OpenAI format
    # The fix should ensure tool_calls are preserved
    result = transform_ollama_response(mock_ollama_response)

    assert result["choices"][0]["message"]["tool_calls"] is not None
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
```

### Test 3: Arguments should be JSON string (OpenAI format)

```python
def test_ollama_tool_call_arguments_are_stringified():
    """Ollama returns arguments as dict, OpenAI expects JSON string."""
    # Ollama returns: {"arguments": {"location": "Tokyo"}}
    # OpenAI expects: {"arguments": "{\"location\": \"Tokyo\"}"}

    # The fix should stringify the arguments
    assert isinstance(
        result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        str
    )
```

## Reproduction Commands

```bash
# Direct Ollama (works)
curl -s http://localhost:11434/api/chat -d '{
  "model": "qwen3:14b",
  "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}],
  "stream": false
}' | jq '.message.tool_calls'
# Returns: [{"function": {"name": "get_weather", "arguments": {"location": "Tokyo"}}}]

# Through LiteLLM (broken)
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Weather in Tokyo?"}], "tools": [...]}' \
  | jq '.choices[0].message'
# Returns: {"content": "{}", "role": "assistant"}  # tool_calls missing!
```

## Fix Checklist

- [ ] Find where Ollama response is transformed to OpenAI format
- [ ] Ensure `tool_calls` is extracted from `message.tool_calls`
- [ ] Handle the `thinking` field (either include it or ignore it, but don't let it break tool_calls)
- [ ] Stringify `arguments` dict to JSON string for OpenAI compatibility
- [ ] Add unit test for qwen3-style responses with `thinking` field
- [ ] Test with actual qwen3:14b model
