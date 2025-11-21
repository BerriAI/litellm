---
slug: gemini_3
title: "DAY 0 Support: Gemini 3 on LiteLLM"
date: 2025-11-19T10:00:00
authors:
  - name: Sameer Kankute
    title: "SWE @ LiteLLM (LLM Translation)"
    url: https://in.linkedin.com/in/sameer-kankute
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQHB_loQYd5gjg/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1719137160975?e=1765411200&v=beta&t=c8396f--_lH6Fb_pVvx_jGholPfcl0bvwmNynbNdnII
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
tags: [gemini, day 0 support, llms]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info

This guide covers common questions and best practices for using `gemini-3-pro-preview` with LiteLLM Proxy and SDK.

:::

## Quick Start

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
from litellm import completion
import os

os.environ["GEMINI_API_KEY"] = "your-api-key"

response = completion(
    model="gemini/gemini-3-pro-preview",
    messages=[{"role": "user", "content": "Hello!"}],
    reasoning_effort="low"
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Add to config.yaml:**

```yaml
model_list:
  - model_name: gemini-3-pro-preview
    litellm_params:
      model: gemini/gemini-3-pro-preview
      api_key: os.environ/GEMINI_API_KEY
```

**2. Start proxy:**

```bash
litellm --config /path/to/config.yaml
```

**3. Make request:**

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-3-pro-preview",
    "messages": [{"role": "user", "content": "Hello!"}],
    "reasoning_effort": "low"
  }'
```

</TabItem>
</Tabs>

## Supported Endpoints

LiteLLM provides **full end-to-end support** for Gemini 3 Pro Preview on:

- ✅ `/v1/chat/completions` - OpenAI-compatible chat completions endpoint
- ✅ [`/v1/messages`](../../docs/anthropic_unified) - Anthropic-compatible messages endpoint

Both endpoints support:
- Streaming and non-streaming responses
- Function calling with thought signatures
- Multi-turn conversations
- All Gemini 3-specific features

## Thought Signatures

#### What are Thought Signatures?

Thought signatures are encrypted representations of the model's internal reasoning process. They're essential for maintaining context across multi-turn conversations, especially with function calling.

#### How Thought Signatures Work

1. **Automatic Extraction**: When Gemini 3 returns a function call, LiteLLM automatically extracts the `thought_signature` from the response
2. **Storage**: Thought signatures are stored in `provider_specific_fields.thought_signature` of tool calls
3. **Automatic Preservation**: When you include the assistant's message in conversation history, LiteLLM automatically preserves and returns thought signatures to Gemini

## Example: Multi-Turn Function Calling

#### Streaming with Thought Signatures

When using streaming mode with `stream_chunk_builder()`, thought signatures are now automatically preserved:

<Tabs>
<TabItem value="streaming" label="Streaming SDK">

```python
import os
import litellm
from litellm import completion

os.environ["GEMINI_API_KEY"] = "your-api-key"

MODEL = "gemini/gemini-3-pro-preview"

messages = [
    {"role": "system", "content": "You are a helpful assistant. Use the calculate tool."},
    {"role": "user", "content": "What is 2+2?"},
]

tools = [{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Calculate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
}]

print("Step 1: Sending request with stream=True...")
response = completion(
    model=MODEL,
    messages=messages,
    stream=True,
    tools=tools,
    reasoning_effort="low"
)

# Collect all chunks
chunks = []
for part in response:
    chunks.append(part)

# Reconstruct message using stream_chunk_builder
# Thought signatures are now preserved automatically!
full_response = litellm.stream_chunk_builder(chunks, messages=messages)
print(f"Full response: {full_response}")

assistant_msg = full_response.choices[0].message

# ✅ Thought signature is now preserved in provider_specific_fields
if assistant_msg.tool_calls and assistant_msg.tool_calls[0].provider_specific_fields:
    thought_sig = assistant_msg.tool_calls[0].provider_specific_fields.get("thought_signature")
    print(f"Thought signature preserved: {thought_sig is not None}")

# Append assistant message (includes thought signatures automatically)
messages.append(assistant_msg)

# Mock tool execution
messages.append({
    "role": "tool",
    "content": "4",
    "tool_call_id": assistant_msg.tool_calls[0].id
})

print("\nStep 2: Sending tool result back to model...")
response_2 = completion(
    model=MODEL,
    messages=messages,
    stream=True,
    tools=tools,
    reasoning_effort="low"
)

for part in response_2:
    if part.choices[0].delta.content:
        print(part.choices[0].delta.content, end="")
print()  # New line
```

**Key Points:**
- ✅ `stream_chunk_builder()` now preserves `provider_specific_fields` including thought signatures
- ✅ Thought signatures are automatically included when appending `assistant_msg` to conversation history
- ✅ Multi-turn conversations work seamlessly with streaming

</TabItem>
<TabItem value="sdk" label="Non-Streaming SDK">

```python
from openai import OpenAI
import json

client = OpenAI(api_key="sk-1234", base_url="http://localhost:4000")

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
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

# Step 1: Initial request
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]

response = client.chat.completions.create(
    model="gemini-3-pro-preview",
    messages=messages,
    tools=tools,
    reasoning_effort="low"
)

# Step 2: Append assistant message (thought signatures automatically preserved)
messages.append(response.choices[0].message)

# Step 3: Execute tool and append result
for tool_call in response.choices[0].message.tool_calls:
    if tool_call.function.name == "get_weather":
        result = {"temperature": 30, "unit": "celsius"}
        messages.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id
        })

# Step 4: Follow-up request (thought signatures automatically included)
response2 = client.chat.completions.create(
    model="gemini-3-pro-preview",
    messages=messages,
    tools=tools,
    reasoning_effort="low"
)

print(response2.choices[0].message.content)
```

**Key Points:**
- ✅ Thought signatures are automatically extracted from `response.choices[0].message.tool_calls[].provider_specific_fields.thought_signature`
- ✅ When you append `response.choices[0].message` to your conversation history, thought signatures are automatically preserved
- ✅ You don't need to manually extract or manage thought signatures

</TabItem>
<TabItem value="proxy" label="cURL">

```bash
# Step 1: Initial request
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-3-pro-preview",
    "messages": [
      {"role": "user", "content": "What'\''s the weather in Tokyo?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "reasoning_effort": "low"
  }'
```

**Response includes thought signature:**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Tokyo\"}"
        },
        "provider_specific_fields": {
          "thought_signature": "CpcHAdHtim9+q4rstcbvQC0ic4x1/vqQlCJWgE+UZ6dTLYGHMMBkF/AxqL5UmP6SY46uYC8t4BTFiXG5zkw6EMJ..."
        }
      }]
    }
  }]
}
```

```bash
# Step 2: Follow-up request (include assistant message with thought signature)
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-3-pro-preview",
    "messages": [
      {"role": "user", "content": "What'\''s the weather in Tokyo?"},
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [{
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"Tokyo\"}"
          },
          "provider_specific_fields": {
            "thought_signature": "CpcHAdHtim9+q4rstcbvQC0ic4x1/vqQlCJWgE+UZ6dTLYGHMMBkF/AxqL5UmP6SY46uYC8t4BTFiXG5zkw6EMJ..."
          }
        }]
      },
      {
        "role": "tool",
        "content": "{\"temperature\": 30, \"unit\": \"celsius\"}",
        "tool_call_id": "call_abc123"
      }
    ],
    "tools": [...],
    "reasoning_effort": "low"
  }'
```

</TabItem>
</Tabs>

#### Important Notes on Thought Signatures

1. **Automatic Handling**: LiteLLM automatically extracts and preserves thought signatures. You don't need to manually manage them.

2. **Parallel Function Calls**: When the model makes parallel function calls, only the **first function call** has a thought signature.

3. **Sequential Function Calls**: In multi-step function calling, each step's first function call has its own thought signature that must be preserved.

4. **Required for Context**: Thought signatures are essential for maintaining reasoning context. Without them, the model may lose context of its previous reasoning.

## Conversation History: Switching from Non-Gemini-3 Models

#### Common Question: Will switching from a non-Gemini-3 model to Gemini-3 break conversation history?

**Answer: No!** LiteLLM automatically handles this by adding dummy thought signatures when needed.

#### How It Works

When you switch from a model that doesn't use thought signatures (e.g., `gemini-2.5-flash`) to Gemini 3, LiteLLM:

1. **Detects missing signatures**: Identifies assistant messages with tool calls that lack thought signatures
2. **Adds dummy signature**: Automatically injects a dummy thought signature (`skip_thought_signature_validator`) for compatibility
3. **Maintains conversation flow**: Your conversation history continues to work seamlessly

#### Example: Switching Models Mid-Conversation

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
from openai import OpenAI

client = OpenAI(api_key="sk-1234", base_url="http://localhost:4000")

# Step 1: Start with gemini-2.5-flash (no thought signatures)
messages = [{"role": "user", "content": "What's the weather?"}]

response1 = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=messages,
    tools=[...],
    reasoning_effort="low"
)

# Append assistant message (no tool call thought signature from gemini-2.5-flash)
messages.append(response1.choices[0].message)

# Step 2: Switch to gemini-3-pro-preview
# LiteLLM automatically adds dummy thought signature to the previous assistant message
response2 = client.chat.completions.create(
    model="gemini-3-pro-preview",  # 👈 Switched model
    messages=messages,  # 👈 Same conversation history
    tools=[...],
    reasoning_effort="low"
)

# ✅ Works seamlessly! No errors, no breaking changes
print(response2.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="cURL">

```bash
# Step 1: Start with gemini-2.5-flash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "What'\''s the weather?"}],
    "tools": [...],
    "reasoning_effort": "low"
  }'

# Step 2: Switch to gemini-3-pro-preview with same conversation history
# LiteLLM automatically handles the missing thought signature
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-3-pro-preview",  # 👈 Switched model
    "messages": [
      {"role": "user", "content": "What'\''s the weather?"},
      {
        "role": "assistant",
        "tool_calls": [...]  # 👈 No thought_signature from gemini-2.5-flash
      }
    ],
    "tools": [...],
    "reasoning_effort": "low"
  }'
# ✅ Works! LiteLLM adds dummy signature automatically
```

</TabItem>
</Tabs>

#### Dummy Signature Details

The dummy signature used is: `base64("skip_thought_signature_validator")`

This is the recommended approach by Google for handling conversation history from models that don't support thought signatures. It allows Gemini 3 to:
- Accept the conversation history without validation errors
- Continue the conversation seamlessly
- Maintain context across model switches

## Thinking Level Parameter

#### How `reasoning_effort` Maps to `thinking_level`

For Gemini 3 Pro Preview, LiteLLM automatically maps `reasoning_effort` to the new `thinking_level` parameter:

| `reasoning_effort` | `thinking_level` | Notes |
|-------------------|------------------|-------|
| `"minimal"` | `"low"` | Maps to low thinking level |
| `"low"` | `"low"` | Default for most use cases |
| `"medium"` | `"high"` | Medium not available yet, maps to high |
| `"high"` | `"high"` | Maximum reasoning depth |
| `"disable"` | `"low"` | Gemini 3 cannot fully disable thinking |
| `"none"` | `"low"` | Gemini 3 cannot fully disable thinking |

#### Default Behavior

If you don't specify `reasoning_effort`, LiteLLM automatically sets `thinking_level="low"` for Gemini 3 models, to avoid high costs. 

### Example Usage

<Tabs>
<TabItem value="sdk" label="Python SDK">

```python
from litellm import completion

# Low thinking level (faster, lower cost)
response = completion(
    model="gemini/gemini-3-pro-preview",
    messages=[{"role": "user", "content": "What's the weather?"}],
    reasoning_effort="low"  # Maps to thinking_level="low"
)

# High thinking level (deeper reasoning, higher cost)
response = completion(
    model="gemini/gemini-3-pro-preview",
    messages=[{"role": "user", "content": "Solve this complex math problem step by step."}],
    reasoning_effort="high"  # Maps to thinking_level="high"
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```bash
# Low thinking level
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-3-pro-preview",
    "messages": [{"role": "user", "content": "What'\''s the weather?"}],
    "reasoning_effort": "low"
  }'

# High thinking level
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gemini-3-pro-preview",
    "messages": [{"role": "user", "content": "Solve this complex problem."}],
    "reasoning_effort": "high"
  }'
```

</TabItem>
</Tabs>

## Important Notes

1. **Gemini 3 Cannot Disable Thinking**: Unlike Gemini 2.5 models, Gemini 3 cannot fully disable thinking. Even when you set `reasoning_effort="none"` or `"disable"`, it maps to `thinking_level="low"`.

2. **Temperature Recommendation**: For Gemini 3 models, LiteLLM defaults `temperature` to `1.0` and strongly recommends keeping it at this default. Setting `temperature < 1.0` can cause:
   - Infinite loops
   - Degraded reasoning performance
   - Failure on complex tasks

3. **Automatic Defaults**: If you don't specify `reasoning_effort`, LiteLLM automatically sets `thinking_level="low"` for optimal performance.

## Using with Claude Code CLI

You can use `gemini-3-pro-preview` with **Claude Code CLI** - Anthropic's command-line interface. This allows you to use Gemini 3 Pro Preview with Claude Code's native syntax and workflows.

### Setup

**1. Add Gemini 3 Pro Preview to your `config.yaml`:**

```yaml
model_list:
  - model_name: gemini-3-pro-preview
    litellm_params:
      model: gemini/gemini-3-pro-preview
      api_key: os.environ/GEMINI_API_KEY

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

**2. Set environment variables:**

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export LITELLM_MASTER_KEY="sk-1234567890"  # Generate a secure key
```

**3. Start LiteLLM Proxy:**

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

**4. Configure Claude Code to use LiteLLM Proxy:**

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
```

**5. Use Gemini 3 Pro Preview with Claude Code:**

```bash
# Claude Code will use gemini-3-pro-preview from your LiteLLM proxy
claude --model gemini-3-pro-preview

```

### Example Usage

Once configured, you can interact with Gemini 3 Pro Preview using Claude Code's native interface:

```bash
$ claude --model gemini-3-pro-preview
> Explain how thought signatures work in multi-turn conversations.

# Gemini 3 Pro Preview responds through Claude Code interface
```

### Benefits

- ✅ **Native Claude Code Experience**: Use Gemini 3 Pro Preview with Claude Code's familiar CLI interface
- ✅ **Unified Authentication**: Single API key for all models through LiteLLM proxy
- ✅ **Cost Tracking**: All usage tracked through LiteLLM's centralized logging
- ✅ **Seamless Model Switching**: Easily switch between Claude and Gemini models
- ✅ **Full Feature Support**: All Gemini 3 features (thought signatures, function calling, etc.) work through Claude Code

### Troubleshooting

**Claude Code not finding the model:**
- Ensure the model name in Claude Code matches exactly: `gemini-3-pro-preview`
- Verify your proxy is running: `curl http://0.0.0.0:4000/health`
- Check that `ANTHROPIC_BASE_URL` points to your LiteLLM proxy

**Authentication errors:**
- Verify `ANTHROPIC_AUTH_TOKEN` matches your LiteLLM master key
- Ensure `GEMINI_API_KEY` is set correctly
- Check LiteLLM proxy logs for detailed error messages

## Best Practices

#### 1. Always Include Thought Signatures in Conversation History

When building multi-turn conversations with function calling:

✅ **Do:**
```python
# Append the full assistant message (includes thought signatures)
messages.append(response.choices[0].message)
```

❌ **Don't:**
```python
# Don't manually construct assistant messages without thought signatures
messages.append({
    "role": "assistant",
    "tool_calls": [...]  # Missing thought signatures!
})
```

#### 2. Use Appropriate Thinking Levels

- **`reasoning_effort="low"`**: For simple queries, quick responses, cost optimization
- **`reasoning_effort="high"`**: For complex problems requiring deep reasoning

#### 3. Keep Temperature at Default

For Gemini 3 models, always use `temperature=1.0` (default). Lower temperatures can cause issues.

#### 4. Handle Model Switches Gracefully

When switching from non-Gemini-3 to Gemini-3:
- ✅ LiteLLM automatically handles missing thought signatures
- ✅ No manual intervention needed
- ✅ Conversation history continues seamlessly

## Troubleshooting

#### Issue: Missing Thought Signatures

**Symptom**: Error when including assistant messages in conversation history

**Solution**: Ensure you're appending the full assistant message from the response:
```python
messages.append(response.choices[0].message)  # ✅ Includes thought signatures
```

#### Issue: Conversation Breaks When Switching Models

**Symptom**: Errors when switching from gemini-2.5-flash to gemini-3-pro-preview

**Solution**: This should work automatically! LiteLLM adds dummy signatures. If you see errors, ensure you're using the latest LiteLLM version.

#### Issue: Infinite Loops or Poor Performance

**Symptom**: Model gets stuck or produces poor results

**Solution**: 
- Ensure `temperature=1.0` (default for Gemini 3)
- Check that `reasoning_effort` is set appropriately
- Verify you're using the correct model name: `gemini/gemini-3-pro-preview`

## Additional Resources

- [Gemini Provider Documentation](../gemini.md)
- [Thought Signatures Guide](../gemini.md#thought-signatures)
- [Reasoning Content Documentation](../../reasoning_content.md)
- [Function Calling Guide](../../function_calling.md)

