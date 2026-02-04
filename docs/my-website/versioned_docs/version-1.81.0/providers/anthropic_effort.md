import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Anthropic Effort Parameter

Control how many tokens Claude uses when responding with the `effort` parameter, trading off between response thoroughness and token efficiency.

## Overview

The `effort` parameter allows you to control how eager Claude is about spending tokens when responding to requests. This gives you the ability to trade off between response thoroughness and token efficiency, all with a single model.

**Note**: The effort parameter is currently in beta and only supported by Claude Opus 4.5. LiteLLM automatically adds the `effort-2025-11-24` beta header when:
- `reasoning_effort` parameter is provided (for Claude Opus 4.5 only)

For Claude Opus 4.5, `reasoning_effort="medium"`—both are automatically mapped to the correct format.

## How Effort Works

By default, Claude uses maximum effort—spending as many tokens as needed for the best possible outcome. By lowering the effort level, you can instruct Claude to be more conservative with token usage, optimizing for speed and cost while accepting some reduction in capability.

**Tip**: Setting `effort` to `"high"` produces exactly the same behavior as omitting the `effort` parameter entirely.

The effort parameter affects **all tokens** in the response, including:
- Text responses and explanations
- Tool calls and function arguments
- Extended thinking (when enabled)

This approach has two major advantages:
1. It doesn't require thinking to be enabled in order to use it.
2. It can affect all token spend including tool calls. For example, lower effort would mean Claude makes fewer tool calls.

This gives a much greater degree of control over efficiency.

## Effort Levels

| Level | Description | Typical use case |
|-------|-------------|------------------|
| `high` | Maximum capability—Claude uses as many tokens as needed for the best possible outcome. Equivalent to not setting the parameter. | Complex reasoning, difficult coding problems, agentic tasks |
| `medium` | Balanced approach with moderate token savings. | Agentic tasks that require a balance of speed, cost, and performance |
| `low` | Most efficient—significant token savings with some capability reduction. | Simpler tasks that need the best speed and lowest costs, such as subagents |

## Quick Start

### Using LiteLLM SDK

<Tabs>
<TabItem value="python" label="Python">

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{
        "role": "user",
        "content": "Analyze the trade-offs between microservices and monolithic architectures"
    }],
    reasoning_effort="medium"  # Automatically mapped to output_config for Opus 4.5
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="typescript" label="TypeScript">

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const response = await client.messages.create({
  model: "claude-opus-4-5-20251101",
  max_tokens: 4096,
  messages: [{
    role: "user",
    content: "Analyze the trade-offs between microservices and monolithic architectures"
  }],
  output_config: {
    effort: "medium"
  }
});

console.log(response.content[0].text);
```

</TabItem>
</Tabs>

### Using LiteLLM Proxy

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "anthropic/claude-opus-4-5-20251101",
    "messages": [{
      "role": "user",
      "content": "Analyze the trade-offs between microservices and monolithic architectures"
    }],
    "output_config": {
      "effort": "medium"
    }
  }'
```

### Direct Anthropic API Call

```bash
curl https://api.anthropic.com/v1/messages \
  --header "x-api-key: $ANTHROPIC_API_KEY" \
  --header "anthropic-version: 2023-06-01" \
  --header "anthropic-beta: effort-2025-11-24" \
  --header "content-type: application/json" \
  --data '{
    "model": "claude-opus-4-5-20251101",
    "max_tokens": 4096,
    "messages": [{
      "role": "user",
      "content": "Analyze the trade-offs between microservices and monolithic architectures"
    }],
    "output_config": {
      "effort": "medium"
    }
  }'
```

## Model Compatibility

The effort parameter is currently only supported by:
- **Claude Opus 4.5** (`claude-opus-4-5-20251101`)

## When Should I Adjust the Effort Parameter?

- Use **high effort** (the default) when you need Claude's best work—complex reasoning, nuanced analysis, difficult coding problems, or any task where quality is the top priority.

- Use **medium effort** as a balanced option when you want solid performance without the full token expenditure of high effort.

- Use **low effort** when you're optimizing for speed (because Claude answers with fewer tokens) or cost—for example, simple classification tasks, quick lookups, or high-volume use cases where marginal quality improvements don't justify additional latency or spend.

## Effort with Tool Use

When using tools, the effort parameter affects both the explanations around tool calls and the tool calls themselves. Lower effort levels tend to:
- Combine multiple operations into fewer tool calls
- Make fewer tool calls
- Proceed directly to action

Example with tools:

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{
        "role": "user",
        "content": "Check the weather in multiple cities"
    }],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
    output_config={
        "effort": "low"  # Will make fewer tool calls
    }
)
```

## Effort with Extended Thinking

The effort parameter works seamlessly with extended thinking. When both are enabled, effort controls the token budget across all response types:

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{
        "role": "user",
        "content": "Solve this complex problem"
    }],
    thinking={
        "type": "enabled",
        "budget_tokens": 5000
    },
    output_config={
        "effort": "medium"  # Affects both thinking and response tokens
    }
)
```

## Best Practices

1. **Start with the default (high)** for new tasks, then experiment with lower effort levels if you're looking to optimize costs.

2. **Use medium effort for production agentic workflows** where you need a balance of quality and efficiency.

3. **Reserve low effort for high-volume, simple tasks** like classification, routing, or data extraction where speed matters more than nuanced responses.

4. **Monitor token usage** to understand the actual savings from different effort levels for your specific use cases.

5. **Test with your specific prompts** as the impact of effort levels can vary based on task complexity.

## Provider Support

The effort parameter is supported across all Anthropic-compatible providers:

- **Standard Anthropic API**: ✅ Supported (Claude Opus 4.5)
- **Azure Anthropic / Microsoft Foundry**: ✅ Supported (Claude Opus 4.5)
- **Amazon Bedrock**: ✅ Supported (Claude Opus 4.5)
- **Google Cloud Vertex AI**: ✅ Supported (Claude Opus 4.5)

LiteLLM automatically handles:
- Beta header injection (`effort-2025-11-24`) for all providers
- Parameter mapping: `reasoning_effort` → `output_config={"effort": ...}` for Claude Opus 4.5

## Usage and Pricing

Token usage with different effort levels is tracked in the standard usage object. Lower effort levels result in fewer output tokens, which directly reduces costs:

```python
response = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{"role": "user", "content": "Analyze this"}],
    output_config={"effort": "low"}
)

print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

## Troubleshooting

### Beta header not being added

LiteLLM automatically adds the `effort-2025-11-24` beta header when:
- `reasoning_effort` parameter is provided (for Claude Opus 4.5 only)

If you're not seeing the header:

1. Ensure you're using `reasoning_effort` parameter
2. Verify the model is Claude Opus 4.5
3. Check that LiteLLM version supports this feature

### Invalid effort value error

Only three values are accepted: `"high"`, `"medium"`, `"low"`. Any other value will raise a validation error:

```python
# ❌ This will raise an error
output_config={"effort": "very_low"}

# ✅ Use one of the valid values
output_config={"effort": "low"}
```

### Model not supported

Currently, only Claude Opus 4.5 supports the effort parameter. Using it with other models may result in the parameter being ignored or an error.

## Related Features

- [Extended Thinking](/docs/providers/anthropic_extended_thinking) - Control Claude's reasoning process
- [Tool Use](/docs/providers/anthropic_tools) - Enable Claude to use tools and functions
- [Programmatic Tool Calling](/docs/providers/anthropic_programmatic_tool_calling) - Let Claude write code that calls tools
- [Prompt Caching](/docs/providers/anthropic_prompt_caching) - Cache prompts to reduce costs

## Additional Resources

- [Anthropic Effort Documentation](https://docs.anthropic.com/en/docs/build-with-claude/effort)
- [LiteLLM Anthropic Provider Guide](/docs/providers/anthropic)
- [Cost Optimization Best Practices](/docs/guides/cost_optimization)

