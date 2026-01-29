# Gemini Deep Research

Use Google's [Deep Research](https://ai.google.dev/gemini-api/docs/deep-research) agent through LiteLLM's Interactions API.

| Feature | Supported | Notes |
|---------|-----------|-------|
| Async Execution | ✅ | Research runs in background |
| Polling | ✅ | Check status with `get()` |
| Multi-source Analysis | ✅ | Analyzes 100+ sources |

## Overview

Deep Research is an autonomous research agent that analyzes multiple sources and generates comprehensive reports. Unlike regular model calls, Deep Research:

- Uses the `agent` parameter instead of `model`
- Runs asynchronously (requires polling for results)
- Takes several minutes to complete (analyzes 100+ sources)

## LiteLLM Python SDK Usage

### Start a Research Task

To start a Deep Research task, use `litellm.interactions.create()` with the `agent` parameter set to the Deep Research agent ID. The `background=True` parameter is required since Deep Research runs asynchronously.

```python showLineNumbers title="Start Deep Research"
import litellm
import os

os.environ["GEMINI_API_KEY"] = "your-api-key"

# Start Deep Research (runs in background)
interaction = litellm.interactions.create(
    agent="deep-research-pro-preview-12-2025",
    input="Research the competitive landscape of EV batteries.",
    background=True
)

print(f"Research started: {interaction.id}")
print(f"Status: {interaction.status}")
```

The response contains an `id` that you'll use to check the status and retrieve results.

### Poll for Results

Since Deep Research runs asynchronously, you need to poll the interaction status until it completes. Use `litellm.interactions.get()` with the interaction ID to check the current status.

```python showLineNumbers title="Poll for Completion"
import litellm
import time

# Poll until complete
while True:
    interaction = litellm.interactions.get(interaction_id=interaction.id)
    print(f"Status: {interaction.status}")

    if interaction.status == "completed":
        # Print the research report
        for output in interaction.outputs:
            if hasattr(output, "text"):
                print(output.text)
        break
    elif interaction.status in ["failed", "cancelled"]:
        print(f"Research {interaction.status}")
        break

    time.sleep(10)  # Check every 10 seconds
```

The `status` field will be `in_progress` while the research is running, and `completed` when the report is ready. Research typically takes 5-15 minutes depending on the complexity of the query.

### Complete Example

Here's a complete example that starts a research task and waits for the results:

```python showLineNumbers title="Full Deep Research Example"
import litellm
import os
import time

os.environ["GEMINI_API_KEY"] = "your-api-key"

def run_deep_research(query: str) -> str:
    """Run Deep Research and return the report."""

    # Start research
    interaction = litellm.interactions.create(
        agent="deep-research-pro-preview-12-2025",
        input=query,
        background=True
    )
    print(f"Research started: {interaction.id}")

    # Poll for completion
    while True:
        interaction = litellm.interactions.get(interaction_id=interaction.id)
        print(f"Status: {interaction.status}")

        if interaction.status == "completed":
            for output in interaction.outputs:
                if hasattr(output, "text"):
                    return output.text
            return ""
        elif interaction.status in ["failed", "cancelled"]:
            raise Exception(f"Research {interaction.status}")

        time.sleep(10)

# Run research
report = run_deep_research(
    "What are the latest developments in quantum computing?"
)
print(report)
```

### Async Usage

For async applications, use `acreate()` and `aget()` instead:

```python showLineNumbers title="Async Deep Research"
import litellm
import asyncio
import os

os.environ["GEMINI_API_KEY"] = "your-api-key"

async def run_deep_research_async(query: str) -> str:
    # Start research
    interaction = await litellm.interactions.acreate(
        agent="deep-research-pro-preview-12-2025",
        input=query,
        background=True
    )

    # Poll for completion
    while True:
        interaction = await litellm.interactions.aget(
            interaction_id=interaction.id
        )

        if interaction.status == "completed":
            for output in interaction.outputs:
                if hasattr(output, "text"):
                    return output.text
            return ""
        elif interaction.status in ["failed", "cancelled"]:
            raise Exception(f"Research {interaction.status}")

        await asyncio.sleep(10)

# Run
report = asyncio.run(run_deep_research_async(
    "What are the key AI breakthroughs in 2025?"
))
print(report)
```

## LiteLLM Proxy Usage

:::note
Proxy support for Deep Research (using the `agent` parameter) is coming soon.
:::

## Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent` | string | Yes | Agent ID: `deep-research-pro-preview-12-2025` |
| `input` | string | Yes | Research query or instructions |
| `background` | boolean | Yes | Must be `true` for Deep Research |
| `tools` | array | No | Additional tools (e.g., `file_search`) |
| `previous_interaction_id` | string | No | Continue from previous interaction |

## Response Status Values

| Status | Description |
|--------|-------------|
| `in_progress` | Research is running |
| `completed` | Research finished successfully |
| `failed` | Research encountered an error |
| `cancelled` | Research was cancelled |

## Limitations

- **Execution Time**: Research typically takes 5-15 minutes
- **No Custom Tools**: Cannot provide custom function calling tools
- **No Structured Output**: Does not support JSON schema output
- **Quota**: Uses significantly more resources than standard completions

## Related

- [Interactions API](/docs/interactions) - Base interactions endpoint documentation
- [Google Deep Research Docs](https://ai.google.dev/gemini-api/docs/deep-research) - Official Google documentation
