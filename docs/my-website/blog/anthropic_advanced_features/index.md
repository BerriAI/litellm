---
slug: anthropic_advanced_features
title: "Advanced Anthropic Features in LiteLLM: Tool Search, Programmatic Tool Calling, Input Examples, and Effort Control"
date: 2025-01-25T10:00:00
authors:
  - name: Sameer Kankute
    title: SWE @ LiteLLM (LLM Translation)
    url: https://www.linkedin.com/in/sameer-kankute/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQHB_loQYd5gjg/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1719137160975?e=1765411200&v=beta&t=c8396f--_lH6Fb_pVvx_jGholPfcl0bvwmNynbNdnII
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
tags: [anthropic, claude, tool search, programmatic tool calling, effort, advanced features]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

:::info

This guide covers Anthropic's latest advanced features now available in LiteLLM: Tool Search, Programmatic Tool Calling, Tool Input Examples, and the Effort Parameter.

:::

We're excited to announce support for Anthropic's latest advanced features in LiteLLM! These powerful capabilities enable you to build more efficient, scalable, and cost-effective AI applications with Claude.

## Table of Contents

1. [Tool Search](#tool-search)
2. [Programmatic Tool Calling](#programmatic-tool-calling)
3. [Tool Input Examples](#tool-input-examples)
4. [Effort Parameter: Control Token Usage](#effort-parameter)
5. [Cost Tracking: Monitor Tool Search Usage](#cost-tracking)
6. [Combining Features](#combining-features)

---

## Tool Search {#tool-search}

### Usage Example

```python
import litellm
import os

# Configure your API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Define your tools with defer_loading
tools = [
    # Tool search tool (regex variant)
    {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex"
    },
    # Deferred tools - loaded on-demand
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location. Returns temperature and conditions.",
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
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        },
        "defer_loading": True  # Load on-demand
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search through files in the workspace using keywords",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["query"]
            }
        },
        "defer_loading": True
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute SQL queries against the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            }
        },
        "defer_loading": True
    }
]

# Make a request - Claude will search for and use relevant tools
response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{
        "role": "user",
        "content": "What's the weather like in San Francisco?"
    }],
    tools=tools
)

print("Claude's response:", response.choices[0].message.content)
print("Tool calls:", response.choices[0].message.tool_calls)

# Check tool search usage
if hasattr(response.usage, 'server_tool_use'):
    print(f"Tool searches performed: {response.usage.server_tool_use.tool_search_requests}")
```

### BM25 Variant (Natural Language Search)

For natural language queries instead of regex patterns:

```python
tools = [
    {
        "type": "tool_search_tool_bm25_20251119",  # Natural language variant
        "name": "tool_search_tool_bm25"
    },
    # ... your deferred tools
]
```

---

## Programmatic Tool Calling {#programmatic-tool-calling}

### Usage Example

```python
import litellm
import json

# Define tools that can be called programmatically
tools = [
    # Code execution tool (required for programmatic calling)
    {
        "type": "code_execution_20250825",
        "name": "code_execution"
    },
    # Tool that can be called from code
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute a SQL query against the sales database. Returns a list of rows as JSON objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL query to execute"
                    }
                },
                "required": ["sql"]
            }
        },
        "allowed_callers": ["code_execution_20250825"]  # Enable programmatic calling
    }
]

# First request
response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{
        "role": "user",
        "content": "Query sales data for West, East, and Central regions, then tell me which had the highest revenue"
    }],
    tools=tools
)

print("Claude's response:", response.choices[0].message)

# Handle tool calls
messages = [
    {"role": "user", "content": "Query sales data for West, East, and Central regions, then tell me which had the highest revenue"},
    {"role": "assistant", "content": response.choices[0].message.content, "tool_calls": response.choices[0].message.tool_calls}
]

# Process each tool call
for tool_call in response.choices[0].message.tool_calls:
    # Check if it's a programmatic call
    if hasattr(tool_call, 'caller') and tool_call.caller:
        print(f"Programmatic call to {tool_call.function.name}")
        print(f"Called from: {tool_call.caller}")
    
    # Simulate tool execution
    if tool_call.function.name == "query_database":
        args = json.loads(tool_call.function.arguments)
        # Simulate database query
        result = json.dumps([
            {"region": "West", "revenue": 150000},
            {"region": "East", "revenue": 180000},
            {"region": "Central", "revenue": 120000}
        ])
        
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result
            }]
        })

# Get final response
final_response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=messages,
    tools=tools
)

print("\nFinal answer:", final_response.choices[0].message.content)
```

---

## Tool Input Examples {#tool-input-examples}

### Usage Example

```python
import litellm

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new calendar event with attendees and reminders",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_time": {
                        "type": "string",
                        "description": "ISO 8601 format: YYYY-MM-DDTHH:MM:SS"
                    },
                    "duration_minutes": {"type": "integer"},
                    "attendees": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "optional": {"type": "boolean"}
                            }
                        }
                    },
                    "reminders": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "minutes_before": {"type": "integer"},
                                "method": {"type": "string", "enum": ["email", "popup"]}
                            }
                        }
                    }
                },
                "required": ["title", "start_time", "duration_minutes"]
            }
        },
        # Provide concrete examples
        "input_examples": [
            {
                "title": "Team Standup",
                "start_time": "2025-01-15T09:00:00",
                "duration_minutes": 30,
                "attendees": [
                    {"email": "alice@company.com", "optional": False},
                    {"email": "bob@company.com", "optional": False}
                ],
                "reminders": [
                    {"minutes_before": 15, "method": "popup"}
                ]
            },
            {
                "title": "Lunch Break",
                "start_time": "2025-01-15T12:00:00",
                "duration_minutes": 60
                # Demonstrates optional fields can be omitted
            }
        ]
    }
]

response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{
        "role": "user",
        "content": "Schedule a team meeting for tomorrow at 2pm for 45 minutes with john@company.com and sarah@company.com"
    }],
    tools=tools
)

print("Tool call:", response.choices[0].message.tool_calls[0].function.arguments)
```

---

## Effort Parameter: Control Token Usage {#effort-parameter}

### Usage Example

```python
import litellm

message = "Analyze the trade-offs between microservices and monolithic architectures"

# High effort (default) - Maximum capability
response_high = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{"role": "user", "content": message}],
    output_config={"effort": "high"}
)

print("High effort response:")
print(response_high.choices[0].message.content)
print(f"Tokens used: {response_high.usage.completion_tokens}\n")

# Medium effort - Balanced approach
response_medium = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{"role": "user", "content": message}],
    output_config={"effort": "medium"}
)

print("Medium effort response:")
print(response_medium.choices[0].message.content)
print(f"Tokens used: {response_medium.usage.completion_tokens}\n")

# Low effort - Maximum efficiency
response_low = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{"role": "user", "content": message}],
    output_config={"effort": "low"}
)

print("Low effort response:")
print(response_low.choices[0].message.content)
print(f"Tokens used: {response_low.usage.completion_tokens}\n")

# Compare token usage
print("Token Comparison:")
print(f"High:   {response_high.usage.completion_tokens} tokens")
print(f"Medium: {response_medium.usage.completion_tokens} tokens")
print(f"Low:    {response_low.usage.completion_tokens} tokens")
```

### Effort with Tool Use

Lower effort affects both explanations and tool calls:

```python
tools = [
    {
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
    }
]

# Low effort = fewer tool calls, more direct
response = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{
        "role": "user",
        "content": "Check weather in San Francisco, New York, and London"
    }],
    tools=tools,
    output_config={"effort": "low"}  # May combine into fewer calls
)
```

---

## Cost Tracking: Monitor Tool Search Usage {#cost-tracking}

### Understanding Tool Search Costs

Tool search operations are tracked separately in the usage object, allowing you to monitor and optimize costs.

### Tracking Example

```python
import litellm

tools = [
    {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex"
    },
    # ... 100 deferred tools
]

response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{
        "role": "user",
        "content": "Find and use the weather tool for San Francisco"
    }],
    tools=tools
)

# Standard token usage
print("Token Usage:")
print(f"  Input tokens:  {response.usage.prompt_tokens}")
print(f"  Output tokens: {response.usage.completion_tokens}")
print(f"  Total tokens:  {response.usage.total_tokens}")

# Tool search specific usage
if hasattr(response.usage, 'server_tool_use') and response.usage.server_tool_use:
    print(f"\nTool Search Usage:")
    print(f"  Search requests: {response.usage.server_tool_use.tool_search_requests}")
    
    # Calculate cost (example pricing)
    input_cost = response.usage.prompt_tokens * 0.000003  # $3 per 1M tokens
    output_cost = response.usage.completion_tokens * 0.000015  # $15 per 1M tokens
    search_cost = response.usage.server_tool_use.tool_search_requests * 0.0001  # Example
    
    total_cost = input_cost + output_cost + search_cost
    
    print(f"\nCost Breakdown:")
    print(f"  Input tokens:   ${input_cost:.6f}")
    print(f"  Output tokens:  ${output_cost:.6f}")
    print(f"  Tool searches:  ${search_cost:.6f}")
    print(f"  Total:          ${total_cost:.6f}")
```

### Cost Optimization Tips

1. **Keep frequently used tools non-deferred** (3-5 tools)
2. **Use tool search for large catalogs** (10+ tools)
3. **Monitor search requests** to identify optimization opportunities
4. **Combine with effort parameter** for maximum efficiency

```python
# Optimized for cost
response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "Simple query"}],
    tools=tools_with_search,
    output_config={"effort": "low"}  # Reduce output tokens
)
```

---

## Combining Features {#combining-features}

### The Power of Integration

These features work together seamlessly. Here's a real-world example combining all of them:

```python
import litellm
import json

# Large tool catalog with search, programmatic calling, and examples
tools = [
    # Enable tool search
    {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex"
    },
    # Enable programmatic calling
    {
        "type": "code_execution_20250825",
        "name": "code_execution"
    },
    # Database tool with all features
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute SQL queries against the analytics database. Returns JSON array of results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT statement"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum rows to return"
                    }
                },
                "required": ["sql"]
            }
        },
        "defer_loading": True,  # Tool search
        "allowed_callers": ["code_execution_20250825"],  # Programmatic calling
        "input_examples": [  # Input examples
            {
                "sql": "SELECT region, SUM(revenue) as total FROM sales GROUP BY region",
                "limit": 100
            }
        ]
    },
    # ... 50 more tools with defer_loading
]

# Make request with effort control
response = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{
        "role": "user",
        "content": "Analyze sales by region for the last quarter and identify top performers"
    }],
    tools=tools,
    output_config={"effort": "medium"}  # Balanced efficiency
)

# Track comprehensive usage
print("Complete Usage Metrics:")
print(f"  Input tokens:     {response.usage.prompt_tokens}")
print(f"  Output tokens:    {response.usage.completion_tokens}")
print(f"  Total tokens:     {response.usage.total_tokens}")

if hasattr(response.usage, 'server_tool_use') and response.usage.server_tool_use:
    print(f"  Tool searches:    {response.usage.server_tool_use.tool_search_requests}")

print(f"\nResponse: {response.choices[0].message.content}")
```

### Real-World Benefits

This combination enables:

1. **Massive scale** - Handle 1000+ tools efficiently
2. **Low latency** - Programmatic calling reduces round trips
3. **High accuracy** - Input examples ensure correct tool usage
4. **Cost control** - Effort parameter optimizes token spend
5. **Full visibility** - Track all usage metrics

---

## Getting Started

### Installation

```bash
pip install litellm --upgrade
```

### Configuration

```python
import os
import litellm

# Set your API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# LiteLLM automatically handles beta headers for all features
```

### Supported Models

| Feature | Supported Models |
|---------|-----------------|
| Tool Search | Claude Opus 4.5, Sonnet 4.5 |
| Programmatic Tool Calling | Claude Opus 4.5, Sonnet 4.5 |
| Input Examples | Claude Opus 4.5, Sonnet 4.5 |
| Effort Parameter | Claude Opus 4.5 only |

### Supported Endpoints

**Note**: All features are supported on the `/chat/completions` endpoint only.

| Feature | Supported Models |
|---------|-----------------|
| Tool Search | Claude Opus 4.5, Sonnet 4.5 |
| Programmatic Tool Calling | Claude Opus 4.5, Sonnet 4.5 |
| Input Examples | Claude Opus 4.5, Sonnet 4.5 |
| Effort Parameter | Claude Opus 4.5 only |

### Provider Support

All features work across:
- ✅ Standard Anthropic API
- ✅ Azure Anthropic
- ✅ Vertex AI Anthropic
- ✅ LiteLLM Proxy

---

## Conclusion

These advanced Anthropic features in LiteLLM enable you to build more sophisticated, efficient, and cost-effective AI applications:

- **Tool Search** scales to thousands of tools
- **Programmatic Tool Calling** reduces latency and tokens
- **Input Examples** improve accuracy
- **Effort Parameter** controls costs

All features work seamlessly together and are supported across all Anthropic providers through LiteLLM's unified interface.

### Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Anthropic Tool Search Docs](https://docs.litellm.ai/docs/providers/anthropic_tool_search)
- [Anthropic Programmatic Tool Calling Docs](https://docs.litellm.ai/docs/providers/anthropic_programmatic_tool_calling)
- [Anthropic Input Examples Docs](https://docs.litellm.ai/docs/providers/anthropic_tool_input_examples)
- [Anthropic Effort Parameter Docs](https://docs.litellm.ai/docs/providers/anthropic_effort)

### Get Started Today

```bash
pip install litellm --upgrade
```

Happy building! 🚀

