# Anthropic Programmatic Tool Calling

Programmatic tool calling allows Claude to write code that calls your tools programmatically within a code execution container, rather than requiring round trips through the model for each tool invocation. This reduces latency for multi-tool workflows and decreases token consumption by allowing Claude to filter or process data before it reaches the model's context window.

:::info
Programmatic tool calling is currently in public beta. LiteLLM automatically detects tools with the `allowed_callers` field and adds the appropriate beta header based on your provider:

- **Anthropic API & Microsoft Foundry**: `advanced-tool-use-2025-11-20`
- **Amazon Bedrock**: `advanced-tool-use-2025-11-20`
- **Google Cloud Vertex AI**: Not supported

This feature requires the code execution tool to be enabled.
:::

## Model Compatibility

Programmatic tool calling is available on the following models:

| Model | Tool Version |
|-------|--------------|
| Claude Opus 4.5 (`claude-opus-4-5-20251101`) | `code_execution_20250825` |
| Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) | `code_execution_20250825` |

## Quick Start

Here's a simple example where Claude programmatically queries a database multiple times and aggregates results:

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[
        {
            "role": "user",
            "content": "Query sales data for the West, East, and Central regions, then tell me which region had the highest revenue"
        }
    ],
    tools=[
        {
            "type": "code_execution_20250825",
            "name": "code_execution"
        },
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
            "allowed_callers": ["code_execution_20250825"]
        }
    ]
)

print(response)
```

## How It Works

When you configure a tool to be callable from code execution and Claude decides to use that tool:

1. Claude writes Python code that invokes the tool as a function, potentially including multiple tool calls and pre/post-processing logic
2. Claude runs this code in a sandboxed container via code execution
3. When a tool function is called, code execution pauses and the API returns a `tool_use` block with a `caller` field
4. You provide the tool result, and code execution continues (intermediate results are not loaded into Claude's context window)
5. Once all code execution completes, Claude receives the final output and continues working on the task

This approach is particularly useful for:

- **Large data processing**: Filter or aggregate tool results before they reach Claude's context
- **Multi-step workflows**: Save tokens and latency by calling tools serially or in a loop without sampling Claude in-between tool calls
- **Conditional logic**: Make decisions based on intermediate tool results

## The `allowed_callers` Field

The `allowed_callers` field specifies which contexts can invoke a tool:

```python
{
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Execute a SQL query against the database",
        "parameters": {...}
    },
    "allowed_callers": ["code_execution_20250825"]
}
```

**Possible values:**

- `["direct"]` - Only Claude can call this tool directly (default if omitted)
- `["code_execution_20250825"]` - Only callable from within code execution
- `["direct", "code_execution_20250825"]` - Callable both directly and from code execution

:::tip
We recommend choosing either `["direct"]` or `["code_execution_20250825"]` for each tool rather than enabling both, as this provides clearer guidance to Claude for how best to use the tool.
:::

## The `caller` Field in Responses

Every tool use block includes a `caller` field indicating how it was invoked:

**Direct invocation (traditional tool use):**

```python
{
    "type": "tool_use",
    "id": "toolu_abc123",
    "name": "query_database",
    "input": {"sql": "<sql>"},
    "caller": {"type": "direct"}
}
```

**Programmatic invocation:**

```python
{
    "type": "tool_use",
    "id": "toolu_xyz789",
    "name": "query_database",
    "input": {"sql": "<sql>"},
    "caller": {
        "type": "code_execution_20250825",
        "tool_id": "srvtoolu_abc123"
    }
}
```

The `tool_id` references the code execution tool that made the programmatic call.

## Container Lifecycle

Programmatic tool calling uses code execution containers:

- **Container creation**: A new container is created for each session unless you reuse an existing one
- **Expiration**: Containers expire after approximately 4.5 minutes of inactivity (subject to change)
- **Container ID**: Pass the `container` parameter to reuse an existing container
- **Reuse**: Pass the container ID to maintain state across requests

```python
# First request - creates a new container
response1 = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "Query the database"}],
    tools=[...]
)

# Get container ID from response (if available in response metadata)
container_id = response1.get("container", {}).get("id")

# Second request - reuse the same container
response2 = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[...],
    tools=[...],
    container=container_id  # Reuse container
)
```

:::warning
When a tool is called programmatically and the container is waiting for your tool result, you must respond before the container expires. Monitor the `expires_at` field. If the container expires, Claude may treat the tool call as timed out and retry it.
:::

## Example Workflow

### Step 1: Initial Request

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{
        "role": "user",
        "content": "Query customer purchase history from the last quarter and identify our top 5 customers by revenue"
    }],
    tools=[
        {
            "type": "code_execution_20250825",
            "name": "code_execution"
        },
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Execute a SQL query against the sales database. Returns a list of rows as JSON objects.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query to execute"}
                    },
                    "required": ["sql"]
                }
            },
            "allowed_callers": ["code_execution_20250825"]
        }
    ]
)
```

### Step 2: API Response with Tool Call

Claude writes code that calls your tool. The response includes:

```python
{
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "I'll query the purchase history and analyze the results."
        },
        {
            "type": "server_tool_use",
            "id": "srvtoolu_abc123",
            "name": "code_execution",
            "input": {
                "code": "results = await query_database('<sql>')\ntop_customers = sorted(results, key=lambda x: x['revenue'], reverse=True)[:5]"
            }
        },
        {
            "type": "tool_use",
            "id": "toolu_def456",
            "name": "query_database",
            "input": {"sql": "<sql>"},
            "caller": {
                "type": "code_execution_20250825",
                "tool_id": "srvtoolu_abc123"
            }
        }
    ],
    "stop_reason": "tool_use"
}
```

### Step 3: Provide Tool Result

```python
# Add assistant's response and tool result to conversation
messages = [
    {"role": "user", "content": "Query customer purchase history..."},
    {
        "role": "assistant",
        "content": response.choices[0].message.content,
        "tool_calls": response.choices[0].message.tool_calls
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_def456",
                "content": '[{"customer_id": "C1", "revenue": 45000}, ...]'
            }
        ]
    }
]

# Continue the conversation
response2 = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=messages,
    tools=[...]
)
```

### Step 4: Final Response

Once code execution completes, Claude provides the final response:

```python
{
    "content": [
        {
            "type": "code_execution_tool_result",
            "tool_use_id": "srvtoolu_abc123",
            "content": {
                "type": "code_execution_result",
                "stdout": "Top 5 customers by revenue:\n1. Customer C1: $45,000\n...",
                "stderr": "",
                "return_code": 0
            }
        },
        {
            "type": "text",
            "text": "I've analyzed the purchase history from last quarter. Your top 5 customers generated $167,500 in total revenue..."
        }
    ],
    "stop_reason": "end_turn"
}
```

## Advanced Patterns

### Batch Processing with Loops

Claude can write code that processes multiple items efficiently:

```python
# Claude writes code like this:
regions = ["West", "East", "Central", "North", "South"]
results = {}
for region in regions:
    data = await query_database(f"SELECT SUM(revenue) FROM sales WHERE region='{region}'")
    results[region] = data[0]["total"]

top_region = max(results.items(), key=lambda x: x[1])
print(f"Top region: {top_region[0]} with ${top_region[1]:,}")
```

This pattern:
- Reduces model round-trips from N (one per region) to 1
- Processes large result sets programmatically before returning to Claude
- Saves tokens by only returning aggregated conclusions

### Early Termination

Claude can stop processing as soon as success criteria are met:

```python
endpoints = ["us-east", "eu-west", "apac"]
for endpoint in endpoints:
    status = await check_health(endpoint)
    if status == "healthy":
        print(f"Found healthy endpoint: {endpoint}")
        break  # Stop early
```

### Data Filtering

```python
logs = await fetch_logs(server_id)
errors = [log for log in logs if "ERROR" in log]
print(f"Found {len(errors)} errors")
for error in errors[-10:]:  # Only return last 10 errors
    print(error)
```

## Best Practices

### Tool Design

- **Provide detailed output descriptions**: Since Claude deserializes tool results in code, clearly document the format (JSON structure, field types, etc.)
- **Return structured data**: JSON or other easily parseable formats work best for programmatic processing
- **Keep responses concise**: Return only necessary data to minimize processing overhead

### When to Use Programmatic Calling

**Good use cases:**

- Processing large datasets where you only need aggregates or summaries
- Multi-step workflows with 3+ dependent tool calls
- Operations requiring filtering, sorting, or transformation of tool results
- Tasks where intermediate data shouldn't influence Claude's reasoning
- Parallel operations across many items (e.g., checking 50 endpoints)

**Less ideal use cases:**

- Single tool calls with simple responses
- Tools that need immediate user feedback
- Very fast operations where code execution overhead would outweigh the benefit

## Token Efficiency

Programmatic tool calling can significantly reduce token consumption:

- **Tool results from programmatic calls are not added to Claude's context** - only the final code output is
- **Intermediate processing happens in code** - filtering, aggregation, etc. don't consume model tokens
- **Multiple tool calls in one code execution** - reduces overhead compared to separate model turns

For example, calling 10 tools directly uses ~10x the tokens of calling them programmatically and returning a summary.

## Provider Support

LiteLLM supports programmatic tool calling across the following Anthropic-compatible providers:

- **Standard Anthropic API** (`anthropic/claude-sonnet-4-5-20250929`) ✅
- **Azure Anthropic / Microsoft Foundry** (`azure/claude-sonnet-4-5-20250929`) ✅
- **Amazon Bedrock** (`bedrock/invoke/anthropic.claude-sonnet-4-5-20250929-v1:0`) ✅
- **Google Cloud Vertex AI** (`vertex_ai/claude-sonnet-4-5-20250929`) ❌ Not supported

The beta header (`advanced-tool-use-2025-11-20`) is automatically added when LiteLLM detects tools with the `allowed_callers` field.

## Limitations

### Feature Incompatibilities

- **Structured outputs**: Tools with `strict: true` are not supported with programmatic calling
- **Tool choice**: You cannot force programmatic calling of a specific tool via `tool_choice`
- **Parallel tool use**: `disable_parallel_tool_use: true` is not supported with programmatic calling

### Tool Restrictions

The following tools cannot currently be called programmatically:

- Web search
- Web fetch
- Tools provided by an MCP connector

## Troubleshooting

### Common Issues

**"Tool not allowed" error**

- Verify your tool definition includes `"allowed_callers": ["code_execution_20250825"]`
- Check that you're using a compatible model (Claude Sonnet 4.5 or Opus 4.5)

**Container expiration**

- Ensure you respond to tool calls within the container's lifetime (~4.5 minutes)
- Consider implementing faster tool execution

**Beta header not added**

- LiteLLM automatically adds the beta header when it detects `allowed_callers`
- If you're manually setting headers, ensure you include `advanced-tool-use-2025-11-20`

## Related Features

- [Anthropic Tool Search](./anthropic_tool_search.md) - Dynamically discover and load tools on-demand
- [Anthropic Provider](./anthropic.md) - General Anthropic provider documentation

