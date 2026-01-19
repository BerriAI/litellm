# WebSearch Interception Architecture

Server-side WebSearch tool execution for models that don't natively support it (e.g., Bedrock/Claude).

## How It Works

User makes **ONE** `litellm.messages.acreate()` call → Gets final answer with search results.
The agentic loop happens transparently on the server.

## LiteLLM Standard Web Search Tool

LiteLLM defines a standard web search tool format (`litellm_web_search`) that all native provider tools are converted to. This enables consistent interception across providers.

**Standard Tool Definition** (defined in `tools.py`):
```python
{
    "name": "litellm_web_search",
    "description": "Search the web for information...",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
}
```

**Tool Name Constant**: `LITELLM_WEB_SEARCH_TOOL_NAME = "litellm_web_search"` (defined in `litellm/constants.py`)

### Supported Tool Formats

The interception system automatically detects and handles:

| Tool Format | Example | Provider | Detection Method | Future-Proof |
|-------------|---------|----------|------------------|-------------|
| **LiteLLM Standard** | `name="litellm_web_search"` | Any | Direct name match | N/A |
| **Anthropic Native** | `type="web_search_20250305"` | Bedrock, Claude API | Type prefix: `startswith("web_search_")` | ✅ Yes (web_search_2026, etc.) |
| **Claude Code CLI** | `name="web_search"`, `type="web_search_20250305"` | Claude Code | Name + type check | ✅ Yes (version-agnostic) |
| **Legacy** | `name="WebSearch"` | Custom | Name match | N/A (backwards compat) |

**Future Compatibility**: The `startswith("web_search_")` check in `tools.py` automatically supports future Anthropic web search versions.

### Claude Code CLI Integration

Claude Code (Anthropic's official CLI) sends web search requests using Anthropic's native tool format:

```python
{
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 8
}
```

**What Happens:**
1. Claude Code sends native `web_search_20250305` tool to LiteLLM proxy
2. LiteLLM intercepts and converts to `litellm_web_search` standard format
3. Bedrock receives converted tool (NOT native format)
4. Model returns `tool_use` block for `litellm_web_search` (not `server_tool_use`)
5. LiteLLM's agentic loop intercepts the `tool_use`
6. Executes `litellm.asearch()` using configured provider (Perplexity, Tavily, etc.)
7. Returns final answer to Claude Code user

**Without Interception**: Bedrock would receive native tool → try to execute natively → return `web_search_tool_result_error` with `invalid_tool_input`

**With Interception**: LiteLLM converts → Bedrock returns tool_use → LiteLLM executes search → Returns final answer ✅

### Native Tool Conversion

Native tools are converted to LiteLLM standard format **before** sending to the provider:

1. **Conversion Point** (`litellm/llms/anthropic/experimental_pass_through/messages/handler.py`):
   - In `anthropic_messages()` function (lines 60-127)
   - Runs BEFORE the API request is made
   - Detects native web search tools using `is_web_search_tool()`
   - Converts to `litellm_web_search` format using `get_litellm_web_search_tool()`
   - Prevents provider from executing search natively (avoids `web_search_tool_result_error`)

2. **Response Detection** (`transformation.py`):
   - Detects `tool_use` blocks with any web search tool name
   - Handles: `litellm_web_search`, `WebSearch`, `web_search`
   - Extracts search queries for execution

**Example Conversion**:
```python
# Input (Claude Code's native tool)
{
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 8
}

# Output (LiteLLM standard)
{
    "name": "litellm_web_search",
    "description": "Search the web for information...",
    "input_schema": {...}
}
```

---

## Request Flow

### Without Interception (Client-Side)
User manually handles tool execution:
1. User calls `litellm.messages.acreate()` → Gets `tool_use` response
2. User executes `litellm.asearch()`
3. User calls `litellm.messages.acreate()` again with results
4. User gets final answer

**Result**: 2 API calls, manual tool execution

### With Interception (Server-Side)
Server handles tool execution automatically:

```mermaid
sequenceDiagram
    participant User
    participant Messages as litellm.messages.acreate()
    participant Handler as llm_http_handler.py
    participant Logger as WebSearchInterceptionLogger
    participant Router as proxy_server.llm_router
    participant Search as litellm.asearch()
    participant Provider as Bedrock API

    User->>Messages: acreate(tools=[WebSearch])
    Messages->>Handler: async_anthropic_messages_handler()
    Handler->>Provider: Request
    Provider-->>Handler: Response (tool_use)
    Handler->>Logger: async_should_run_agentic_loop()
    Logger->>Logger: Detect WebSearch tool_use
    Logger-->>Handler: (True, tools)
    Handler->>Logger: async_run_agentic_loop(tools)
    Logger->>Router: Get search_provider from search_tools
    Router-->>Logger: search_provider
    Logger->>Search: asearch(query, provider)
    Search-->>Logger: Search results
    Logger->>Logger: Build tool_result message
    Logger->>Messages: acreate() with results
    Messages->>Provider: Request with search results
    Provider-->>Messages: Final answer
    Messages-->>Logger: Final response
    Logger-->>Handler: Final response
    Handler-->>User: Final answer (with search results)
```

**Result**: 1 API call from user, server handles agentic loop

---

## Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **WebSearchInterceptionLogger** | `handler.py` | CustomLogger that implements agentic loop hooks |
| **Tool Standardization** | `tools.py` | Standard tool definition, detection, and utilities |
| **Tool Name Constant** | `constants.py` | `LITELLM_WEB_SEARCH_TOOL_NAME = "litellm_web_search"` |
| **Tool Conversion** | `anthropic/.../ handler.py` | Converts native tools to LiteLLM standard before API call |
| **Transformation Logic** | `transformation.py` | Detect tool_use, build tool_result messages, format search responses |
| **Agentic Loop Hooks** | `integrations/custom_logger.py` | Base hooks: `async_should_run_agentic_loop()`, `async_run_agentic_loop()` |
| **Hook Orchestration** | `llms/custom_httpx/llm_http_handler.py` | `_call_agentic_completion_hooks()` - calls hooks after response |
| **Router Search Tools** | `proxy/proxy_server.py` | `llm_router.search_tools` - configured search providers |
| **Search Endpoints** | `proxy/search_endpoints/endpoints.py` | Router logic for selecting search provider |

---

## Configuration

```python
from litellm.integrations.websearch_interception import (
    WebSearchInterceptionLogger,
    get_litellm_web_search_tool,
)
from litellm.types.utils import LlmProviders

# Enable for Bedrock with specific search tool
litellm.callbacks = [
    WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.BEDROCK],
        search_tool_name="my-perplexity-tool"  # Optional: uses router's first tool if None
    )
]

# Make request with LiteLLM standard tool (recommended)
response = await litellm.messages.acreate(
    model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[{"role": "user", "content": "What is LiteLLM?"}],
    tools=[get_litellm_web_search_tool()],  # LiteLLM standard
    max_tokens=1024,
    stream=True  # Auto-converted to non-streaming
)

# OR send native tools - they're auto-converted to LiteLLM standard
response = await litellm.messages.acreate(
    model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[{"role": "user", "content": "What is LiteLLM?"}],
    tools=[{
        "type": "web_search_20250305",  # Native Anthropic format
        "name": "web_search",
        "max_uses": 8
    }],
    max_tokens=1024,
)
```

---

## Streaming Support

WebSearch interception works transparently with both streaming and non-streaming requests.

**How streaming is handled:**
1. User makes request with `stream=True` and WebSearch tool
2. Before API call, `anthropic_messages()` detects WebSearch + interception enabled
3. Converts `stream=True` → `stream=False` internally
4. Agentic loop executes with non-streaming responses
5. Final response returned to user (non-streaming)

**Why this approach:**
- Server-side agentic loops require consuming full responses to detect tool_use
- User opts into this behavior by enabling WebSearch interception
- Provides seamless experience without client changes

**Testing:**
- **Non-streaming**: `test_websearch_interception_e2e.py`
- **Streaming**: `test_websearch_interception_streaming_e2e.py`

---

## Search Provider Selection

1. If `search_tool_name` specified → Look up in `llm_router.search_tools`
2. If not found or None → Use first available search tool
3. If no router or no tools → Fallback to `perplexity`

Example router config:
```yaml
search_tools:
  - search_tool_name: "my-perplexity-tool"
    litellm_params:
      search_provider: "perplexity"
  - search_tool_name: "my-tavily-tool"
    litellm_params:
      search_provider: "tavily"
```

---

## Message Flow

### Initial Request
```python
messages = [{"role": "user", "content": "What is LiteLLM?"}]
tools = [{"name": "WebSearch", ...}]
```

### First API Call (Internal)
**Response**: `tool_use` with `name="WebSearch"`, `input={"query": "what is litellm"}`

### Server Processing
1. Logger detects WebSearch tool_use
2. Looks up search provider from router
3. Executes `litellm.asearch(query="what is litellm", search_provider="perplexity")`
4. Gets results: `"Title: LiteLLM Docs\nURL: docs.litellm.ai\n..."`

### Follow-Up Request (Internal)
```python
messages = [
    {"role": "user", "content": "What is LiteLLM?"},
    {"role": "assistant", "content": [{"type": "tool_use", ...}]},
    {"role": "user", "content": [{"type": "tool_result", "content": "search results..."}]}
]
```

### User Receives
```python
response.content[0].text
# "Based on the search results, LiteLLM is a unified interface..."
```

---

## Testing

**E2E Tests**:
- `test_websearch_interception_e2e.py` - Non-streaming real API calls to Bedrock
- `test_websearch_interception_streaming_e2e.py` - Streaming real API calls to Bedrock

**Unit Tests**: `test_websearch_interception.py`
Mocked tests for tool detection, provider filtering, edge cases.
