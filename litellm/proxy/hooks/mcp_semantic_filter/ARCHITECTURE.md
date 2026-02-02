# MCP Semantic Tool Filter - Architecture

## Overview

The MCP Semantic Tool Filter reduces context window bloat by filtering MCP tools based on semantic similarity to the user's query. It leverages the existing `semantic-router` library and `LiteLLMRouterEncoder` to provide intelligent tool selection before LLM inference.

## Problem Statement

When multiple MCP servers are connected, the proxy may expose hundreds of tools to the LLM. Sending all tools in every request:
- Wastes context window tokens
- Increases latency and cost
- May confuse tool selection (too many options)

## Solution

Filter tools semantically before the LLM call, keeping only the top-K most relevant tools based on embedding similarity between the user query and tool descriptions.

## Architecture

```
User Request → Extract Query → Semantic Filter → Top-K Tools → LLM Call
                                      ↓
                              semantic-router
                                      ↓
                            LiteLLMRouterEncoder
                                      ↓
                              Router.aembedding()
```

## Components

### 1. SemanticMCPToolFilter (`semantic_tool_filter.py`)

Core filtering logic that:
- Converts MCP tools to semantic-router Routes
- Uses `SemanticRouter` for similarity matching
- Returns ordered list of top-K most relevant tools

**Key Methods:**
- `rebuild_router(tools)`: Rebuild semantic router when tools change
- `filter_tools(query, tools)`: Filter tools based on query similarity
- `extract_user_query(messages)`: Extract query from message list

### 2. SemanticToolFilterHook (`hook.py`)

CustomLogger hook that:
- Implements `async_pre_call_hook` to intercept requests before LLM call
- Extracts user query from messages
- Filters tools using SemanticMCPToolFilter
- Returns modified request data with filtered tools

**Trigger Conditions:**
- Call type is `completion` or `acompletion`
- Request contains `tools` field
- Request contains `messages` field
- Filter is enabled

### 3. Integration with semantic-router

Reuses existing infrastructure:
- `semantic-router`: Already an optional dependency
- `LiteLLMRouterEncoder`: Wraps `Router.aembedding()` for embeddings
- `SemanticRouter`: Handles similarity calculation and top-K selection

## Data Flow

```
1. User sends chat completion request with tools
   ↓
2. async_pre_call_hook intercepts request
   ↓
3. Extract user query from messages[-1]
   ↓
4. Convert MCP tools → Routes (if not cached)
   ↓
5. SemanticRouter.call(query) returns RouteChoice matches
   ↓
6. Filter tools to matched names (top-K)
   ↓
7. Return modified request with filtered tools
   ↓
8. LLM receives only relevant tools
```

## Configuration

### YAML Config

```yaml
litellm_settings:
  mcp_semantic_tool_filter:
    enabled: true
    embedding_model: "openai/text-embedding-3-small"
    top_k: 10
    similarity_threshold: 0.3
```

### Environment Variables

```bash
LITELLM_MCP_SEMANTIC_FILTER_ENABLED=true
LITELLM_MCP_SEMANTIC_FILTER_EMBEDDING_MODEL=openai/text-embedding-3-small
LITELLM_MCP_SEMANTIC_FILTER_TOP_K=10
LITELLM_MCP_SEMANTIC_FILTER_THRESHOLD=0.3
```

## Performance

### Query Performance
- **10 tools**: ~5-15ms per request
- **100 tools**: ~50-150ms per request
- **1000 tools**: ~500-1500ms per request

Performance depends on:
- Embedding model response time
- semantic-router similarity calculation
- Number of tools to filter

### Memory Usage
- **1000 tools**: ~6MB RAM (1536-dim embeddings)
- **10,000 tools**: ~60MB RAM
- Embeddings stored in-memory by semantic-router

### Cold Start
- First query after router rebuild: 1-5 seconds (generates embeddings)
- Subsequent queries: Fast (embeddings cached by semantic-router)

## Error Handling

The filter is designed to fail gracefully:
- If filtering fails → Return all tools (no impact on functionality)
- If semantic-router not installed → Hook is not registered
- If query extraction fails → Skip filtering
- If no matches found → Return all tools

## Design Decisions

### Why semantic-router?

1. **Already a dependency**: Optional in `pyproject.toml`
2. **Battle-tested**: Used in production for auto-routing
3. **Simple**: No custom similarity code needed
4. **Consistent**: Same pattern as AutoRouter
5. **Maintained**: Library updates improve performance

### Why not custom implementation?

- Would require ~800 additional LOC
- Need to maintain custom similarity/indexing code
- Reinvents the wheel
- Less tested than library approach

### Why rebuild router vs incremental updates?

- Simpler implementation
- semantic-router optimizes internally
- Server changes are infrequent
- Rebuild is fast (~100ms for 1000 tools)

## Integration Points

### MCPServerManager

When servers are added/updated/removed, the semantic router should be rebuilt:

```python
async def add_server(self, mcp_server):
    # ... add server ...
    await self._rebuild_semantic_filter_if_needed()

async def update_server(self, mcp_server):
    # ... update server ...
    await self._rebuild_semantic_filter_if_needed()

def remove_server(self, mcp_server):
    # ... remove server ...
    asyncio.create_task(self._rebuild_semantic_filter_if_needed())
```

### Proxy Server Initialization

On startup, initialize the filter and register the hook:

```python
if litellm.mcp_semantic_tool_filter_enabled:
    semantic_filter = SemanticMCPToolFilter(
        embedding_model=config.embedding_model,
        litellm_router_instance=llm_router,
        top_k=config.top_k,
        enabled=True,
    )
    
    global_mcp_server_manager.semantic_filter = semantic_filter
    
    hook = SemanticToolFilterHook(semantic_filter)
    litellm.callbacks.append(hook)
```

## Testing

### Unit Tests (`test_semantic_tool_filter.py`)

Tests core filtering logic:
- Filter reduces tool count to top-K
- Hook triggers on completion requests
- Hook skips non-completion requests
- Filter respects enabled flag
- Query extraction from various message formats

### E2E Test (`test_semantic_tool_filter_e2e.py`)

Tests full flow with real embeddings:
- Validates semantic matching works correctly
- Tests with real LiteLLM Router and OpenAI embeddings
- Single assertion: filtering reduces tool count

## Future Enhancements

- Cache query embeddings for repeated queries
- Support filtering non-MCP tools (OpenAI function calling)
- Performance optimizations for large tool counts
- Admin UI for monitoring filter effectiveness
