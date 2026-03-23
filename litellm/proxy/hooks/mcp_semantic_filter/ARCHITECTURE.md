# MCP Semantic Tool Filter Architecture

## Why Filter MCP Tools

When multiple MCP servers are connected, the proxy may expose hundreds of tools. Sending all tools in every request wastes context window tokens and increases cost. The semantic filter keeps only the top-K most relevant tools based on embedding similarity.

```mermaid
sequenceDiagram
    participant Client
    participant Hook as SemanticToolFilterHook
    participant Filter as SemanticMCPToolFilter
    participant Router as semantic-router
    participant LLM

    Client->>Hook: POST /chat/completions
    Note over Client,Hook: tools: [100+ MCP tools]
    Note over Client,Hook: messages: [{"role": "user", "content": "Get my Jira issues"}]
    
    rect rgb(240, 240, 240)
        Note over Hook: 1. Extract User Query
        Hook->>Filter: filter_tools("Get my Jira issues", tools)
    end
    
    rect rgb(240, 240, 240)
        Note over Filter: 2. Convert Tools → Routes
        Note over Filter: Tool name + description → Route
    end
    
    rect rgb(240, 240, 240)
        Note over Filter: 3. Semantic Matching
        Filter->>Router: router(query)
        Router->>Router: Embeddings + similarity
        Router-->>Filter: [top 10 matches]
    end
    
    rect rgb(240, 240, 240)
        Note over Filter: 4. Return Filtered Tools
        Filter-->>Hook: [10 relevant tools]
    end
    
    Hook->>LLM: POST /chat/completions
    Note over Hook,LLM: tools: [10 Jira-related tools] ← FILTERED
    Note over Hook,LLM: messages: [...] ← UNCHANGED
    
    LLM-->>Client: Response (unchanged)
```

## Filter Operations

The hook intercepts requests before they reach the LLM:

| Operation | Description |
|-----------|-------------|
| **Extract query** | Get user message from `messages[-1]` |
| **Convert to Routes** | Transform MCP tools into semantic-router Routes |
| **Semantic match** | Use `semantic-router` to find top-K similar tools |
| **Filter tools** | Replace request `tools` with filtered subset |

## Trigger Conditions

The filter only runs when:
- Call type is `completion` or `acompletion`
- Request contains `tools` field
- Request contains `messages` field
- Filter is enabled in config

## What Does NOT Change

- Request messages
- Response body
- Non-tool parameters

## Integration with semantic-router

Reuses existing LiteLLM infrastructure:
- `semantic-router` - Already an optional dependency
- `LiteLLMRouterEncoder` - Wraps `Router.aembedding()` for embeddings
- `SemanticRouter` - Handles similarity calculation and top-K selection

## Configuration

```yaml
litellm_settings:
  mcp_semantic_tool_filter:
    enabled: true
    embedding_model: "openai/text-embedding-3-small"
    top_k: 10
    similarity_threshold: 0.3
```

## Error Handling

The filter fails gracefully:
- If filtering fails → Return all tools (no impact on functionality)
- If query extraction fails → Skip filtering
- If no matches found → Return all tools
