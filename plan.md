# Semantic MCP Tool Auto-Filtering - Implementation Plan

## Overview

Filter MCP tools semantically before LLM inference to reduce context window bloat and improve tool selection accuracy. Leverage the existing LiteLLM semantic router infrastructure.

## Architecture

```
User Prompt → Embed Query → Semantic Search (Tool Index) → Filtered Tools → LLM
```

## Components

### 1. MCP Tool Embedding Index

**Location:** `litellm/proxy/_experimental/mcp_server/tool_embedding_index.py`

- Store tool embeddings with metadata (name, description, server_id)
- Support configurable embedding model via `LiteLLMRouterEncoder`
- Hash tool descriptions to detect changes

**Schema (in-memory + DB persistence):**
```python
class MCPToolEmbedding:
    tool_name: str
    server_id: str
    description: str
    description_hash: str
    embedding: List[float]
    updated_at: datetime
```

### 2. Embedding Lifecycle Management

**When to create/update embeddings:**

| Event | Action |
|-------|--------|
| Proxy startup | Load from DB; generate missing |
| MCP server added (runtime) | Generate embeddings for new tools |
| MCP server updated | Re-embed if description_hash changed |
| MCP server removed | Delete embeddings |
| Periodic sync | Background task to catch drift |

**Integration points:**
- `MCPServerManager.add_server()` → trigger embedding generation
- `MCPServerManager.update_server()` → check hash, re-embed if needed
- `MCPServerManager.remove_server()` → delete embeddings

### 3. Persistence Layer

**Database table:** `LiteLLM_MCPToolEmbedding`

```sql
CREATE TABLE LiteLLM_MCPToolEmbedding (
    id UUID PRIMARY KEY,
    tool_name VARCHAR NOT NULL,
    server_id VARCHAR NOT NULL,
    description TEXT,
    description_hash VARCHAR,
    embedding BYTEA,  -- serialized float array
    updated_at TIMESTAMP,
    UNIQUE(tool_name, server_id)
);
```

**Prisma schema addition:** `litellm/proxy/schema.prisma`

### 4. Semantic Filter Hook

**Location:** `litellm/proxy/_experimental/mcp_server/semantic_tool_filter.py`

```python
class SemanticMCPToolFilter:
    def __init__(
        self,
        embedding_model: str,
        router_instance: Router,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
    ):
        self.encoder = LiteLLMRouterEncoder(router_instance, embedding_model)
        self.index = MCPToolEmbeddingIndex()
    
    async def filter_tools(
        self,
        user_query: str,
        available_tools: List[MCPTool],
    ) -> List[MCPTool]:
        # 1. Embed user query
        # 2. Compute similarities against tool index
        # 3. Return top-k or above threshold
```

### 5. Integration with Existing Flows

**Option A: Pre-call hook (recommended)**
- Hook into `list_tools()` flow when tools are being injected
- Intercept at `proxy_server.py` `/chat/completions` before tool injection

**Option B: New endpoint parameter**
- Add `semantic_tool_filter: bool` to `/chat/completions`
- Add `tool_filter_top_k` and `tool_filter_threshold` params

### 6. Configuration

**Config YAML:**
```yaml
litellm_settings:
  mcp_semantic_tool_filter:
    enabled: true
    embedding_model: "openai/text-embedding-3-small"
    top_k: 10
    similarity_threshold: 0.3
    cache_embeddings: true
```

## Implementation Order

1. **MCPToolEmbeddingIndex** - In-memory index with CRUD ops
2. **Database persistence** - Prisma schema + read/write
3. **Embedding generation hooks** - Integrate with MCPServerManager
4. **SemanticMCPToolFilter** - Query-time filtering logic
5. **Proxy integration** - Wire into `/chat/completions` flow
6. **Tests** - Unit + integration tests

## Key Design Decisions

- **Reuse `LiteLLMRouterEncoder`** from existing semantic router
- **Hash-based change detection** for efficient updates
- **Async embedding generation** to not block server startup
- **Graceful degradation** - return all tools if embedding fails

## Files to Modify/Create

| File | Action |
|------|--------|
| `litellm/proxy/schema.prisma` | Add MCPToolEmbedding table |
| `litellm/proxy/_experimental/mcp_server/tool_embedding_index.py` | New - index class |
| `litellm/proxy/_experimental/mcp_server/semantic_tool_filter.py` | New - filter logic |
| `litellm/proxy/_experimental/mcp_server/mcp_server_manager.py` | Hook embedding lifecycle |
| `litellm/proxy/proxy_server.py` | Integrate filter at completion endpoint |
| `litellm/proxy/_types.py` | Add config types |
| `tests/proxy_unit_tests/test_mcp_semantic_filter.py` | New - tests |
