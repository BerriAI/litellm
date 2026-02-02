# Semantic MCP Tool Auto-Filtering - Implementation Plan

## Overview

Filter MCP tools semantically before LLM inference to reduce context window bloat and improve tool selection accuracy. Leverage the existing LiteLLM semantic router infrastructure.

## Architecture

```
User Prompt → Embed Query → Semantic Search (Tool Index) → Filtered Tools → LLM
```

## Storage Decision: In-Memory (not DB)

**Rationale:**
- Typical embedding: 1536 floats × 4 bytes = ~6KB per tool
- 1000 tools = ~6MB, 10,000 tools = ~60MB — manageable in memory
- DB storage adds serialization overhead and query latency per request
- Embeddings are derived data (can be regenerated from tool descriptions)
- PostgreSQL BYTEA/JSON not optimized for vector similarity search

**Tradeoff:**
- On restart, embeddings regenerated (acceptable cold-start cost)
- For very large deployments (100k+ tools), consider external vector DB

## Components

### 1. MCP Tool Embedding Index (In-Memory)

**Location:** `litellm/proxy/_experimental/mcp_server/tool_embedding_index.py`

```python
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class ToolEmbeddingEntry:
    tool_name: str
    server_id: str
    description: str
    description_hash: str  # SHA256 of description for change detection
    embedding: List[float]

class MCPToolEmbeddingIndex:
    """In-memory semantic index for MCP tools."""
    
    def __init__(
        self,
        embedding_model: str,
        litellm_router_instance: "Router",
        similarity_threshold: float = 0.3,
    ):
        self.embedding_model = embedding_model
        self.router = litellm_router_instance
        self.similarity_threshold = similarity_threshold
        self._index: Dict[str, ToolEmbeddingEntry] = {}  # key: f"{server_id}:{tool_name}"
        self._embeddings_matrix: Optional[np.ndarray] = None  # cached for batch similarity
        self._index_keys: List[str] = []  # ordered keys matching matrix rows
        self._dirty = True  # rebuild matrix when True
    
    def _compute_hash(self, description: str) -> str:
        return hashlib.sha256(description.encode()).hexdigest()[:16]
    
    async def add_tool(self, tool_name: str, server_id: str, description: str) -> None:
        """Add or update a tool in the index."""
        key = f"{server_id}:{tool_name}"
        new_hash = self._compute_hash(description)
        
        # Skip if unchanged
        if key in self._index and self._index[key].description_hash == new_hash:
            return
        
        # Generate embedding
        embedding = await self._generate_embedding(description)
        
        self._index[key] = ToolEmbeddingEntry(
            tool_name=tool_name,
            server_id=server_id,
            description=description,
            description_hash=new_hash,
            embedding=embedding,
        )
        self._dirty = True
    
    async def add_tools_batch(self, tools: List[Tuple[str, str, str]]) -> None:
        """Batch add tools: [(tool_name, server_id, description), ...]"""
        to_embed = []
        for tool_name, server_id, description in tools:
            key = f"{server_id}:{tool_name}"
            new_hash = self._compute_hash(description)
            if key not in self._index or self._index[key].description_hash != new_hash:
                to_embed.append((key, tool_name, server_id, description, new_hash))
        
        if not to_embed:
            return
        
        # Batch embed
        descriptions = [t[3] for t in to_embed]
        embeddings = await self._generate_embeddings_batch(descriptions)
        
        for (key, tool_name, server_id, description, hash_val), embedding in zip(to_embed, embeddings):
            self._index[key] = ToolEmbeddingEntry(
                tool_name=tool_name,
                server_id=server_id,
                description=description,
                description_hash=hash_val,
                embedding=embedding,
            )
        self._dirty = True
    
    def remove_tool(self, tool_name: str, server_id: str) -> None:
        """Remove a tool from the index."""
        key = f"{server_id}:{tool_name}"
        if key in self._index:
            del self._index[key]
            self._dirty = True
    
    def remove_server_tools(self, server_id: str) -> None:
        """Remove all tools for a server."""
        keys_to_remove = [k for k in self._index if k.startswith(f"{server_id}:")]
        for key in keys_to_remove:
            del self._index[key]
        if keys_to_remove:
            self._dirty = True
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Search for tools similar to query.
        Returns: [(tool_name, server_id, similarity_score), ...]
        """
        if not self._index:
            return []
        
        threshold = threshold or self.similarity_threshold
        
        # Rebuild matrix if dirty
        if self._dirty:
            self._rebuild_matrix()
        
        # Embed query
        query_embedding = await self._generate_embedding(query)
        query_vec = np.array(query_embedding)
        
        # Cosine similarity
        similarities = np.dot(self._embeddings_matrix, query_vec) / (
            np.linalg.norm(self._embeddings_matrix, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Filter and sort
        results = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                entry = self._index[self._index_keys[idx]]
                results.append((entry.tool_name, entry.server_id, float(score)))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def _rebuild_matrix(self) -> None:
        """Rebuild the embeddings matrix for efficient batch similarity."""
        self._index_keys = list(self._index.keys())
        if self._index_keys:
            self._embeddings_matrix = np.array([
                self._index[k].embedding for k in self._index_keys
            ])
        else:
            self._embeddings_matrix = np.array([])
        self._dirty = False
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = await self.router.aembedding(
            model=self.embedding_model,
            input=[text],
        )
        return response.data[0]["embedding"]
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = await self.router.aembedding(
            model=self.embedding_model,
            input=texts,
        )
        return [item["embedding"] for item in response.data]
    
    def __len__(self) -> int:
        return len(self._index)
```

### 2. Semantic Tool Filter

**Location:** `litellm/proxy/_experimental/mcp_server/semantic_tool_filter.py`

```python
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set
from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool
    from litellm.router import Router

class SemanticMCPToolFilter:
    """Filters MCP tools based on semantic similarity to user query."""
    
    def __init__(
        self,
        embedding_model: str,
        litellm_router_instance: "Router",
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        enabled: bool = True,
    ):
        from litellm.proxy._experimental.mcp_server.tool_embedding_index import (
            MCPToolEmbeddingIndex,
        )
        
        self.enabled = enabled
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.index = MCPToolEmbeddingIndex(
            embedding_model=embedding_model,
            litellm_router_instance=litellm_router_instance,
            similarity_threshold=similarity_threshold,
        )
    
    async def index_tools(self, tools: List["MCPTool"], server_id: str) -> None:
        """Index tools from a server for semantic search."""
        if not self.enabled:
            return
        
        tool_data = [
            (tool.name, server_id, tool.description or tool.name)
            for tool in tools
        ]
        await self.index.add_tools_batch(tool_data)
        verbose_logger.debug(
            f"Indexed {len(tool_data)} tools from server {server_id}. "
            f"Total tools in index: {len(self.index)}"
        )
    
    def remove_server_tools(self, server_id: str) -> None:
        """Remove all tools for a server from the index."""
        self.index.remove_server_tools(server_id)
    
    async def filter_tools(
        self,
        query: str,
        available_tools: List["MCPTool"],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List["MCPTool"]:
        """
        Filter tools to only those semantically relevant to the query.
        
        Args:
            query: User's message/query
            available_tools: Full list of available MCP tools
            top_k: Max tools to return (default: self.top_k)
            threshold: Min similarity score (default: self.similarity_threshold)
        
        Returns:
            Filtered list of tools sorted by relevance
        """
        if not self.enabled or not available_tools:
            return available_tools
        
        top_k = top_k or self.top_k
        threshold = threshold or self.similarity_threshold
        
        try:
            # Search for relevant tools
            search_results = await self.index.search(
                query=query,
                top_k=top_k,
                threshold=threshold,
            )
            
            if not search_results:
                verbose_logger.debug(
                    f"No tools matched query with threshold {threshold}. "
                    "Returning all tools as fallback."
                )
                return available_tools
            
            # Build set of relevant tool names
            relevant_tools: Set[str] = {name for name, _, _ in search_results}
            
            # Filter and preserve order by relevance
            tool_by_name = {tool.name: tool for tool in available_tools}
            filtered = []
            for tool_name, _, score in search_results:
                if tool_name in tool_by_name:
                    filtered.append(tool_by_name[tool_name])
                    verbose_logger.debug(
                        f"Tool '{tool_name}' matched with score {score:.3f}"
                    )
            
            verbose_logger.info(
                f"Semantic filter: {len(available_tools)} -> {len(filtered)} tools"
            )
            return filtered
            
        except Exception as e:
            verbose_logger.warning(
                f"Semantic tool filter failed: {e}. Returning all tools."
            )
            return available_tools
    
    def extract_user_query(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the user's query from messages for semantic search."""
        # Get the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle content blocks
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            texts.append(block)
                    return " ".join(texts)
        return ""
```

### 3. Integration Hook: `async_pre_call_hook`

**Location:** New CustomLogger subclass or extend existing proxy hook

The `async_pre_call_hook` in `CustomLogger` is the right place to filter tools before the LLM call:

```python
# In litellm/proxy/hooks/semantic_tool_filter_hook.py

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.caching.caching import DualCache

class SemanticToolFilterHook(CustomLogger):
    """Pre-call hook that filters MCP tools semantically."""
    
    def __init__(self, semantic_filter: "SemanticMCPToolFilter"):
        super().__init__()
        self.filter = semantic_filter
    
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """Filter tools before LLM call based on user query."""
        
        # Only filter for chat completions with tools
        if call_type not in ("completion", "acompletion"):
            return None
        
        tools = data.get("tools")
        if not tools:
            return None
        
        messages = data.get("messages", [])
        if not messages:
            return None
        
        # Extract user query
        user_query = self.filter.extract_user_query(messages)
        if not user_query:
            return None
        
        try:
            # Filter tools semantically
            filtered_tools = await self.filter.filter_tools(
                query=user_query,
                available_tools=tools,
            )
            
            # Update data with filtered tools
            data["tools"] = filtered_tools
            
            verbose_proxy_logger.debug(
                f"Semantic tool filter: {len(tools)} -> {len(filtered_tools)} tools"
            )
            
            return data
            
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Semantic tool filter hook failed: {e}. Proceeding with all tools."
            )
            return None
```

### 4. Embedding Lifecycle Integration

**Modify:** `litellm/proxy/_experimental/mcp_server/mcp_server_manager.py`

```python
# Add to MCPServerManager class

async def _index_server_tools(self, server: MCPServer) -> None:
    """Index tools from a server for semantic search."""
    if not hasattr(self, 'semantic_filter') or self.semantic_filter is None:
        return
    
    try:
        tools = await self._get_tools_from_server(server)
        await self.semantic_filter.index_tools(tools, server.server_id)
    except Exception as e:
        verbose_logger.warning(f"Failed to index tools for {server.name}: {e}")

# Modify add_server()
async def add_server(self, mcp_server: LiteLLM_MCPServerTable):
    # ... existing code ...
    self.registry[mcp_server.server_id] = new_server
    
    # Index tools for semantic search
    await self._index_server_tools(new_server)

# Modify update_server()
async def update_server(self, mcp_server: LiteLLM_MCPServerTable):
    # ... existing code ...
    
    # Re-index tools (hash check inside will skip unchanged)
    await self._index_server_tools(new_server)

# Modify remove_server()
def remove_server(self, mcp_server: LiteLLM_MCPServerTable):
    # ... existing code ...
    
    # Remove from semantic index
    if hasattr(self, 'semantic_filter') and self.semantic_filter:
        self.semantic_filter.remove_server_tools(mcp_server.server_id)
```

### 5. Configuration

**Config YAML:**
```yaml
litellm_settings:
  mcp_semantic_tool_filter:
    enabled: true
    embedding_model: "openai/text-embedding-3-small"  # or any LiteLLM-supported model
    top_k: 10
    similarity_threshold: 0.3
```

**Environment Variables:**
```bash
LITELLM_MCP_SEMANTIC_FILTER_ENABLED=true
LITELLM_MCP_SEMANTIC_FILTER_EMBEDDING_MODEL=openai/text-embedding-3-small
LITELLM_MCP_SEMANTIC_FILTER_TOP_K=10
LITELLM_MCP_SEMANTIC_FILTER_THRESHOLD=0.3
```

## Implementation Order

1. **`tool_embedding_index.py`** - Core in-memory index with hash-based change detection
2. **`semantic_tool_filter.py`** - Filter logic with graceful fallback
3. **`semantic_tool_filter_hook.py`** - CustomLogger hook for `async_pre_call_hook`
4. **MCPServerManager integration** - Add/update/remove hooks
5. **Proxy startup wiring** - Initialize filter, register hook
6. **Tests** - Unit + integration tests

## Files to Create/Modify

| File | Action |
|------|--------|
| `litellm/proxy/_experimental/mcp_server/tool_embedding_index.py` | **New** |
| `litellm/proxy/_experimental/mcp_server/semantic_tool_filter.py` | **New** |
| `litellm/proxy/hooks/semantic_tool_filter_hook.py` | **New** |
| `litellm/proxy/_experimental/mcp_server/mcp_server_manager.py` | Modify |
| `litellm/proxy/proxy_server.py` | Modify (init filter, register hook) |
| `litellm/proxy/_types.py` | Add config types |
| `tests/proxy_unit_tests/test_mcp_semantic_filter.py` | **New** |

## Key Design Decisions

1. **In-memory storage** - Fast, no DB overhead, acceptable cold-start cost
2. **Hash-based change detection** - Only re-embed when descriptions change
3. **Graceful degradation** - Return all tools if filter fails
4. **`async_pre_call_hook`** - Standard hook pattern, runs before LLM call
5. **Reuse `LiteLLMRouterEncoder`** infrastructure for embeddings
6. **NumPy for similarity** - Efficient batch cosine similarity computation
