# MCP Semantic Tool Filtering (Experimental)

LiteLLM Proxy can automatically shortlist the most relevant MCP tools for each `/chat/completions` call by embedding tool descriptions and running semantic search over them.

> **Experimental:** This subsystem is in active development; expect breaking changes while we iterate with early adopters.

## When to Use

Enable semantic filtering when:

- You expose many MCP servers/tools and want the proxy to pick only the few that match the user query.
- You need deterministic behavior across providers (Anthropic, OpenAI, etc.) without hand-curating tool lists per request.
- You can tolerate the small CPU/memory cost of embedding tool metadata at startup.

If you only have a handful of tools or already specify them explicitly in requests, keep the feature disabled to avoid extra overhead.

## Installation

Pick and install the vector store backend you plan to use (see [Storage Backends](#storage-backends)). Each backend provides its own installation instructions; for example, the FAISS backend is bundled with `pip install "litellm[semantic-mcp-filter]"`.

## Configuration

Declare the feature under `litellm_settings.semantic_mcp_filter` inside your proxy config:

```yaml
litellm_settings:
  semantic_mcp_filter:
    enabled: true
    embedding:
      model: text-embedding-3-large
      parameters:
        api_key: os.environ/OPENAI_API_KEY
    vector_store:
      backend: faiss
      metric: cosine
      top_k: 4
```

Key fields:

- `enabled`: master switch.
- `embedding`: which embedding model to call via `litellm.embedding` plus any provider-specific kwargs.
- `vector_store`: backend options (currently FAISS), metric, and default `top_k`.
- `include_servers`: optional allowlist of MCP server labels to index.

When the proxy boots, it enumerates all MCP servers (config + DB), fetches their tools, embeds the combined `name + description + schema`, and stores the vectors in FAISS. Subsequent MCP server additions/removals trigger incremental re-embedding only for the affected tools.

## Request-Time Controls

Clients can opt in/out per `/chat/completions` request by adding a `semantic_filter` block to the MCP tool definition:

```json
{
  "type": "mcp",
  "server_label": "litellm",
  "server_url": "litellm_proxy/mcp/",
  "require_approval": "never",
  "semantic_filter": {
    "top_k": 3,
    "server_labels": ["everything"]
  }
}
```

Supported options:

- `top_k`: override how many tools to return for this request.
- `server_labels`: narrow the candidate MCP servers for similarity search.

The proxy extracts the latest user message, embeds it with the configured model, queries the configured vector store, and inserts the winning tools back into the OpenAI-format `tools` array before calling the upstream model. If filtering fails (e.g., embedding error), the proxy logs a warning and falls back to the full tool set to avoid blocking the request.

## Storage Backends

Semantic filtering is storage-agnostic. Each backend adheres to the same interface (upserts, deletes, similarity search) so you can drop in alternative vector databases over time. Currently we ship a FAISS backend and expect to add more as user needs evolve.

### FAISS

FAISS is the default proof-of-concept backend: fast, in-memory, and ideal for on-prem deployments without extra infrastructure.

**Install**

```bash
pip install "litellm[semantic-mcp-filter]"
```

This installs `faiss-cpu`. If FAISS cannot be imported when the proxy starts, semantic filtering is automatically disabled.

**Configure**

Set the vector store backend to `faiss` and pick a metric. Example snippet:

```yaml
litellm_settings:
  semantic_mcp_filter:
    vector_store:
      backend: faiss
      metric: cosine
      top_k: 4
```

The FAISS backend keeps everything in memory; persistency hooks will be added alongside future storage engines.
