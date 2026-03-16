---
name: litellm-context7
description: "Fetch up-to-date LiteLLM documentation via Context7 before writing code. Use when working with LiteLLM provider implementations, proxy configuration, SDK usage, or any LiteLLM API. Ensures code examples use current APIs instead of outdated training data."
---

# LiteLLM Documentation via Context7

Always fetch current LiteLLM documentation from Context7 before writing or modifying code. This prevents hallucinated APIs and ensures you're using the latest patterns.

## When to Use

- Implementing or modifying LLM provider integrations
- Configuring the LiteLLM Proxy (LLM Gateway)
- Writing code that calls `litellm.completion()`, `litellm.embedding()`, or other SDK methods
- Adding new providers or updating existing provider transformations
- Working with proxy features (auth, rate limiting, budgets, load balancing)
- Answering questions about LiteLLM configuration or API usage
- Writing or updating documentation

## How to Fetch Docs

### Option 1: Context7 MCP (if available)

If you have the Context7 MCP server configured, use the `query-docs` tool:

```
resolve-library-id: libraryName="litellm"
query-docs: libraryId="/berriai/litellm" query="your specific question"
```

### Option 2: Context7 CLI

```bash
# Search for a specific topic
npx ctx7 docs /berriai/litellm "how to add a new provider"

# Search for proxy configuration
npx ctx7 docs /berriai/litellm "proxy config yaml model_list"

# Search for SDK usage
npx ctx7 docs /berriai/litellm "completion function parameters"
```

### Option 3: Add `use context7` to your prompt

Simply append `use context7` to any prompt about LiteLLM to trigger documentation lookup.

## Key LiteLLM Patterns (Quick Reference)

These are common patterns — always verify against fetched docs for the latest API.

### Provider Model Format
```
provider/model-name
```
Examples: `anthropic/claude-3-opus`, `openai/gpt-4`, `azure/gpt-4-turbo`, `bedrock/anthropic.claude-3`

### SDK Completion Call
```python
import litellm

response = litellm.completion(
    model="anthropic/claude-3-opus",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Proxy Config (config.yaml)
```yaml
model_list:
  - model_name: my-model
    litellm_params:
      model: provider/model-name
      api_key: os.environ/API_KEY
```

### Start Proxy
```bash
litellm --config config.yaml
```

## Repository Structure (for contributors)

When fetching docs, know which area of the codebase you're working in:

- `litellm/llms/` — Provider-specific implementations
- `litellm/proxy/` — Proxy server (LLM Gateway)
- `litellm/router.py` + `litellm/router_utils/` — Load balancing and fallbacks
- `litellm/types/` — Type definitions and schemas
- `litellm/integrations/` — Third-party integrations
- `docs/my-website/docs/providers/` — Provider documentation
- `tests/test_litellm/` — Unit tests

## What to Fetch Docs For

| Task | Suggested Query |
|------|----------------|
| Adding a provider | `"how to add a new LLM provider"` |
| Provider transformations | `"BaseConfig transformation class"` |
| Proxy authentication | `"proxy authentication api keys"` |
| Rate limiting | `"rate limiting budget management"` |
| Streaming | `"streaming responses async"` |
| Function/tool calling | `"function calling tool use"` |
| Error handling | `"exception handling error types"` |
| Caching | `"caching redis in-memory"` |
| Load balancing | `"router load balancing fallback"` |
| Embeddings | `"embedding function usage"` |
| Image generation | `"image generation providers"` |
| Cost tracking | `"cost tracking spend logs"` |
| MCP integration | `"MCP model context protocol"` |

## Development Rules (from AGENTS.md)

When contributing code changes based on fetched docs:

1. **Always fetch before implementing** — don't rely on memory for API signatures or config formats
2. **Follow existing patterns** — check `litellm/llms/{provider}/` for provider implementations
3. **Type safety** — use proper type hints, update `litellm/types/` as needed
4. **Testing** — add tests in `tests/test_litellm/`, run `make test-unit`
5. **Proxy DB access** — use Prisma model methods, never raw SQL
6. **Don't hardcode model flags** — put model-specific capabilities in `model_prices_and_context_window.json`
7. **Verify proxy config syntax** — use `os.environ/KEY_NAME` (not `$KEY_NAME`) for env vars in YAML
8. **Use the library ID** `/berriai/litellm` — this is the canonical Context7 identifier
