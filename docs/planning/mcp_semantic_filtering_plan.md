# MCP Semantic Filtering: `defer_loading` + Tool Search Integration Plan

## Overview

Enable LiteLLM users to configure MCP tools with **deferred loading** (`defer_loading: true`) at the Virtual Key / MCP server level, so that when a request is made to the LLM, deferred tools are **not exposed** in the initial tool list. Instead, the LLM sees a **tool search tool** (e.g., `tool_search_tool_regex`) alongside non-deferred tools. When the LLM needs additional tools, it invokes the tool search tool, and LiteLLM returns the 3-5 most semantically relevant deferred tools as `tool_reference` blocks for the LLM to select from.

---

## Current State

### What exists today

| Component | File(s) | Status |
|-----------|---------|--------|
| **Semantic tool filter** | `litellm/proxy/_experimental/mcp_server/semantic_tool_filter.py` | Filters ALL tools pre-call via embedding similarity. Works as a blunt filter — reduces N tools to top-K before LLM sees any. |
| **Semantic filter hook** | `litellm/proxy/hooks/mcp_semantic_filter/hook.py` | `async_pre_call_hook` that expands MCP references, runs semantic filter, replaces `data["tools"]`. |
| **MCP server model** | `litellm/types/mcp_server/mcp_server_manager.py` | `MCPServer` has `allowed_tools` / `disallowed_tools` but no `defer_loading` per-tool config. |
| **Anthropic `defer_loading`** | `litellm/types/llms/anthropic.py` (`AnthropicMessagesTool`) | Supported as a pass-through field in the Anthropic tool type. Transformation in `litellm/llms/anthropic/chat/transformation.py`. |
| **Anthropic tool search types** | `litellm/types/llms/anthropic_tool_search.py` | Beta header config for `tool_search_tool_regex` / `tool_search_tool_bm25`. |
| **MCP tool configuration UI** | `ui/litellm-dashboard/src/components/mcp_tools/mcp_tool_configuration.tsx` | Checkbox list of tools with Enable/Disable. No defer_loading toggle. |
| **MCP permission management UI** | `ui/litellm-dashboard/src/components/mcp_tools/MCPPermissionManagement.tsx` | Access groups, allow_all_keys, extra headers. |
| **LiteLLM Proxy MCP handler** | `litellm/responses/mcp/litellm_proxy_mcp_handler.py` | Handles `mcp` tool type with `server_url="litellm_proxy"`. Expands MCP refs to OpenAI function defs. |

### What's missing

1. **Per-tool `defer_loading` configuration** on the MCP server/virtual key — no way to mark individual MCP tools as deferred.
2. **Tool search tool injection** — the semantic filter currently replaces the full tools list pre-call rather than injecting a tool search tool for the LLM to call on-demand.
3. **Tool search callback handling** — when the LLM invokes `tool_search_tool_regex` or `tool_search_tool_bm25`, LiteLLM needs to intercept that, run semantic search over deferred tools, and return `tool_reference` blocks.
4. **UI for defer_loading** — no toggle in the MCP tool configuration UI.
5. **UI for semantic filter settings** — no way to configure embedding model, top_k, similarity threshold from the dashboard.

---

## Proposed Architecture

### End-to-End Flow

```
┌──────────────────────────────────────────────────────────────┐
│  1. Admin configures MCP server tools                        │
│     - tool A: defer_loading = false (always visible)         │
│     - tool B: defer_loading = true  (hidden, searchable)     │
│     - tool C: defer_loading = true  (hidden, searchable)     │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  2. Client sends request with MCP tools                      │
│     tools: [{ type: "mcp", server_url: "litellm_proxy" }]   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  3. SemanticToolFilterHook (pre_call_hook) processes tools    │
│     a. Expand MCP references → full tool definitions         │
│     b. Split into:                                           │
│        - non_deferred_tools (defer_loading=false or unset)   │
│        - deferred_tools (defer_loading=true)                 │
│     c. Build/use semantic index for deferred_tools           │
│     d. Inject tool_search_tool into tools list               │
│     e. Final tools = [tool_search_tool] + non_deferred_tools │
│     f. Store deferred_tools in request metadata for later     │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  4. LLM receives: tool_search_tool + non-deferred tools      │
│     LLM decides it needs more tools → calls tool_search_tool │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│  5. LiteLLM intercepts tool_search_tool call                 │
│     a. Extract search query/regex from tool call args        │
│     b. Run semantic search over deferred_tools               │
│     c. Return 3-5 tool_reference blocks to LLM              │
│     d. LLM selects and invokes discovered tools              │
└──────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Backend — Per-Tool `defer_loading` Configuration

#### 1.1 Extend MCPServer model with `deferred_tools` field

**File:** `litellm/types/mcp_server/mcp_server_manager.py`

Add a new field to `MCPServer`:

```python
class MCPServer(BaseModel):
    ...
    deferred_tools: Optional[List[str]] = None
    # List of tool names that should have defer_loading=true.
    # These tools will NOT appear in the initial tool list sent to the LLM.
    # Instead, they will be discoverable via the tool_search_tool.
```

**Why a separate list instead of per-tool objects?** The current `allowed_tools`/`disallowed_tools` pattern uses simple string lists. A `deferred_tools` list follows the same convention and avoids breaking the existing schema. The alternative (a dict mapping tool names to config objects) is more flexible but adds schema migration complexity.

#### 1.2 Extend the database schema

**File:** `litellm/proxy/schema.prisma` — `LiteLLM_MCPServerTable`

Add `deferred_tools String[]` column (or JSON field depending on DB backend).

Run `prisma migrate dev` to generate migration.

#### 1.3 Update MCP management endpoints

**File:** `litellm/proxy/management_endpoints/mcp_management_endpoints.py`

- `POST /v1/mcp/server` — accept `deferred_tools` in create payload
- `PUT /v1/mcp/server/{server_id}` — accept `deferred_tools` in update payload
- `GET /v1/mcp/server` — return `deferred_tools` in response

#### 1.4 Update MCP tool listing to annotate deferred tools

**File:** `litellm/proxy/_experimental/mcp_server/mcp_server_manager.py`

When `get_tools_for_server()` returns tools, annotate each tool with `defer_loading=True` if its name is in `server.deferred_tools`.

---

### Phase 2: Backend — Tool Search Tool Injection

#### 2.1 Update the semantic filter hook to split deferred vs non-deferred

**File:** `litellm/proxy/hooks/mcp_semantic_filter/hook.py`

Modify `async_pre_call_hook`:

```python
async def async_pre_call_hook(self, ...):
    # ... existing expansion logic ...

    # NEW: Split tools by defer_loading
    deferred_tools = []
    non_deferred_tools = []
    for tool in expanded_tools:
        if self._is_deferred(tool):
            deferred_tools.append(tool)
        else:
            non_deferred_tools.append(tool)

    if deferred_tools:
        # Inject tool_search_tool
        tool_search_tool = self._build_tool_search_tool(
            variant="regex"  # or configurable: "regex" | "bm25"
        )
        data["tools"] = [tool_search_tool] + non_deferred_tools

        # Store deferred tools in metadata for later retrieval
        metadata = data.get("metadata", {})
        metadata["_deferred_mcp_tools"] = deferred_tools
        data["metadata"] = metadata
    else:
        # No deferred tools — apply existing semantic filtering
        # (current behavior: top-K filter over all tools)
        ...
```

#### 2.2 Build tool_search_tool definition

**File:** `litellm/proxy/hooks/mcp_semantic_filter/hook.py` (or new utility)

```python
def _build_tool_search_tool(self, variant: str = "regex") -> dict:
    """Build the tool_search_tool definition for injection."""
    if variant == "regex":
        return {
            "type": "tool_search_tool_regex_20251119",
            "name": "tool_search_tool_regex",
        }
    elif variant == "bm25":
        return {
            "type": "tool_search_tool_bm25_20251119",
            "name": "tool_search_tool_bm25",
        }
```

**Provider compatibility note:** Tool search is currently only supported by Anthropic, Vertex AI, and Bedrock (for Anthropic models). For non-Anthropic providers, we should fall back to the existing semantic pre-filter behavior (top-K). The hook should check `data.get("model")` or the provider to decide.

#### 2.3 Mark deferred tools with `defer_loading: true`

When building the full tool list for Anthropic-compatible providers, the deferred tools that ARE included (e.g., when tool_search returns them) must have `defer_loading: true` set. This is already supported in the Anthropic transformation layer (`litellm/llms/anthropic/chat/transformation.py`).

For the initial request, deferred tools are NOT included at all — they're held in metadata. They only appear when the tool_search_tool returns `tool_reference` blocks.

---

### Phase 3: Backend — Tool Search Callback Handling

#### 3.1 Intercept tool_search_tool invocations

When the LLM calls `tool_search_tool_regex` or `tool_search_tool_bm25`, LiteLLM's response processing needs to handle this. There are two approaches:

**Option A: Provider-native tool search (preferred for Anthropic)**

For Anthropic/Vertex/Bedrock, pass `tool_search_tool` and all deferred tools (with `defer_loading: true`) directly to the provider API. The provider handles the search natively. LiteLLM just needs to:

1. Include `tool_search_tool` in the tools list
2. Include all deferred tools with `defer_loading: true`
3. Set the appropriate beta header (already in `litellm/types/llms/anthropic_tool_search.py`)

This is simpler and leverages the provider's native capability.

**Option B: LiteLLM-managed tool search (for non-Anthropic providers)**

For providers that don't support tool_search natively, LiteLLM manages the search:

1. Pre-call hook removes deferred tools, injects a custom "search_tools" function tool
2. When LLM calls the search function, a post-processing hook intercepts it
3. Hook runs semantic search over deferred tools using `SemanticMCPToolFilter`
4. Returns matched tool definitions back to the LLM in a follow-up turn

**File locations for Option B:**
- `litellm/proxy/hooks/mcp_semantic_filter/hook.py` — add response interception
- `litellm/proxy/_experimental/mcp_server/semantic_tool_filter.py` — reuse `filter_tools()`

#### 3.2 Handle the search result → tool_reference flow

For **Option A** (Anthropic-native), the flow is:

```python
# In the pre-call hook, for Anthropic-compatible providers:
if has_deferred_tools and is_anthropic_compatible(model):
    # Include ALL tools, but mark deferred ones
    all_tools_with_defer = []
    for tool in non_deferred_tools:
        all_tools_with_defer.append(tool)
    for tool in deferred_tools:
        tool["defer_loading"] = True
        all_tools_with_defer.append(tool)

    # Inject tool_search_tool
    tool_search = self._build_tool_search_tool(variant)
    data["tools"] = [tool_search] + all_tools_with_defer
```

The provider API handles:
- Not showing deferred tools initially
- Running search when LLM invokes tool_search_tool
- Returning tool_reference blocks
- LLM selecting and calling discovered tools

For **Option B** (LiteLLM-managed), additional work is needed in Phase 4.

---

### Phase 4: Backend — LiteLLM-Managed Tool Search (Non-Anthropic Providers)

#### 4.1 Create a synthetic tool search function

**New file:** `litellm/proxy/hooks/mcp_semantic_filter/tool_search_function.py`

```python
LITELLM_TOOL_SEARCH_FUNCTION = {
    "type": "function",
    "function": {
        "name": "litellm_tool_search",
        "description": "Search for additional tools by describing what you need. Returns relevant tool definitions you can then call.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of the tool capability you're looking for"
                }
            },
            "required": ["query"]
        }
    }
}
```

#### 4.2 Intercept tool call responses

**File:** `litellm/proxy/hooks/mcp_semantic_filter/hook.py`

Add `async_post_call_hook` or modify the completion response processing:

```python
async def async_log_success_event(self, ...):
    """Check if the LLM called litellm_tool_search and inject tool definitions."""
    # If response contains a tool_call to "litellm_tool_search":
    #   1. Extract query from tool call arguments
    #   2. Run self.filter.filter_tools(query, deferred_tools, top_k=5)
    #   3. Format matched tools as a tool response message
    #   4. (Requires re-calling the LLM with the expanded tool list)
```

**Important:** This creates a multi-turn conversation pattern. The proxy would need to:
1. Detect `litellm_tool_search` in the response
2. Auto-respond with matched tool definitions
3. Add matched tools to the tools list
4. Re-call the LLM with the updated context

This is more complex and should be considered a Phase 4 / follow-up item.

---

### Phase 5: UI — Defer Loading Toggle

#### 5.1 Add defer_loading toggle to tool configuration

**File:** `ui/litellm-dashboard/src/components/mcp_tools/mcp_tool_configuration.tsx`

Extend the existing tool list to include a second toggle per tool:

```
┌─────────────────────────────────────────────────────────┐
│ Tool Configuration                                  3/5 │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ☑ get_weather                          [Enabled]    │ │
│ │   Get the weather at a specific location            │ │
│ │   ☐ Defer loading (discoverable via tool search)    │ │
│ └─────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ☑ send_email                           [Enabled]    │ │
│ │   Send an email to a recipient                      │ │
│ │   ☑ Defer loading (discoverable via tool search)    │ │
│ └─────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ☑ create_calendar_event                [Enabled]    │ │
│ │   Create a new calendar event                       │ │
│ │   ☑ Defer loading (discoverable via tool search)    │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ ℹ️ Deferred tools are not shown to the LLM initially.  │
│   The LLM can discover them via tool search when needed.│
└─────────────────────────────────────────────────────────┘
```

**Changes:**
- Add `deferredTools: string[]` state alongside `allowedTools`
- Add a secondary checkbox "Defer loading" for each enabled tool
- Only show the defer checkbox for enabled tools
- Pass `deferred_tools` to the create/update API call

#### 5.2 Add defer_loading column props

**File:** `ui/litellm-dashboard/src/components/mcp_tools/mcp_tool_configuration.tsx`

```typescript
interface MCPToolConfigurationProps {
  accessToken: string | null;
  oauthAccessToken?: string | null;
  formValues: Record<string, any>;
  allowedTools: string[];
  existingAllowedTools: string[] | null;
  onAllowedToolsChange: (tools: string[]) => void;
  // NEW:
  deferredTools: string[];
  existingDeferredTools: string[] | null;
  onDeferredToolsChange: (tools: string[]) => void;
}
```

#### 5.3 Update create/edit MCP server forms

**Files:**
- `ui/litellm-dashboard/src/components/mcp_tools/create_mcp_server.tsx`
- `ui/litellm-dashboard/src/components/mcp_tools/mcp_server_edit.tsx`

Add `deferred_tools` to the form data and API payload.

#### 5.4 Display deferred status in tool list/view

**Files:**
- `ui/litellm-dashboard/src/components/mcp_tools/mcp_tools.tsx`
- `ui/litellm-dashboard/src/components/mcp_tools/mcp_server_view.tsx`

Show a badge or tag indicating which tools are deferred.

---

### Phase 6: UI — Semantic Filter Configuration

#### 6.1 Add semantic filter settings panel

**New component:** `ui/litellm-dashboard/src/components/mcp_tools/semantic_filter_settings.tsx`

This panel allows admins to configure the semantic filtering behavior:

```
┌─────────────────────────────────────────────────────────┐
│ Semantic Tool Search Settings                           │
├─────────────────────────────────────────────────────────┤
│ Enabled:            [Toggle: ON]                        │
│ Embedding Model:    [text-embedding-3-small ▼]          │
│ Max Results (top_k): [5]                                │
│ Similarity Threshold: [0.3]                             │
│ Search Variant:     [● Regex  ○ BM25]                   │
│                                                         │
│ [Save Settings]                                         │
└─────────────────────────────────────────────────────────┘
```

#### 6.2 Backend endpoint for semantic filter config

**File:** `litellm/proxy/management_endpoints/mcp_management_endpoints.py`

```
GET  /v1/mcp/semantic_filter/settings — get current config
PUT  /v1/mcp/semantic_filter/settings — update config
```

These read/write from `litellm_settings.mcp_semantic_tool_filter` in the proxy config.

---

## Configuration Examples

### YAML Config (proxy_config.yaml)

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-5-20250929

litellm_settings:
  mcp_semantic_tool_filter:
    enabled: true
    embedding_model: "text-embedding-3-small"
    top_k: 5
    similarity_threshold: 0.3
    tool_search_variant: "regex"  # NEW: "regex" or "bm25"
    # NEW: when true, uses provider-native tool search for compatible providers
    # when false, always uses LiteLLM-managed search
    prefer_native_tool_search: true
```

### MCP Server Config with Deferred Tools

```yaml
mcp_servers:
  - server_id: "zapier"
    name: "Zapier MCP"
    url: "https://mcp.zapier.com/sse"
    transport: "sse"
    allowed_tools:
      - "create_zap"
      - "send_email"
      - "create_spreadsheet_row"
      - "send_slack_message"
      - "create_trello_card"
    deferred_tools:  # NEW
      - "create_spreadsheet_row"
      - "send_slack_message"
      - "create_trello_card"
    # Result: LLM sees create_zap + send_email immediately
    #         create_spreadsheet_row, send_slack_message, create_trello_card
    #         are discoverable via tool search
```

### Client Request Example

```python
import litellm

response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[
        {"role": "user", "content": "Add a row to my expenses spreadsheet"}
    ],
    tools=[
        {"type": "mcp", "server_url": "litellm_proxy"}
    ],
)
# LiteLLM proxy will:
# 1. Expand MCP tools from Zapier server
# 2. Include create_zap + send_email as regular tools
# 3. Include tool_search_tool_regex
# 4. Include create_spreadsheet_row, send_slack_message, create_trello_card
#    with defer_loading=true
# 5. Claude searches and finds create_spreadsheet_row
# 6. Claude calls create_spreadsheet_row
```

---

## Testing Plan

### Unit Tests

| Test | File | Description |
|------|------|-------------|
| `test_deferred_tools_split` | `tests/test_litellm/proxy/_experimental/mcp_server/test_semantic_tool_filter.py` | Verify tools are correctly split into deferred/non-deferred |
| `test_tool_search_injection` | same | Verify tool_search_tool is injected when deferred tools exist |
| `test_no_injection_without_deferred` | same | Verify no tool_search_tool when no deferred tools |
| `test_provider_native_vs_managed` | same | Verify correct path chosen based on provider |
| `test_defer_loading_flag_passthrough` | `tests/test_litellm/llms/anthropic/chat/test_anthropic_chat_transformation.py` | Verify `defer_loading: true` is passed to Anthropic API |
| `test_deferred_tools_crud` | `tests/proxy_unit_tests/test_mcp_management.py` | Verify create/update/get with deferred_tools field |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_e2e_deferred_tool_discovery` | Full flow: configure deferred tools → make request → LLM uses tool search → discovers tool → calls tool |
| `test_e2e_non_anthropic_fallback` | Verify graceful fallback to pre-filter for non-Anthropic providers |
| `test_e2e_mixed_tools` | Mix of deferred and non-deferred tools from multiple MCP servers |

---

## Migration & Backward Compatibility

1. **`deferred_tools` defaults to `None`/empty** — existing MCP server configs are unaffected. All tools remain non-deferred by default.
2. **Semantic filter behavior unchanged when no deferred tools** — if `deferred_tools` is empty, the existing top-K pre-filter continues to work as before.
3. **DB migration** — add `deferred_tools` column with default `NULL`. Non-breaking for existing rows.
4. **API backward compatibility** — `deferred_tools` is optional in all endpoints. Existing API clients don't need to change.

---

## Implementation Priority

| Priority | Phase | Effort | Description |
|----------|-------|--------|-------------|
| **P0** | Phase 1 | Medium | Per-tool defer_loading config (model + DB + API) |
| **P0** | Phase 2 | Medium | Tool search injection in pre-call hook |
| **P0** | Phase 3.1 (Option A) | Low | Provider-native tool search for Anthropic |
| **P1** | Phase 5.1-5.3 | Medium | UI defer_loading toggle |
| **P1** | Phase 5.4 | Low | UI deferred status display |
| **P2** | Phase 4 | High | LiteLLM-managed tool search for non-Anthropic |
| **P2** | Phase 6 | Medium | UI semantic filter settings panel |

---

## Open Questions

1. **Should `deferred_tools` be per-server or per-key?** Current plan is per-server. Per-key would allow different users to see different tools as deferred, but adds complexity. Recommendation: start per-server, extend to per-key later via `mcp_tool_permissions`.

2. **How to handle multi-server deferred tools?** When a user has MCP tools from multiple servers, deferred tools from all servers should be aggregated into a single pool for tool search. The semantic index should cover all deferred tools across all accessible servers.

3. **Should the tool_search_variant be configurable per-server or globally?** Recommendation: global config in `litellm_settings.mcp_semantic_tool_filter.tool_search_variant`, with per-server override possible.

4. **Rate limiting on tool search?** Should we limit how many times the LLM can invoke tool_search in a single conversation? Anthropic's native implementation likely handles this, but for LiteLLM-managed search (Phase 4), we may want a cap.
