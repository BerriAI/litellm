# MCP Semantic Filtering — UI Configuration Plan

## Goal

Expose the **existing** `mcp_semantic_tool_filter` backend config on the LiteLLM dashboard so admins can enable/configure it without editing YAML.

Today these settings only live in `proxy_config.yaml`:

```yaml
litellm_settings:
  mcp_semantic_tool_filter:
    enabled: true
    embedding_model: "text-embedding-3-small"
    top_k: 5
    similarity_threshold: 0.3
```

There is **no UI** for this. The plan is to add a settings panel to the existing **MCP Servers** page.

---

## Where to Put It

The MCP Servers page (`/tools/mcp-servers`) already has **2 tabs**: "All Servers" and "Connect".

Add a **3rd tab: "Semantic Filtering"**.

This follows the existing pattern — MCP-related settings live on the MCP page, not scattered in general settings.

```
┌──────────────┬──────────┬─────────────────────┐
│ All Servers  │ Connect  │ Semantic Filtering   │
└──────────────┴──────────┴─────────────────────┘
```

---

## UI Design

Simple form with 4 fields matching the existing backend config, plus a save button:

```
┌─────────────────────────────────────────────────────────────┐
│  Semantic Tool Filtering                                    │
│                                                             │
│  Automatically filters MCP tools based on semantic          │
│  similarity to the user's query, reducing the number of     │
│  tools sent to the LLM.                                     │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Enabled               [Toggle: OFF]                   │  │
│  │                                                       │  │
│  │ Embedding Model       [text-embedding-3-small    ▼]   │  │
│  │                                                       │  │
│  │ Max Results (top_k)   [  5  ]                         │  │
│  │                                                       │  │
│  │ Similarity Threshold  [  0.3  ]                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ℹ  Requires `semantic-router` pip package.                 │
│     Embedding model must be available in your model list.   │
│                                                             │
│                            [Save]   [Reset to Defaults]     │
└─────────────────────────────────────────────────────────────┘
```

### Field Details

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| Enabled | Toggle (Switch) | OFF | Master on/off for the feature |
| Embedding Model | Text input (or dropdown of available models) | `text-embedding-3-small` | Must match a model in `model_list` |
| Max Results (top_k) | Number input | `5` | Range: 1-50 |
| Similarity Threshold | Number input | `0.3` | Range: 0.0-1.0, step 0.05 |

All fields disabled (greyed out) when "Enabled" toggle is OFF, except the toggle itself.

---

## Backend: API Endpoints

Use the **existing config update pattern** already used by Router Settings / General Settings.

### Read current config

```
GET /config/list?config_type=general_settings
```

The response already includes `litellm_settings` fields. We just need to read `mcp_semantic_tool_filter` from the response. If it's not already exposed, we add it to the config list response.

Alternatively, add a dedicated lightweight endpoint:

```
GET /v1/mcp/semantic_filter/settings
```

Response:
```json
{
  "enabled": false,
  "embedding_model": "text-embedding-3-small",
  "top_k": 5,
  "similarity_threshold": 0.3,
  "stored_in_db": true
}
```

### Update config

Option A — Use existing field update API:

```
POST /config/field/update
{
  "field_name": "mcp_semantic_tool_filter",
  "field_value": {
    "enabled": true,
    "embedding_model": "text-embedding-3-small",
    "top_k": 5,
    "similarity_threshold": 0.3
  },
  "config_type": "litellm_settings"
}
```

This requires extending the `config_type` literal to accept `"litellm_settings"` (currently only accepts `"general_settings"`).

Option B — Dedicated endpoint:

```
PUT /v1/mcp/semantic_filter/settings
{
  "enabled": true,
  "embedding_model": "text-embedding-3-small",
  "top_k": 5,
  "similarity_threshold": 0.3
}
```

**Recommendation: Option B** — a dedicated endpoint is simpler and doesn't require modifying the general config system. It writes to the DB-backed config and triggers a config reload.

---

## Files to Change

### Backend (2-3 files)

| File | Change |
|------|--------|
| `litellm/proxy/management_endpoints/mcp_management_endpoints.py` | Add `GET /v1/mcp/semantic_filter/settings` and `PUT /v1/mcp/semantic_filter/settings` endpoints |
| `litellm/proxy/proxy_server.py` | Register the new endpoints (if not auto-registered via router) |
| `litellm/proxy/_types.py` | Add request/response Pydantic models for the settings payload |

### Frontend (3 files)

| File | Change |
|------|--------|
| `ui/litellm-dashboard/src/components/mcp_tools/mcp_servers.tsx` | Add "Semantic Filtering" tab to the existing TabGroup |
| `ui/litellm-dashboard/src/components/mcp_tools/semantic_filter_settings.tsx` | **New file** — the settings form component |
| `ui/litellm-dashboard/src/components/networking.tsx` | Add `getMCPSemanticFilterSettings()` and `updateMCPSemanticFilterSettings()` API calls |

---

## Implementation Steps

1. **Add backend endpoints** — `GET` + `PUT` for `/v1/mcp/semantic_filter/settings`
   - Read: pull from `litellm_settings` (in-memory) with DB fallback
   - Write: save to DB config, trigger config reload so `SemanticToolFilterHook` reinitializes

2. **Add networking functions** in `networking.tsx`

3. **Create `semantic_filter_settings.tsx`** component with the form

4. **Add tab** in `mcp_servers.tsx` — third tab rendering the new component

5. **Test** — verify settings round-trip (save in UI → reflected in backend behavior)

---

## What This Does NOT Cover

- Per-tool `defer_loading` configuration (future work)
- Tool search tool injection (future work)
- Any changes to the semantic filter logic itself
- Non-Anthropic provider support

This plan only exposes existing config knobs in the UI. The backend behavior is unchanged.
