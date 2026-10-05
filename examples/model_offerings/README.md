# External model offerings

Set `general_settings.model_offerings_path` in the normal proxy configuration and keep its `model_list` empty. Disable `store_model_in_db`; the database may still serve authentication and usage tracking. Existing proxy configurations behave as before when the path is absent

```yaml
model_list: []
general_settings:
  model_offerings_path: /app/offerings/offerings.yaml
  store_model_in_db: false
```

Mount a host directory into Docker, for example `./offerings:/app/offerings:ro`. Keep `offerings.yaml` inside it. Replace the file atomically within that directory so the container sees the new inode. A bind mount of an individual file does not reliably see atomic replacement

```yaml
version: 1
config_poll_seconds: 5
inventory_poll_seconds: 300
providers:
  openrouter:
    provider: openrouter
    api_key: os.environ/OPENROUTER_API_KEY
  vercel:
    provider: vercel_ai_gateway
    api_key: os.environ/VERCEL_API_KEY
  local:
    provider: openai
    api_base: http://inference-server:8000/v1
    api_key: os.environ/INFERENCE_API_KEY
  subscribed:
    provider: chatgpt
offerings:
  - model_name: selected
    source: auto
    provider: openrouter
    upstream_model: organization/model
    enabled: true
  - model_name: handwritten
    source: manual
    provider: openrouter
    upstream_model: organization/another-model
    model_info:
      mode: chat
      context_window: 262144
      supports_function_calling: true
      request_defaults:
        output_token_budget: 8192
        output_token_budget_by_reasoning_effort:
          low: 65536
          high: 65536
          xhigh: 65536
          max: 131072
```

These example token values illustrate operator configuration, not a claim about the example supplier models. Request defaults express chosen budgets and do not populate independent output capability limits

`source` belongs to each offering. AUTO entries require membership in the latest successful full inventory for their connection. Missing entries disappear from model listings and return a useful unavailable error before forwarding. A returning entry becomes available again. AUTO metadata combines the verified catalog with downloaded supplier fields and then explicit overrides, in increasing priority, and freezes the result in the serving snapshot. MANUAL entries use only handwritten definitions and are independent of supplier inventory membership. Both sources accept shared `model_info` overrides. Wildcards and duplicate public aliases are rejected

Startup validates the complete file and awaits initial inventory requests. AUTO entries remain unavailable when no successful inventory exists; MANUAL entries remain usable. A successful empty inventory means absence. HTTP errors, authentication failures, timeouts and malformed responses retain the last successful inventory for unchanged credentials. Changing connection credentials starts with unknown availability and forces a fresh inventory. ChatGPT uses its existing native OAuth account and cannot initiate device login from discovery or request handling

The file is checked every five seconds by default, with a configurable range of 1 to 300 seconds. Inventories refresh every 300 seconds by default, with a range of 1 to 86400 seconds. Changing that interval triggers a fresh inventory request and resets its deadline. Invalid file edits retain the whole last-valid serving configuration, including credentials, and log only the error kind. Unselected inventory entries never become public routes. Caller connection overrides and runtime model-management mutations are rejected in external mode; edit the external file instead

Each HTTP request pins one immutable serving snapshot through the complete ASGI response lifetime, including streaming. Publishing a new snapshot changes subsequent requests while requests already running finish on their previous native Router. Snapshot creation isolates deployment indexes and metadata caches while reusing the initialized Router's bounded client, usage and routing resources. It does not register fresh callbacks or close shared HTTP clients on each reload. The inventory HTTP client and polling task are closed at proxy shutdown after requests drain
