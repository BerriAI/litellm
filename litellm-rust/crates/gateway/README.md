# Rust gateway

The binary reads YAML from `LITELLM_CONFIG` and mounts configured MCP servers alongside inference routes. MCP startup requires a resolvable `general_settings.master_key`. Every MCP and REST request uses the gateway's shared authentication middleware

## Run MCP locally

From the repository root, start the Python fixture in one terminal:

```sh
MCP_HOST=127.0.0.1 MCP_PORT=8090 .venv/bin/python tests/mcp_tests/mcp_e2e_upstream_server.py
```

Start the gateway in another terminal with the example configuration:

```sh
LITELLM_MASTER_KEY=local-mcp-key \
LITELLM_CONFIG=litellm-rust/crates/gateway/mcp_config.yaml \
HOST=127.0.0.1 PORT=4000 \
cargo run --manifest-path litellm-rust/Cargo.toml -p litellm-gateway
```

Call a tool through REST:

```sh
curl -sS http://127.0.0.1:4000/mcp-rest/tools/call \
  -H 'Authorization: Bearer local-mcp-key' \
  -H 'Content-Type: application/json' \
  -d '{"server_id":"math","name":"add","arguments":{"a":2,"b":3}}'
```

MCP clients can connect to `/mcp` or the scoped `/math/mcp` endpoint with the same admission credential. Tool names on the MCP endpoint carry the configured alias or server-name prefix

## Configuration behavior

`Config::load` resolves nested `include` files once in breadth-first order. Included lists append and other top-level values replace earlier values. Paths resolve relative to the declaring file, with the root config directory as a fallback

`environment_variables` provides scalar overrides to the injected secret source and HTTP settings without changing process globals. Credentials, static headers, URLs, and stdio environment values can reference `os.environ/NAME`. Model providers and MCP share the host's HTTP settings, including TLS certificates and environment proxies. Explicit environment HTTP settings take precedence over `litellm_settings`

Each MCP entry supports HTTP or stdio transport, an optional pinned `server_id`, alias, tool allowlist, timeout in seconds, and positive `max_concurrent_requests`. Omitted or empty tool allowlists allow all upstream tools. HTTP entries accept `static_headers` and static API-key, bearer, basic, authorization, or token credentials. Stdio entries accept `command`, `args`, and `env`. Use explicit secret references in `env` to pass configured credentials to a child process

The gateway initializes upstreams before opening its listener. Startup fails when credentials cannot be resolved or an upstream cannot initialize. Shutdown cancels upstream connections and terminates owned stdio children

The config schema preserves Python sections, but parsing a section does not implement its service. MCP OAuth, database-backed server management, guardrails, spend accounting, local Python tools, and legacy SSE upstream connections remain unimplemented. Unsupported MCP settings and configured guardrail policies fail at startup. Incoming legacy SSE clients remain supported
