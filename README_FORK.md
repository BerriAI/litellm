**LiteLLM Fork Overview**

- This fork adds a paved, opt-in path for a tiny mini-agent, an env‑gated `codex-agent` provider, a pragmatic HTTP tools adapter, and a reorganized smoke test suite. All upstream public APIs remain compatible.

**Highlights**
- Mini‑Agent + HTTP Façade
  - Endpoints: `/agent/run` (deterministic envelope), `/v1/chat/completions` (OpenAI shim), `/ready` (health).
  - Backends: `local` (in‑process tools), `http` (external MCP-like tools), `echo` (hermetic), and fallback to Router for one‑shot completion.
  - Envelope: `{ok, final_answer, stopped_reason, messages, metrics}` with `metrics.escalated` and `metrics.used_model`.
  - Tracing: opt‑in JSONL appends via `MINI_AGENT_STORE_TRACES=1` and `MINI_AGENT_STORE_PATH`.
  - Escalation: budget‑stop aware with optional `escalate_model`; short‑circuit for chutes in nd‑smokes via `NDSMOKE_SHORTCIRCUIT_CHUTES=1`.
  - Files: `litellm/experimental_mcp_client/mini_agent/agent_proxy.py`, `litellm/experimental_mcp_client/mini_agent/__init__.py`.

- HTTP Tools Invoker (MCP‑style over HTTP)
  - Lists tools from `/tools`, invokes via `/invoke`.
  - Merges env/request headers; preserves request header casing; supports Authorization pass‑through; retries on 429 (bounded).
  - File: `litellm/experimental_mcp_client/mini_agent/http_tools_invoker.py`.

- Local Tools (in‑process)
  - `exec_python`: captures `rc/stdout/stderr`, enforces timeouts; safe shaping for tool results.
  - `exec_shell`: allowlist + timeout; returns `rc/stdout/stderr`; blocks disallowed commands.
  - Parallel tool execution preserves original call order in stitched messages.
  - Files: `litellm/experimental_mcp_client/mini_agent/litellm_mcp_mini_agent.py`, `litellm/experimental_mcp_client/mini_agent/exec_rpc_server.py`.

- Env‑Gated Provider: `codex-agent`
  - Purpose: HTTP‑only adapter that posts to a compatible `/v1/chat/completions` endpoint.
  - Enable: `LITELLM_ENABLE_CODEX_AGENT=1`.
  - Config: `CODEX_AGENT_API_BASE` (or `api_base` param), optional `CODEX_AGENT_API_KEY` → `Authorization: Bearer ...`.
  - Aliases: `codex-agent`, `codex_cli_agent`.
  - Sync + async via `httpx`, clean non‑2xx error surfacing.
  - Files: `litellm/llms/codex_agent.py`, docs at `docs/my-website/docs/providers/codex_agent.md`.

- Response Utilities
  - `extract_content(obj|dict)`, `assemble_stream_text(async_iter)`, `augment_json_with_cost(json_str, resp)`.
  - File: `litellm/extras/response_utils.py` (import‑guarded by tests).

- Smokes Reorg (Deterministic vs Live)
  - Deterministic tests moved under `tests/local_testing/`.
  - Live-ish and optional E2E under `tests/ndsmoke_e2e/` and `tests/smoke/` (env‑gated).
  - Make targets for E2E: `e2e-up`, `e2e-run`, `e2e-down` (Docker/compose harness).
  - Key shapes locked by `tests/local_testing/test_agent_proxy_response_shapes.py` and HTTP tools invoker contract tests.

**Quick Start**
- Mini‑Agent server
  - `uvicorn litellm.experimental_mcp_client.mini_agent.agent_proxy:app --port 8788`
  - Health: `curl localhost:8788/ready`
  - Deterministic run: `curl -X POST localhost:8788/agent/run -H 'Content-Type: application/json' -d '{"model":"dummy","tool_backend":"echo","messages":[{"role":"user","content":"hi"}]}'`

- Codex‑Agent via Router
  - `export LITELLM_ENABLE_CODEX_AGENT=1`
  - `export CODEX_AGENT_API_BASE=http://127.0.0.1:8788`  (e.g., mini‑agent OpenAI shim)
  - `export CODEX_AGENT_API_KEY=sk-optional`
  - Router entry: `{"model_name":"codex-agent-1","litellm_params":{"model":"codex-agent/mini"}}`

- HTTP Tools (external)
  - `MINI_AGENT_TOOL_HTTP_HEADERS='{"Authorization":"Bearer <token>"}'` for default headers.
  - POST `/agent/run` with `{"tool_backend":"http","tool_http_base_url":"https://tools.example"}`.

**Design Notes**
- Backwards compatibility: Upstream public APIs preserved; new surfaces are opt‑in and env‑gated.
- Determinism first: Hermetic backends (`echo`, `local`) and clear skip conditions for live tests.
- Safety: subprocess tools enforce timeouts and use kill()+wait(); HTTP adapters propagate non‑2xx errors and add bounded 429 retries.

**Known Limitations**
- `codex-agent` is HTTP‑only in this fork; CLI/binary integration may land later.
- Live E2E tests require local services (mini‑agent, exec‑rpc) and are skip‑friendly by default.

**References**
- Mini‑Agent proxy: `litellm/experimental_mcp_client/mini_agent/agent_proxy.py`
- Mini‑Agent loop: `litellm/experimental_mcp_client/mini_agent/litellm_mcp_mini_agent.py`
- HTTP tools: `litellm/experimental_mcp_client/mini_agent/http_tools_invoker.py`
- Codex provider: `litellm/llms/codex_agent.py`
- Docs: `docs/my-website/docs/providers/codex_agent.md`
- Smokes: `tests/local_testing/`, `tests/smoke/`, `tests/ndsmoke_e2e/`

