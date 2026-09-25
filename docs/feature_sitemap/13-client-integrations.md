# 13 Client integrations

Pointing third-party tools at the gateway: coding agents (Claude Code, Codex, Cursor, Copilot, Gemini CLI, and friends), chat UIs (Claude Desktop, Open WebUI, Retool), agent SDKs, and plain OpenAI/Anthropic SDK drop-ins, plus the VS Code extension. Most leaves here are documentation and compatibility surfaces rather than dedicated code paths; the endpoints they ride on live in [01-llm-endpoints.md](01-llm-endpoints.md)

## Coding agents

### clients.claude_code: Claude Code (compat matrix, /v1/messages, count_tokens, beta headers, statusline budgets, non-anthropic models)
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/proxy/client_setup/claude_code, https://docs.litellm.ai/docs/claude_code_compatibility, https://docs.litellm.ai/docs/claude_code_context_management, https://docs.litellm.ai/docs/tutorials/claude_code_beta_headers, https://docs.litellm.ai/docs/tutorials/claude_code_budget_statusline, https://docs.litellm.ai/docs/tutorials/claude_non_anthropic_models
code: `litellm/proxy/anthropic_endpoints/endpoints.py` (`/v1/messages`, `/v1/messages/count_tokens`), `litellm/proxy/anthropic_endpoints/gateway_endpoints.py` (`/managed/settings`), `tests/e2e/claude_code/` (compat suite driver)
tests: `tests/e2e/claude_code/`
registry: llm_claude_code_compat.yaml (feature x provider cells)
verify: set `ANTHROPIC_BASE_URL=http://localhost:4000` and `ANTHROPIC_AUTH_TOKEN=sk-1234`, run `claude "say hi"`, and confirm a working session through the proxy

### clients.claude_desktop_cowork: Claude Desktop and Cowork
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/proxy/client_setup/claude_desktop, https://docs.litellm.ai/docs/tutorials/claude_desktop_cowork
code: `litellm/proxy/anthropic_endpoints/endpoints.py` (`/v1/messages`), `litellm/proxy/anthropic_endpoints/gateway_endpoints.py`
tests: `tests/e2e/claude_code/` (message-shape coverage)
registry: llm_claude_code_compat.yaml
verify: add the proxy as a connector in Claude Desktop per the doc and send a chat through it

### clients.codex: Codex CLI and ChatGPT desktop
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/proxy/client_setup/codex_cli, https://docs.litellm.ai/docs/proxy/client_setup/codex_chatgpt_desktop, https://docs.litellm.ai/docs/tutorials/codex_customer_tracking
code: `litellm/proxy/response_api_endpoints/endpoints.py` (`/v1/responses`), `litellm/proxy/proxy_server.py`
tests: `tests/e2e/llm_translation/test_responses_e2e.py`
registry: llm_conversational.yaml llm.responses.*
verify: set `OPENAI_BASE_URL=http://localhost:4000/v1` and `OPENAI_API_KEY=sk-1234` for codex, run one task, and confirm /v1/responses serves it

### clients.cursor: Cursor
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/tutorials/cursor_integration
code: `litellm/proxy/response_api_endpoints/endpoints.py` (`/cursor/models`, `/cursor/chat/completions`), `litellm/proxy/pass_through_endpoints/llm_passthrough_endpoints.py` (`/cursor/{endpoint:path}`)
tests: `tests/test_litellm/litellm_core_utils/test_streaming_chunk_builder_cursor.py`
registry: none
verify: point Cursor's model settings at http://localhost:4000/cursor with a virtual key and confirm /cursor/models lists deployments

### clients.github_copilot: GitHub Copilot
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/tutorials/github_copilot_integration
code: `litellm/proxy/proxy_server.py` (OpenAI-compatible endpoints copilot calls)
tests: `tests/e2e/llm_translation/`
registry: none
verify: configure Copilot's OpenAI-compatible endpoint to the proxy and confirm completions flow

### clients.gemini_cli_qwen_code: Gemini CLI, Qwen Code, OpenCode, OpenClaw
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/tutorials/litellm_gemini_cli, https://docs.litellm.ai/docs/tutorials/litellm_qwen_code_cli, https://docs.litellm.ai/docs/tutorials/opencode_integration, https://docs.litellm.ai/docs/tutorials/openclaw_integration
code: `litellm/proxy/google_endpoints/endpoints.py` (`/v1beta/models/*:generateContent` for gemini-cli), `litellm/proxy/anthropic_endpoints/endpoints.py` (`/v1/messages` for the anthropic-shaped CLIs), `litellm/proxy/proxy_server.py`
tests: `tests/e2e/llm_translation/`
registry: llm_conversational.yaml llm.messages.*, llm_nonconversational.yaml llm.google_native.*
verify: set the CLI's base url env to the proxy (e.g. `GOOGLE_GEMINI_BASE_URL=http://localhost:4000`) and run one prompt

### clients.openwebui_retool: Open WebUI, Retool
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/tutorials/openweb_ui, https://docs.litellm.ai/docs/tutorials/retool_assist
code: `litellm/proxy/proxy_server.py` (OpenAI-compatible endpoints)
tests: `tests/e2e/llm_translation/`
registry: none
verify: add http://localhost:4000/v1 as an OpenAI connection in Open WebUI with the master key and list models

## SDKs and drop-ins

### clients.agent_sdks: OpenAI Agents SDK, Claude Agent SDK, Google ADK, GenAI SDK, CopilotKit, LiveKit, Letta
surfaces: api, sdk | flags: none
docs: https://docs.litellm.ai/docs/tutorials/openai_agents_sdk, https://docs.litellm.ai/docs/tutorials/claude_agent_sdk, https://docs.litellm.ai/docs/tutorials/google_adk, https://docs.litellm.ai/docs/tutorials/google_genai_sdk, https://docs.litellm.ai/docs/tutorials/copilotkit_sdk, https://docs.litellm.ai/docs/tutorials/livekit_xai_realtime, https://docs.litellm.ai/docs/integrations/letta
code: `litellm/proxy/proxy_server.py`, `litellm/proxy/anthropic_endpoints/endpoints.py`, `litellm/proxy/google_endpoints/endpoints.py`, `litellm/proxy/response_api_endpoints/endpoints.py`
tests: `tests/e2e/llm_translation/`
registry: llm_conversational.yaml llm.responses.*, llm.messages.*
verify: point one agent SDK at the proxy per its tutorial (e.g. Agents SDK with OPENAI_BASE_URL) and run a trivial agent

### clients.openai_sdk_compat: OpenAI SDK drop-in compatibility (python, js)
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/proxy_server, https://docs.litellm.ai/docs/proxy/quick_start
code: `litellm/proxy/proxy_server.py` (OpenAI route surface)
tests: `tests/e2e/llm_translation/` (OpenAI SDK driven tests)
registry: llm_conversational.yaml llm.chat_completions.openai.*
verify: `openai.OpenAI(base_url="http://localhost:4000/v1", api_key="sk-1234").chat.completions.create(model="gpt-5.5", messages=[{"role":"user","content":"hi"}])`

### clients.anthropic_sdk_compat: Anthropic SDK drop-in compatibility
surfaces: api | flags: none
docs: https://docs.litellm.ai/docs/anthropic_unified/index
code: `litellm/proxy/anthropic_endpoints/endpoints.py` (`/v1/messages`)
tests: `tests/e2e/llm_translation/test_messages_e2e.py`
registry: llm_conversational.yaml llm.messages.anthropic.*
verify: `anthropic.Anthropic(base_url="http://localhost:4000", api_key="sk-1234").messages.create(model="claude-sonnet-4-6", max_tokens=10, messages=[{"role":"user","content":"hi"}])`

### clients.vscode_extension: VS Code extension
surfaces: none listed | flags: none
docs: none found
code: `vscode-extension/`
tests: none found
registry: none
verify: install the extension from vscode-extension/, point it at a running proxy, and issue a chat from the sidebar
