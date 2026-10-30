# Gollem Go Agent Framework with LiteLLM

A working example showing how to use [gollem](https://github.com/fugue-labs/gollem), a production-grade Go agent framework, with LiteLLM as a proxy gateway. This lets Go developers access 100+ LLM providers through a single proxy while keeping compile-time type safety for tools and structured output.

## Quick Start

### 1. Start LiteLLM Proxy

The proxy refuses to start without a master key, so generate one first and keep it for step 2:

```bash
export OPENAI_API_KEY="sk-..."
export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 32)"
echo "$LITELLM_MASTER_KEY"

# Simple start with a single model
litellm --model gpt-4o

# Or with the example config for multi-provider access
litellm --config proxy_config.yaml
```

### 2. Run the examples

In a second terminal, give the examples the master key from step 1. They send `LITELLM_API_KEY` to the proxy as their API key:

```bash
export LITELLM_API_KEY="<the master key from step 1>"

# Install Go dependencies
go mod tidy

# Basic agent
go run ./basic

# Agent with type-safe tools
go run ./tools

# Streaming responses
go run ./streaming
```

## Configuration

The included `proxy_config.yaml` sets up three providers through LiteLLM:

```yaml
model_list:
  - model_name: gpt-4o          # OpenAI
  - model_name: claude-sonnet    # Anthropic
  - model_name: gemini-pro       # Google Vertex AI
```

Switch providers in Go by changing a single string — no code changes needed:

```go
model := openai.NewLiteLLM("http://localhost:4000",
    openai.WithModel("gpt-4o"),        // OpenAI
    // openai.WithModel("claude-sonnet"),  // Anthropic
    // openai.WithModel("gemini-pro"),     // Google
    openai.WithAPIKey(os.Getenv("LITELLM_API_KEY")),
)
```

## Examples

### `basic/` — Basic Agent

Connects gollem to LiteLLM and runs a simple prompt. Demonstrates the `NewLiteLLM` constructor and basic agent creation.

### `tools/` — Type-Safe Tools

Shows gollem's compile-time type-safe tool framework working through LiteLLM's tool-use passthrough. The tool parameters are Go structs with JSON tags — the schema is generated automatically at compile time.

### `streaming/` — Streaming Responses

Real-time token streaming using Go 1.23+ range-over-function iterators, proxied through LiteLLM's SSE passthrough.

## How It Works

Gollem's `openai.NewLiteLLM()` constructor creates an OpenAI-compatible provider pointed at your LiteLLM proxy. Since LiteLLM speaks the OpenAI API protocol, everything works out of the box:

- **Chat completions** — standard request/response
- **Tool use** — LiteLLM passes tool definitions and calls through transparently
- **Streaming** — Server-Sent Events proxied through LiteLLM
- **Structured output** — JSON schema response format works with supporting models

```
Go App (gollem) → LiteLLM Proxy → OpenAI / Anthropic / Google / ...
```

## Why Use This?

- **Type-safe Go**: Compile-time type checking for tools, structured output, and agent configuration — no runtime surprises
- **Single proxy, many models**: Switch between OpenAI, Anthropic, Google, and 100+ other providers by changing a model name string
- **Zero-dependency core**: gollem's core has no external dependencies — just stdlib
- **Single binary deployment**: `go build` produces one binary, no pip/venv/Docker needed
- **Cost tracking & rate limiting**: LiteLLM handles cost tracking, rate limits, and fallbacks at the proxy layer

## Environment Variables

```bash
# Proxy terminal: keys for the providers you want to use (set in LiteLLM config or env)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Proxy terminal: required, the proxy refuses to start without a master key
export LITELLM_MASTER_KEY="sk-$(openssl rand -hex 32)"

# Go terminal: the key the examples send to the proxy (the master key or a virtual key)
export LITELLM_API_KEY="<your LiteLLM master key>"

# Optional: point to a non-default LiteLLM proxy
export LITELLM_PROXY_URL="http://localhost:4000"
```

When `LITELLM_API_KEY` is unset, gollem falls back to sending `OPENAI_API_KEY`, which the proxy rejects unless it holds a LiteLLM key.

## Troubleshooting

**Proxy refuses to start with `no master key is set`?**
- Export `LITELLM_MASTER_KEY` before starting it, as in step 1

**Connection errors?**
- Make sure LiteLLM is running: `litellm --model gpt-4o`
- Check the URL is correct (default: `http://localhost:4000`)

**Authentication errors (401, or 400 `No connected db.`)?**
- Set `LITELLM_API_KEY` in the Go terminal to the proxy's master key

**Model not found?**
- Verify the model name matches what's configured in LiteLLM
- Run `curl http://localhost:4000/models -H "Authorization: Bearer $LITELLM_API_KEY"` to see available models

**Tool calls not working?**
- Ensure the underlying model supports tool use (GPT-4o, Claude, Gemini)
- Check LiteLLM logs for any provider-specific errors

## Learn More

- [gollem GitHub](https://github.com/fugue-labs/gollem)
- [gollem API Reference](https://pkg.go.dev/github.com/fugue-labs/gollem/core)
- [LiteLLM Proxy Docs](https://docs.litellm.ai/docs/simple_proxy)
- [LiteLLM Supported Models](https://docs.litellm.ai/docs/providers)
