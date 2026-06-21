# Code Interpreter Interception

Route the OpenAI `code_interpreter` tool to a self-hosted or e2b sandbox, transparently, without changing the request your client sends.

When a client calls the Responses API with a `code_interpreter` tool, LiteLLM swaps the native tool for a function tool, runs the python the model emits inside a sandbox you control, feeds the captured stdout back to the model through an agentic loop, and returns the final response. The client sends a normal `/v1/responses` request and gets a normal response back; the sandbox round-trips happen on the proxy.

## How it works

1. The pre-call hook detects a `code_interpreter` tool in the request and replaces it with a `litellm_code_execution` function tool, so the model emits code as function-call arguments.
2. When the model returns a `litellm_code_execution` call, the interceptor executes the code in a sandbox (reused per request so state persists across iterations of the same call).
3. The captured stdout is appended as `function_call_output` and the request is rerun, looping until the model produces a final answer.

## Setup

### 1. Define a sandbox tool and enable the callback

Register a sandbox under the top-level `sandbox_tools`, then turn on the interceptor with `callbacks` and point `code_interpreter_interception_params.sandbox_tool_name` at it.

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-5
    litellm_params:
      model: openai/gpt-5
      api_key: os.environ/OPENAI_API_KEY

sandbox_tools:
  - sandbox_tool_name: my-e2b-sandbox
    litellm_params:
      sandbox_provider: e2b
      api_key: os.environ/E2B_API_KEY
      api_base: os.environ/E2B_API_BASE   # Optional; set for self-hosted e2b

litellm_settings:
  callbacks: ["code_interpreter_interception"]
  code_interpreter_interception_params:
    enabled_providers: ["openai"]
    sandbox_tool_name: my-e2b-sandbox
```

`enabled_providers` scopes interception to specific upstream providers. Omit it to intercept for every provider. `sandbox_tool_name` must match a `sandbox_tool_name` registered under `sandbox_tools`.

### 2. Set environment variables

```shell
export OPENAI_API_KEY="sk-..."
export E2B_API_KEY="e2b_..."
```

### 3. Start the proxy

```shell
litellm --config config.yaml
```

## Usage

Call `/v1/responses` with a `code_interpreter` tool. Nothing in the request is LiteLLM-specific; this is a standard OpenAI Responses API call.

```shell
curl http://localhost:4000/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "gpt-5",
    "input": "Compute the 20th Fibonacci number using python and tell me the result.",
    "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]
  }'
```

The model writes python, the proxy runs it in your sandbox, and the final response reflects the executed result.

## Limitations (v0)

File upload and download are not supported yet. The interceptor executes code and returns stdout; passing input files into the sandbox or retrieving generated files out of it is not wired up. Only stdout from the executed code is fed back to the model.
