# OpenFPGA

## Overview

| Property | Details |
|-------|-------|
| Description | OpenFPGA provides FPGA-accelerated inference for open-source LLMs. Drop-in replacement for GPU inference — OpenAI-compatible API, deterministic latency, lower energy per token. |
| Provider Route on LiteLLM | `openfpga/` |
| Link to Provider Doc | [OpenFPGA ↗](https://openfpga.ai) |
| Base URL | `https://app.openfpga.ai/api/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />

## What is OpenFPGA?

OpenFPGA is an FPGA-accelerated inference gateway that runs open-source LLMs on Intel Agilex FPGA hardware:

- **FPGA-Optimized Inference**: Custom hardware pipelines synthesized per model on Intel Agilex FPGAs
- **OpenAI-Compatible API**: Drop-in replacement — just change the base URL
- **Deterministic Latency**: No GPU queuing delays, consistent token generation speed
- **Lower Energy Per Token**: 5–20x better energy efficiency than GPU inference
- **Function Calling + Structured Outputs**: Full support for tool use and JSON mode

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["OPENFPGA_API_KEY"] = ""  # your OpenFPGA API key
```

Get your OpenFPGA API key from [app.openfpga.ai](https://app.openfpga.ai).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="OpenFPGA Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["OPENFPGA_API_KEY"] = ""  # your OpenFPGA API key

messages = [{"content": "What is FPGA-accelerated inference?", "role": "user"}]

# OpenFPGA call
response = completion(
    model="openfpga/llama-3.1-8b-fpga",
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="OpenFPGA Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["OPENFPGA_API_KEY"] = ""  # your OpenFPGA API key

messages = [{"content": "Explain how FPGAs differ from GPUs for inference", "role": "user"}]

# OpenFPGA call with streaming
response = completion(
    model="openfpga/llama-3.1-8b-fpga",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export OPENFPGA_API_KEY=""
```

### 2. Start the proxy

```yaml
model_list:
  - model_name: llama-3.1-8b-fpga
    litellm_params:
      model: openfpga/llama-3.1-8b-fpga
      api_key: os.environ/OPENFPGA_API_KEY
```

## Supported OpenAI Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID (e.g. `llama-3.1-8b-fpga`) |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |
| `response_format` | object | Optional. JSON mode or JSON schema enforcement |

## Available Models

| Model | Context Window | Description |
|-------|---------------|-------------|
| `openfpga/llama-3.1-8b-fpga` | 131,072 tokens | Llama 3.1 8B Instruct on Intel Agilex FPGA |

## Additional Resources

- [OpenFPGA Website](https://openfpga.ai)
- [API Documentation](https://app.openfpga.ai/llms-full.txt)
- [OpenAPI Spec](https://app.openfpga.ai/.well-known/openapi.yaml)
- [MCP Server](https://www.npmjs.com/package/@openfpga/mcp)
