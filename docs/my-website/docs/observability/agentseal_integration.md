# AgentSeal - Verifiable Audit Trails for AI Agents

:::tip

This is community maintained. Please make an issue if you run into a bug:
https://github.com/BerriAI/litellm

:::

[AgentSeal](https://agentseal.io) is a logging and audit trail API for AI agents. Every action is recorded with a verifiable history that can't be silently modified. Built for compliance and production use.

## Quick Start

### Pre-Requisites

```shell
pip install agentseal-sdk litellm
```

Get your AgentSeal API key from [agentseal.io](https://agentseal.io).

### Usage with LiteLLM Python SDK

```python
import os
import litellm
from agentseal import Seal
from agentseal.integrations.litellm import AgentSealGuardrail

# Set env variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Initialize AgentSeal
seal = Seal(api_key="your-agentseal-api-key")
agentseal_handler = AgentSealGuardrail(seal=seal, agent="my-app")

# Add AgentSeal as a callback
litellm.callbacks = [agentseal_handler]

# Make LLM calls as usual — every call is recorded
response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize Q1 revenue"}],
)

print(response)
```

Every successful call records:
- Model name
- Prompt preview (first 300 chars)
- Output preview (first 500 chars)
- Token usage (prompt + completion tokens)

Every failed call records:
- Model name
- Error message

### Usage with LiteLLM Proxy

Add AgentSeal as a custom callback in your proxy config:

```yaml title="config.yaml"
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  callbacks: custom_callback.agentseal_handler

environment_variables:
  OPENAI_API_KEY: "your-openai-key"
  AGENTSEAL_API_KEY: "your-agentseal-key"
```

Create a `custom_callback.py` alongside your config:

```python
import os
from agentseal import Seal
from agentseal.integrations.litellm import AgentSealGuardrail

seal = Seal(api_key=os.environ["AGENTSEAL_API_KEY"])
agentseal_handler = AgentSealGuardrail(seal=seal, agent="litellm-proxy")
```

Start the proxy:

```bash
litellm --config config.yaml
```

## What Gets Recorded

| Event | Action Type | Params |
|-------|------------|--------|
| Successful LLM call | `llm:call` | model, prompt_preview, output_preview, prompt_tokens, completion_tokens |
| Failed LLM call | `llm:error` | model, error |

All entries are linked in a SHA-256 hash chain. Verify chain integrity at any time:

```python
result = seal.verify()
# {"valid": True, "entries_verified": 142}
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seal` | `Seal` | required | AgentSeal client instance |
| `agent` | `str` | `"litellm"` | Agent identifier for grouping entries (e.g. `"finance-bot"`, `"support-agent"`) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AGENTSEAL_API_KEY` | Your AgentSeal API key from [agentseal.io](https://agentseal.io) |

## Error Handling

AgentSeal never interrupts your LLM calls. If the AgentSeal API is unreachable or returns an error, a warning is printed to stderr and the LLM call proceeds normally.

## Links

- [Website](https://agentseal.io)
- [Documentation](https://agentseal.io/docs)
- [GitHub (SDK)](https://github.com/JoeyBrar/agentseal-sdk)
- [GitHub (MCP)](https://github.com/JoeyBrar/agentseal-mcp)
- [PyPI](https://pypi.org/project/agentseal-sdk/)
- [npm (MCP server)](https://www.npmjs.com/package/agentseal-mcp)
