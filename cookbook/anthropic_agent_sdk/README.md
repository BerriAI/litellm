# Claude Agent SDK with LiteLLM Gateway

A simple example showing how to use Claude's Agent SDK with LiteLLM as a proxy. This lets you use any LLM provider (OpenAI, Bedrock, Azure, etc.) through the Agent SDK.

## Quick Start

### 1. Install dependencies

```bash
pip install anthropic claude-agent-sdk litellm
```

### 2. Start LiteLLM proxy

```bash
# Simple start with Claude
litellm --model claude-sonnet-4-20250514

# Or with a config file
litellm --config config.yaml
```

### 3. Run the chat

**Basic Agent (no MCP):**

```bash
python main.py
```

**Agent with MCP (DeepWiki2 for research):**

```bash
python agent_with_mcp.py
```

If MCP connection fails, you can disable it:

```bash
USE_MCP=false python agent_with_mcp.py
```

That's it! You can now chat with the agent in your terminal.

### Chat Commands

While chatting, you can use these commands:
- `models` - List all available models (fetched from your LiteLLM proxy)
- `model` - Switch to a different model
- `clear` - Start a new conversation
- `quit` or `exit` - End the chat

The chat automatically fetches available models from your LiteLLM proxy's `/models` endpoint, so you'll always see what's currently configured.

## Configuration

Set these environment variables if needed:

```bash
export LITELLM_PROXY_URL="http://localhost:4000"
export LITELLM_API_KEY="sk-1234"
export LITELLM_MODEL="bedrock-claude-sonnet-4.5"
```

Or just use the defaults - it'll connect to `http://localhost:4000` by default.

## Files

- `main.py` - Basic interactive agent without MCP
- `agent_with_mcp.py` - Agent with MCP server integration (DeepWiki2)
- `common.py` - Shared utilities and functions
- `config.example.yaml` - Example LiteLLM configuration
- `requirements.txt` - Python dependencies

## Example Config File

If you want to use multiple models, create a `config.yaml` (see `config.example.yaml`):

```yaml
model_list:
  - model_name: bedrock-claude-sonnet-4
    litellm_params:
      model: "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
      aws_region_name: "us-east-1"

  - model_name: bedrock-claude-sonnet-4.5
    litellm_params:
      model: "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
      aws_region_name: "us-east-1"
```

Then start LiteLLM with: `litellm --config config.yaml`

## How It Works

The key is pointing the Agent SDK to LiteLLM instead of directly to Anthropic:

```python
# Point to LiteLLM gateway (not Anthropic)
os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:4000"
os.environ["ANTHROPIC_API_KEY"] = "sk-1234"  # Your LiteLLM key

# Use any model configured in LiteLLM
options = ClaudeAgentOptions(
    model="bedrock-claude-sonnet-4",  # or gpt-4, or anything else
    system_prompt="You are a helpful assistant.",
    max_turns=50,
)
```

Note: Don't add `/anthropic` to the base URL - LiteLLM handles the routing automatically.

## Why Use This?

- **Switch providers easily**: Use the same code with OpenAI, Bedrock, Azure, etc.
- **Cost tracking**: LiteLLM tracks spending across all your agent conversations
- **Rate limiting**: Set budgets and limits on your agent usage
- **Load balancing**: Distribute requests across multiple API keys or regions
- **Fallbacks**: Automatically retry with a different model if one fails

## Troubleshooting

**Connection errors?**
- Make sure LiteLLM is running: `litellm --model your-model`
- Check the URL is correct (default: `http://localhost:4000`)

**Authentication errors?**
- Verify your LiteLLM API key is correct
- Make sure the model is configured in your LiteLLM setup

**Model not found?**
- Check the model name matches what's in your LiteLLM config
- Run `litellm --model your-model` to test it works

**Agent with MCP stuck or failing?**
- The MCP server might not be available at `http://localhost:4000/mcp/deepwiki2`
- Try disabling MCP: `USE_MCP=false python agent_with_mcp.py`
- Or use the basic agent: `python main.py`

## Learn More

- [LiteLLM Docs](https://docs.litellm.ai/)
- [Claude Agent SDK](https://github.com/anthropics/anthropic-agent-sdk)
- [LiteLLM Proxy Guide](https://docs.litellm.ai/docs/proxy/quick_start)
