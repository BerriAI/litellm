import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Claude Agent SDK with LiteLLM

Use Anthropic's Claude Agent SDK with any LLM provider through LiteLLM Proxy.

The Claude Agent SDK provides a high-level interface for building AI agents. By pointing it to LiteLLM, you can use the same agent code with OpenAI, Bedrock, Azure, Vertex AI, or any other provider.

## Quick Start

### 1. Install Dependencies

```bash
pip install claude-agent-sdk
```

### 2. Start LiteLLM Proxy

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: bedrock-claude-sonnet-3.5
    litellm_params:
      model: "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0"
      aws_region_name: "us-east-1"

  - model_name: bedrock-claude-sonnet-4
    litellm_params:
      model: "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"
      aws_region_name: "us-east-1"

  - model_name: bedrock-claude-sonnet-4.5
    litellm_params:
      model: "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
      aws_region_name: "us-east-1"

  - model_name: bedrock-claude-opus-4.5
    litellm_params:
      model: "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"
      aws_region_name: "us-east-1"

  - model_name: bedrock-nova-premier
    litellm_params:
      model: "bedrock/amazon.nova-premier-v1:0"
      aws_region_name: "us-east-1"
```

```bash
litellm --config config.yaml
```

### 3. Point Agent SDK to LiteLLM

| Environment Variable | Value | Description |
|---------------------|-------|-------------|
| `ANTHROPIC_BASE_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `ANTHROPIC_API_KEY` | `sk-1234` | Your LiteLLM API key (not Anthropic key) |

```python title="agent.py" showLineNumbers
import os
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

# Point to LiteLLM proxy (not Anthropic)
os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:4000"
os.environ["ANTHROPIC_API_KEY"] = "sk-1234"  # Your LiteLLM key

# Configure agent with any model from your config
options = ClaudeAgentOptions(
    system_prompt="You are a helpful AI assistant.",
    model="bedrock-claude-sonnet-4",  # Use any model from config.yaml
    max_turns=20,
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("What is LiteLLM?")
    
    async for msg in client.receive_response():
        if hasattr(msg, 'content'):
            for content_block in msg.content:
                if hasattr(content_block, 'text'):
                    print(content_block.text, end='', flush=True)
```



## Why Use LiteLLM with Agent SDK?

| Feature | Benefit |
|---------|---------|
| **Multi-Provider** | Use the same agent code with OpenAI, Bedrock, Azure, Vertex AI, etc. |
| **Cost Tracking** | Track spending across all agent conversations |
| **Rate Limiting** | Set budgets and limits on agent usage |
| **Load Balancing** | Distribute requests across multiple API keys or regions |
| **Fallbacks** | Automatically retry with different models if one fails |

## Complete Example

See our [cookbook example](https://github.com/BerriAI/litellm/tree/main/cookbook/anthropic_agent_sdk) for a complete interactive CLI agent that:
- Streams responses in real-time
- Switches between models dynamically
- Fetches available models from the proxy

```bash
# Clone and run the example
git clone https://github.com/BerriAI/litellm.git
cd litellm/cookbook/anthropic_agent_sdk
pip install -r requirements.txt
python main.py
```

## Related Resources

- [Claude Agent SDK Documentation](https://github.com/anthropics/anthropic-agent-sdk)
- [LiteLLM Proxy Quick Start](../proxy/quick_start)
- [Complete Cookbook Example](https://github.com/BerriAI/litellm/tree/main/cookbook/anthropic_agent_sdk)
