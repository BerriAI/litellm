import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Claude Agent SDK with LiteLLM

Use Anthropic's Claude Agent SDK with any LLM provider through LiteLLM Proxy.

The Claude Agent SDK provides a high-level interface for building AI agents. By pointing it to LiteLLM, you can use the same agent code with OpenAI, Bedrock, Azure, Vertex AI, or any other provider.

## Quick Start

### 1. Install Dependencies

```bash
pip install claude-agent-sdk litellm
```

### 2. Start LiteLLM Proxy

<Tabs>
<TabItem value="bedrock" label="AWS Bedrock">

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: bedrock-claude-sonnet-4
    litellm_params:
      model: bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0
      aws_region_name: us-east-1
```

```bash
litellm --config config.yaml
```

</TabItem>
<TabItem value="openai" label="OpenAI">

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY
```

```bash
litellm --config config.yaml
```

</TabItem>
<TabItem value="azure" label="Azure OpenAI">

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: azure-gpt-4
    litellm_params:
      model: azure/gpt-4-deployment
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
```

```bash
litellm --config config.yaml
```

</TabItem>
</Tabs>

### 3. Point Agent SDK to LiteLLM

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

:::info
**Important:** Don't add `/anthropic` to the base URL. LiteLLM handles routing automatically.
:::

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
