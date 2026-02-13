import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# AG2 Agents

Call AG2 agents via LiteLLM's A2A Gateway.

| Property | Details |
|----------|---------|
| Description | AG2 agents exposed over the A2A (Agent-to-Agent) protocol. |
| Provider Route on LiteLLM | A2A Gateway |
| Supported Endpoints | `/v1/a2a/message/send` |
| Provider Doc | [AG2 A2A Support ↗](https://docs.ag2.ai/latest/docs/user-guide/a2a/) |

## LiteLLM A2A Gateway

AG2 includes native A2A support using `A2aAgentServer`. Once your AG2 A2A server is running, you can add it to the LiteLLM Gateway.

### 1. Setup AG2 Agent Server

LiteLLM requires AG2 agents to follow the [A2A (Agent-to-Agent) protocol](https://github.com/google/A2A). AG2 provides a built-in A2A server via `A2aAgentServer`.

#### Install Dependencies (AG2 0.11)

```bash
pip install "ag2[openai]>=0.11,<0.12" uvicorn
```

#### Create Agent + A2A Server

```python title="ag2_agent.py"
from autogen import ConversableAgent, LLMConfig
from autogen.a2a import A2aAgentServer

llm_config = LLMConfig({ "model": "gpt-4o-mini" })

agent = ConversableAgent(
    name="ag2_agent",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

server = A2aAgentServer(
    agent,
    url="http://0.0.0.0:8000"
).build()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="0.0.0.0", port=8000)
```

Server runs at `http://localhost:8000`

### 2. Navigate to Agents

From the sidebar, click "Agents" to open the agent management page, then click "+ Add New Agent".

### 3. Select AG2 Agent Type

Click "A2A Standard" to see available agent types, then select "AG2".

### 4. Configure the Agent

Fill in the following fields:

- **Agent Name** - A unique identifier for your agent (e.g., `ag2-support-agent`)
- **URL** - The base URL where your AG2 A2A server is running (e.g., `http://localhost:8000`)

### 5. Create Agent

Click "Create Agent" to save your configuration.

### 6. Test in Playground

Go to "Playground" in the sidebar, select the `/v1/a2a/message/send` endpoint, and send a test message to your AG2 agent.

## Further Reading

- [AG2 Documentation](https://docs.ag2.ai/latest/)
- [AG2 A2A Guide](https://docs.ag2.ai/latest/docs/user-guide/a2a/)
- [A2A Agent Gateway](../a2a.md)
- [A2A Cost Tracking](../a2a_cost_tracking.md)
