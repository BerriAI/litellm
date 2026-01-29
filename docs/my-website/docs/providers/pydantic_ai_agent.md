import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Pydantic AI Agents

Call Pydantic AI Agents via LiteLLM's A2A Gateway.

| Property | Details |
|----------|---------|
| Description | Pydantic AI agents with native A2A support via the `to_a2a()` method. LiteLLM provides fake streaming support for agents that don't natively stream. |
| Provider Route on LiteLLM | A2A Gateway |
| Supported Endpoints | `/v1/a2a/message/send` |
| Provider Doc | [Pydantic AI Agents ↗](https://ai.pydantic.dev/agents/) |

## LiteLLM A2A Gateway

All Pydantic AI agents need to be exposed as A2A agents using the `to_a2a()` method. Once your agent server is running, you can add it to the LiteLLM Gateway.

### 1. Setup Pydantic AI Agent Server

LiteLLM requires Pydantic AI agents to follow the [A2A (Agent-to-Agent) protocol](https://github.com/google/A2A). Pydantic AI has native A2A support via the `to_a2a()` method, which exposes your agent as an A2A-compliant server.

#### Install Dependencies

```bash
pip install pydantic-ai fasta2a uvicorn
```

#### Create Agent

```python title="agent.py"
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o-mini', instructions='Be helpful!')

@agent.tool_plain
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@agent.tool_plain  
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Native A2A server - Pydantic AI handles it automatically
app = agent.to_a2a()
```

#### Run Server

```bash
uvicorn agent:app --host 0.0.0.0 --port 9999
```

Server runs at `http://localhost:9999`

### 2. Navigate to Agents

From the sidebar, click "Agents" to open the agent management page, then click "+ Add New Agent".

### 3. Select Pydantic AI Agent Type

Click "A2A Standard" to see available agent types, then select "Pydantic AI".

![Select A2A Standard](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/1055acb1-064b-4465-8e6a-8278291bc661/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=395,147)

![Select Pydantic AI](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/0998e38c-8534-40f1-931a-be96c2cae0ad/ascreenshot.jpeg?tl_px=0,52&br_px=2201,1283&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=421,277)

### 4. Configure the Agent

Fill in the following fields:

- **Agent Name** - A unique identifier for your agent (e.g., `test-pydantic-agent`)
- **Agent URL** - The URL where your Pydantic AI agent is running. We use `http://localhost:9999` because that's where we started our Pydantic AI agent server in the previous step.

![Enter Agent Name](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/8cf3fbde-05f3-48d1-81b6-6f857bd6d360/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=443,225)

![Configure Agent Name](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/fb555808-4761-4c49-a415-200ac1bdb525/ascreenshot.jpeg?tl_px=0,0&br_px=2617,1463&force_format=jpeg&q=100&width=1120.0)

![Enter Agent URL](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/303eae61-4352-4fb0-a537-806839c234ba/ascreenshot.jpeg?tl_px=0,212&br_px=2201,1443&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=456,277)

### 5. Create Agent

Click "Create Agent" to save your configuration.

![Create Agent](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/914f3367-df7d-4244-bd4d-e99ce0a6193a/ascreenshot.jpeg?tl_px=416,438&br_px=2618,1669&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=690,277)

### 6. Test in Playground

Go to "Playground" in the sidebar to test your agent.

![Go to Playground](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/c73c9f3b-22af-4105-aafa-2d34c4986ef3/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=44,97)

### 7. Select A2A Endpoint

Click the endpoint dropdown and search for "a2a", then select `/v1/a2a/message/send`.

![Click Endpoint Dropdown](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/196d97ac-bcba-47f0-9880-97b80250e00c/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=261,230)

![Search for A2A](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/26b68f21-29f9-4c4c-b8b5-d2e11cbfd14a/ascreenshot.jpeg?tl_px=0,0&br_px=2617,1463&force_format=jpeg&q=100&width=1120.0)

![Select A2A Endpoint](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/41576fb1-d385-4fb2-84e9-142dd7fe5181/ascreenshot.jpeg?tl_px=0,0&br_px=2201,1230&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=307,270)

### 8. Select Your Agent and Send a Message

Pick your Pydantic AI agent from the dropdown and send a test message.

![Click Agent Dropdown](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/a96d7967-3d54-4cbf-bd3e-b38f1be9df76/ascreenshot.jpeg?tl_px=0,54&br_px=2201,1285&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=274,277)

![Select Agent](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/e05a5a6e-d044-4480-b94e-7c03cfb92ac5/ascreenshot.jpeg?tl_px=0,113&br_px=2201,1344&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=290,277)

![Send Message](https://ajeuwbhvhr.cloudimg.io/https://colony-recorder.s3.amazonaws.com/files/2025-12-16/29162702-968a-401a-aac1-c844bfc5f4a3/ascreenshot.jpeg?tl_px=91,653&br_px=2292,1883&force_format=jpeg&q=100&width=1120.0&wat=1&wat_opacity=0.7&wat_gravity=northwest&wat_url=https://colony-recorder.s3.us-west-1.amazonaws.com/images/watermarks/FB923C_standard.png&wat_pad=524,436)


## Further Reading

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Pydantic AI Agents](https://ai.pydantic.dev/agents/)
- [A2A Agent Gateway](../a2a.md)
- [A2A Cost Tracking](../a2a_cost_tracking.md)
