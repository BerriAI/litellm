import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Agent Usage

Track and visualize agent-level spend directly in the dashboard. Monitor agent usage analytics, spend logs, and activity metrics to understand how your A2A agents are consuming LLM resources.

This feature is **available in v1.80.10-stable and above**.

## Overview

Agent Usage automatically tracks spend and usage for individual A2A agents when you invoke them through the LiteLLM proxy. The agent ID is automatically extracted from your requests, so no manual configuration is required. This allows you to:

- Track spend per agent automatically
- View agent-level usage analytics in the Admin UI
- Filter spend logs and activity metrics by agent ID
- Monitor agent usage patterns and trends
- Understand which agents consume the most resources

<Image img={require('../../img/agent_usage.png')} />

## How to Track Spend

Spend tracking is **automatic** when using A2A endpoints. The agent ID is automatically extracted from the request URL and associated with all spend from that agent invocation. No manual configuration is required.

### Example using A2A SDK

When you invoke an agent through the A2A SDK, spend is automatically tracked:

```python showLineNumbers title="Automatic spend tracking with A2A SDK"
import httpx
from uuid import uuid4
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

LITELLM_BASE_URL = "http://0.0.0.0:4000"
LITELLM_VIRTUAL_KEY = "sk-1234"
agent_id = "agent-123"  # 👈 Agent ID automatically tracked

async with httpx.AsyncClient(headers={"Authorization": f"Bearer {LITELLM_VIRTUAL_KEY}"}) as client:
    # Get agent card and create A2A client
    base_url = f"{LITELLM_BASE_URL}/a2a/{agent_id}"
    resolver = A2ACardResolver(httpx_client=client, base_url=base_url)
    agent_card = await resolver.get_agent_card()
    a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)

    # Send message
    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "What's the status of my order?"}],
                "messageId": uuid4().hex,
            }
        ),
    )
    response = await a2a_client.send_message(request)
```

The agent ID (`agent-123`) is automatically extracted from the URL path (`/a2a/{agent_id}`) and tracked. All spend from this agent invocation will be associated with this agent ID.

## How to View Spend

### View Spend in Admin UI

Navigate to the Agent Usage tab in the Admin UI to view agent-level spend analytics:

#### 1. Access Agent Usage

Go to the Usage page in the Admin UI (`PROXY_BASE_URL/ui/?login=success&page=new_usage`) and click on the **Agent Usage** tab.

<Image img={require('../../img/agent_usage_ui_navigation.png')} />

#### 2. View Agent Analytics

The Agent Usage dashboard provides:

- **Total spend per agent**: View aggregated spend across all agents
- **Daily spend trends**: See how agent spend changes over time
- **Model usage breakdown**: Understand which models each agent uses
- **Activity metrics**: Track requests, tokens, and success rates per agent

<Image img={require('../../img/agent_usage_analytics.png')} />

#### 3. Filter by Agent

Use the agent filter dropdown to view spend for specific agents:

- Select one or more agent IDs from the dropdown
- View filtered analytics, spend logs, and activity metrics
- Compare spend across different agents

<Image img={require('../../img/agent_usage_filter.png')} />

## Use Cases

### Agent Cost Attribution

Track spend per agent to understand resource consumption:

- Monitor individual agent usage
- Identify high-cost agents
- Optimize agent configurations based on usage patterns

### Agent Performance Monitoring

Understand how different agents use your service:

- Analyze usage patterns across agents
- Track request success rates per agent
- Identify agents that may need optimization

### Resource Planning

Plan infrastructure and budgets based on agent usage:

- Forecast costs based on agent usage trends
- Allocate resources to high-usage agents
- Set budgets and alerts per agent

---

## Related Features

- [A2A Agent Gateway](../a2a.md) - Invoke and manage A2A agents through LiteLLM
- [Cost Tracking](./cost_tracking.md) - Comprehensive cost tracking and analytics
- [Customer Usage](./customer_usage.md) - Track end-user spend and usage
